import os
import socket
import hydra
import ray
from omegaconf import OmegaConf
from pprint import pprint

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer, hf_processor
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_validate(config)

def run_validate(config):
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        if config.ray_kwargs.get("enable_debug", False):
            default_runtime_env["env_vars"]["RAY_DEBUG_POST_MORTEM"] = "1"
            print(f"enable debug mode")

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Force disable critic and ref policy for validation to save resources
    # We only need Actor and RewardModel
    if config.critic.optim.get("lr", 0) > 0 or config.critic.get("enable", False):
        print("Disabling critic for validation...")
        config.critic.enable = False
    
    # Disable KL loss related flags to avoid initializing RefPolicy
    config.actor_rollout_ref.actor.use_kl_loss = False
    config.algorithm.use_kl_in_reward = False

    # Create a remote wrapper for ValidationTaskRunner
    ValidationTaskRunnerRemote = ray.remote(ValidationTaskRunner)
    runner = ValidationTaskRunnerRemote.remote()
    ray.get(runner.run.remote(config))

class ValidationTaskRunner(TaskRunner):
    def run(self, config):
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config) # Will skip if disabled
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls) # Will skip if disabled

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # For validation we only need val_reward_fn, but RayPPOTrainer init might check reward_fn
        # We pass the same fn or None if allowed. RayPPOTrainer checks if reward_fn is None in fit() for REMAX.
        # But we won't call fit().
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)
        
        # We only need val_dataset
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )

        # RayPPOTrainer requires train_dataset/train_dataloader to be present in init, 
        # so we might need a dummy train_dataset or pass None and fix RayPPOTrainer.
        # RayPPOTrainer._create_dataloader handles None by creating it from config.data.train_files.
        # If we pass empty train_files, it might fail or create empty dataset.
        # Let's try passing val_dataset as train_dataset to satisfy assertions, 
        # or just let it create a small one.
        # Actually RayPPOTrainer assertions:
        # assert len(self.train_dataloader) >= 1
        # So we need a non-empty train dataset.
        # We can use val_dataset as train_dataset to avoid loading train files.
        
        train_dataset = val_dataset 
        train_sampler = create_rl_sampler(config.data, train_dataset)
        from verl.utils.dataset.rl_dataset import collate_fn

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

        print(f"init_workers start")
        trainer.init_workers()
        print(f"init_workers end")

        # Initialize global_steps to 0 in case _load_checkpoint doesn't set it (e.g. if it returns early)
        trainer.global_steps = 0

        # Load checkpoint if specified
        trainer._load_checkpoint()

        print("Starting validation...")
        val_metrics = trainer._validate()
        print("Validation metrics:")
        pprint(val_metrics)
        
        # Optionally save metrics
        if config.trainer.get("validation_data_dir"):
             print(f"Validation results saved to {config.trainer.validation_data_dir}")

if __name__ == "__main__":
    main()
