set -x


batch_size=128
n_rollout=16
max_response_length=16384
LR=1e-6
model_path=/path/to/model
save_freq=50
test_freq=50


success_threshold_easy=0.75
success_threshold_hard=0.25
easy_len_penalty_coef=0.2
hard_len_reward_coef=0.2

norm_adv_by_std_in_grpo=False

experiment_name=train_qwen3_14b


# For async rollout mode, dataset should return raw chat.
rollout_mode="async"
rollout_name="sglang" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

deepscaler_train_path=data/parquet_data/deepscaler/train.parquet
gsm8k_test_path=data/parquet_data/gsm8k/test.parquet
math500_test_path=data/parquet_data/math500/test.parquet
aime2024_test_path=data/parquet_data/aime2024/test.parquet
aime2025_test_path=data/parquet_data/aime2025/test.parquet
amc23_test_path=data/parquet_data/amc23/test.parquet
gpqa_test_path=data/parquet_data/gpqa/test.parquet
SVAMP_test_path=data/parquet_data/SVAMP/test.parquet
CommonsenseQA_test_path=data/parquet_data/CommonsenseQA/test.parquet


train_files="['$deepscaler_train_path']"
test_files="['$gsm8k_test_path', '$math500_test_path', '$aime2024_test_path', '$aime2025_test_path', '$amc23_test_path', '$gpqa_test_path', '$SVAMP_test_path', '$CommonsenseQA_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=coda \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.critic_warmup=0 \
    trainer.project_name='coda' \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.validation_data_dir=$validation_data_dir \
    algorithm.norm_adv_by_std_in_grpo=$norm_adv_by_std_in_grpo \
    algorithm.coda.success_threshold_easy=$success_threshold_easy \
    algorithm.coda.success_threshold_hard=$success_threshold_hard \
    algorithm.coda.easy_len_penalty_coef=$easy_len_penalty_coef \
    algorithm.coda.hard_len_reward_coef=$hard_len_reward_coef \
    hydra.run.dir=$rollout_data_dir/hydra_logs \
    hydra.sweep.dir=$rollout_data_dir/hydra_logs \
    +trainer.default_hdfs_dir=null \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=1 $@
