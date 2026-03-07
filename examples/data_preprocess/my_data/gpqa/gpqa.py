# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GPQA dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default="data/hf_data/gpqa", help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="data/parquet_data/gpqa", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    dataset = datasets.load_dataset(
        'json',
        data_files={
            'test': os.path.join(local_dataset_path, 'gpqa_test.jsonl'),
        },
    )
    test_dataset = dataset['test']

    instruction_following = "\n\nLet's think step by step and output the final answer within \\boxed{}. Please write your final answer in the form of \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("Question")
            options = example.pop("options")
            
            options_str = ""
            for i, option in enumerate(options):
                label = chr(ord('A') + i)
                options_str += f"{label}. {option}\n"

            question = question_raw + "\n\n" + options_str + instruction_following

            # GPQA answer is an index (0, 1, 2, 3), convert to A, B, C, D
            answer_idx = example.pop("answer")

            answer_idx = int(answer_idx)
            answer = chr(ord('A') + answer_idx)

            data = {
                "data_source": 'my_data/gpqa',
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "science",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    test_dataset = test_dataset.remove_columns(
    [col for col in test_dataset.column_names
        if col not in ["data_source", "prompt", "ability", "reward_model", "extra_info"]]
    )

    hdfs_dir = args.hdfs_dir

    local_save_dir = args.local_save_dir

    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
