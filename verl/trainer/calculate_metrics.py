import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os



def calculate_metrics_mean_sample_once(args):
    prompt, group, n_values, seed = args
    rng = np.random.RandomState(seed)
    
    if 'acc' in group.columns:
        scores = group['acc'].values
    else:
        scores = group['score'].values

    # 尝试计算 response length
    response_lengths = None
    if 'valid_response_length' in group.columns:
        response_lengths = group['valid_response_length'].values

        
    metrics = {}
    
    for n in n_values:
        if n <= len(scores):
            indices = rng.choice(len(scores), size=n, replace=False)
            metrics[f'mean@{n}'] = np.mean(scores[indices])
            
            if response_lengths is not None:
                metrics[f'response_length@{n}'] = np.mean(response_lengths[indices])
            
    return metrics



data_list = ["my_data/aime2025", "my_data/aime2024", "my_data/amc23", "my_data/olympiadbench", "my_data/math500", "my_data/gsm8k", "my_data/deepscaler", "my_data/SVAMP", "my_data/gpqa", "my_data/CommonsenseQA", "my_data/openbookqa"]
dataset_name = "SVAMP"

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from verls jsonl output.")

    parser.add_argument('file_path', nargs='?', default="test_n64_300.jsonl", type=str, help="Path to the .jsonl file")



    parser.add_argument('--n_values', type=int, nargs='+', default=[8], help="List of N values to compute metrics for (e.g. 1 8 64)")
    parser.add_argument('--dataset', type=str, default=f"my_data/{dataset_name}", help="Only process this specific dataset (data_source)")
    args = parser.parse_args()

    print(f"Reading file: {args.file_path}")
    
    # 读取 jsonl
    data = []
    with open(args.file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns found: {df.columns.tolist()}")
    
    if 'data_source' not in df.columns:
        df['data_source'] = 'default'
        
    # 按 data_source 分组处理
    for source_name, source_df in df.groupby('data_source'):
        if args.dataset and source_name != args.dataset:
            continue
            
        print(f"\n{'='*20}")
        print(f"Data Source: {source_name}")
        print(f"Samples: {len(source_df)}")
        
        # 按 input (prompt) 分组
        grouped = source_df.groupby('input')
        unique_prompts = len(grouped)
        print(f"Unique Prompts: {unique_prompts}")
        
        # 准备并行计算任务（只运行一次）
        tasks = []
        for prompt, group in grouped:
            tasks.append((prompt, group, args.n_values, 42))
        
        # 并行计算每个 Prompt 的指标
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(calculate_metrics_mean_sample_once, tasks), total=len(tasks), desc=f"Processing {source_name}"))

        # 聚合结果：计算所有 Prompt 的平均值
        final_metrics = defaultdict(list)
        for res in results:
            for k, v in res.items():
                final_metrics[k].append(v)
        
        # 当前唯一一次运行的平均值
        current_run_means = {}
        for k, v in final_metrics.items():
            current_run_means[k] = np.mean(v)
                
        print(f"\nResults for {source_name}:")
        sorted_keys = sorted(final_metrics.keys(), key=lambda x: (int(x.split('@')[1]), x.split('@')[0]))
        
        print(f"{'Metric':<15} | {'Value':<10}")
        print("-" * 28)
        for k in sorted_keys:
            val = current_run_means[k]
            print(f"{k:<15} | {val:.4f}")

if __name__ == "__main__":
    main()
