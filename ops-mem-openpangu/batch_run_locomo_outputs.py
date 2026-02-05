import subprocess
import re
import os
import sys
import numpy as np
import time
import uuid
import argparse

def run_dataset(dataset_name, log_dir, run_id, base_save_dir, cmd_base_args):
    """
    运行单个数据集的评估。
    
    Args:
        dataset_name: 数据集名称
        log_dir: 日志目录
        run_id: 唯一运行标识符
        base_save_dir: 基础保存目录
        cmd_base_args: 基础命令参数列表
    
    Returns:
        (dataset_name, f1_score, em_score, llm_judge_score, qa_count, success)
    """
    print(f"\n{'='*50}")
    print(f"[Run ID: {run_id}] Processing dataset: {dataset_name}")
    print(f"{'='*50}")
    
    # 为每个数据集创建唯一的日志文件（包含run_id）
    log_file = os.path.join(log_dir, f"{dataset_name}_{run_id}.log")
    
    # 构建唯一的save_dir路径
    unique_save_dir = os.path.join(base_save_dir, f"{dataset_name}")
    
    # 定义命令参数
    cmd = cmd_base_args + [
        "--dataset", dataset_name,
        "--save_dir", unique_save_dir,
    ]
    
    print(f"[Run ID: {run_id}] Logs will be saved to: {log_file}")
    print(f"[Run ID: {run_id}] Output will be saved to: {unique_save_dir}")
    
    # 获取数据集 QA 数量
    qa_count = 0
    dataset_file = f"locomo/{dataset_name}.json"
    try:
        import json
        with open(dataset_file, 'r') as f:
            data = json.load(f)
            qa_count = len(data)
        print(f"[Run ID: {run_id}] Dataset {dataset_name} has {qa_count} QA pairs.")
    except Exception as e:
        print(f"[Run ID: {run_id}] Error reading dataset file {dataset_file}: {e}")
        qa_count = 0
    
    # 执行命令并将输出重定向到日志文件
    try:
        with open(log_file, "w") as f:
            process = subprocess.run(
                cmd, 
                stdout=f, 
                stderr=subprocess.STDOUT, 
                text=True,
                timeout=10800  # 3小时超时
            )
        
        if process.returncode != 0:
            print(f"[Run ID: {run_id}] Error executing for {dataset_name}. Check log file for details.")
            return dataset_name, 0.0, 0.0, 0.0, qa_count, False
    except subprocess.TimeoutExpired:
        print(f"[Run ID: {run_id}] Timeout executing for {dataset_name}.")
        return dataset_name, 0.0, 0.0, 0.0, qa_count, False
    except Exception as e:
        print(f"[Run ID: {run_id}] Exception executing for {dataset_name}: {e}")
        return dataset_name, 0.0, 0.0, 0.0, qa_count, False
    
    # 从日志文件中解析评估指标
    f1_score = 0.0
    em_score = 0.0
    llm_judge_score = 0.0
    found_f1 = False
    found_em = False
    found_llm_judge = False
    
    with open(log_file, "r") as f:
        content = f.read()
        # 查找完整的评估结果行，例如: Evaluation results for QA: {'ExactMatch': 0.6, 'F1': 0.6667, 'LLMJudge': 0.7}
        eval_pattern = r"Evaluation results for QA:\s*\{[^}]+\}"
        eval_matches = re.findall(eval_pattern, content)
        if eval_matches:
            # 解析最后一次出现的评估结果
            last_eval = eval_matches[-1]
            # 提取各个指标
            em_match = re.search(r"['\"]ExactMatch['\"]:\s*([\d\.]+)", last_eval)
            f1_match = re.search(r"['\"]F1['\"]:\s*([\d\.]+)", last_eval)
            llm_judge_match = re.search(r"['\"]LLMJudge['\"]:\s*([\d\.]+)", last_eval)
            
            if em_match:
                em_score = float(em_match.group(1))
                found_em = True
            if f1_match:
                f1_score = float(f1_match.group(1))
                found_f1 = True
            if llm_judge_match:
                llm_judge_score = float(llm_judge_match.group(1))
                found_llm_judge = True
    
    if found_f1 or found_em:
        metrics_str = []
        if found_em:
            metrics_str.append(f"EM={em_score:.4f}")
        if found_f1:
            metrics_str.append(f"F1={f1_score:.4f}")
        if found_llm_judge:
            metrics_str.append(f"LLMJudge={llm_judge_score:.4f}")
        print(f"[Run ID: {run_id}] Finished {dataset_name}. {' '.join(metrics_str)} (Count: {qa_count})")
    else:
        print(f"[Run ID: {run_id}] Warning: Could not find evaluation scores in log for {dataset_name}")
    
    return dataset_name, f1_score, em_score, llm_judge_score, qa_count, True

def generate_run_id():
    """生成唯一的运行标识符"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # 使用UUID的前8位
    return f"{timestamp}_{unique_id}"

def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for locomo datasets")
    parser.add_argument('--run_id', type=str, default=None,
                        help='Unique run ID for this batch. If not provided, will be auto-generated.')
    parser.add_argument('--base_save_dir', type=str, default='outputs_locomo',
                        help='Base directory for saving outputs (default: outputs_locomo)')
    parser.add_argument('--llm_base_url', type=str, default='http://localhost:9000/v1',
                        help='LLM base URL (default: http://localhost:9000/v1)')
    parser.add_argument('--llm_name', type=str, default='pangu_embedded_7b',
                        help='LLM name (default: pangu_embedded_7b)')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2',
                        help='Embedding model name (default: nvidia/NV-Embed-v2)')
    
    args = parser.parse_args()
    
    # 生成或使用提供的run_id
    run_id = args.run_id if args.run_id else generate_run_id()
    
    # 设置日志目录
    base_log_dir = "log_episodic"
    batch_log_dir = os.path.join(base_log_dir, f"batch_locomo_{run_id}")
    os.makedirs(batch_log_dir, exist_ok=True)
    
    # 创建输出目录
    base_save_dir = args.base_save_dir
    os.makedirs(base_save_dir, exist_ok=True)
    
    print(f"Starting batch evaluation for locomo datasets.")
    print(f"Run ID: {run_id}")
    print(f"Batch logs directory: {batch_log_dir}")
    print(f"Base save directory: {base_save_dir}")
    
    # 构建基础命令参数（不包含dataset和save_dir，这些会在run_dataset中添加）
    cmd_base_args = [
        "python", "main_debug_episodic_reranker.py",
        "--llm_base_url", args.llm_base_url,
        "--llm_name", args.llm_name,
        "--embedding_name", args.embedding_name,
        "--generate_episodic_memory", "true",
        "--use_episodic_memory", "true",
        "--episodic_memory_batch_size", "2",
        "--related_chunks_top_k", "5",
        "--related_chunks_llm_filter", "true",
        "--qa_top_k", "10"  # 设置 qa_top_k 为 10
    ]
    
    datasets = [f"locomo_{i}" for i in range(10)]
    
    results = {}
    total_qa_count = 0
    weighted_f1_sum = 0.0
    weighted_em_sum = 0.0
    weighted_llm_judge_sum = 0.0
    
    # 串行执行所有数据集
    print("\nExecuting datasets sequentially...")
    for dataset in datasets:
        dataset_name, f1, em, llm_judge, count, success = run_dataset(
            dataset, batch_log_dir, run_id, base_save_dir, cmd_base_args
        )
        results[dataset_name] = {"f1": f1, "em": em, "llm_judge": llm_judge, "count": count, "success": success}
        if success:
            weighted_f1_sum += f1 * count
            weighted_em_sum += em * count
            weighted_llm_judge_sum += llm_judge * count
            total_qa_count += count
    
    # 打印结果汇总
    print(f"\n{'='*50}")
    print("Final Results Summary")
    print(f"{'='*50}")
    print(f"Run ID: {run_id}")
    
    for dataset in datasets:
        res = results.get(dataset, {"f1": 0.0, "em": 0.0, "llm_judge": 0.0, "count": 0, "success": False})
        f1 = res["f1"]
        em = res.get("em", 0.0)
        llm_judge = res.get("llm_judge", 0.0)
        count = res["count"]
        success = res.get("success", False)
        status = "✓" if success else "✗"
        print(f"{status} {dataset}: F1={f1:.4f}, EM={em:.4f}, LLMJudge={llm_judge:.4f}, Count={count}")
    
    print(f"{'-'*20}")
    
    # 计算加权平均
    if total_qa_count > 0:
        final_weighted_f1 = weighted_f1_sum / total_qa_count
        final_weighted_em = weighted_em_sum / total_qa_count
        final_weighted_llm_judge = weighted_llm_judge_sum / total_qa_count
        print(f"Total QA Pairs: {total_qa_count}")
        print(f"Weighted Average F1 Score: {final_weighted_f1:.4f}")
        print(f"Weighted Average EM Score: {final_weighted_em:.4f}")
        print(f"Weighted Average LLM Judge Score: {final_weighted_llm_judge:.4f}")
    else:
        print("Error: Total QA count is 0.")
        final_weighted_f1 = 0.0
        final_weighted_em = 0.0
        final_weighted_llm_judge = 0.0
    
    print(f"{'='*50}")
    
    # 保存汇总结果到文件
    summary_file = os.path.join(batch_log_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Batch Evaluation Summary\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nResults:\n")
        for dataset in datasets:
            res = results.get(dataset, {"f1": 0.0, "em": 0.0, "llm_judge": 0.0, "count": 0, "success": False})
            f.write(f"{dataset}: F1={res['f1']:.4f}, EM={res.get('em', 0.0):.4f}, LLMJudge={res.get('llm_judge', 0.0):.4f}, Count={res['count']}, Success={res.get('success', False)}\n")
        f.write(f"\nTotal QA Pairs: {total_qa_count}\n")
        f.write(f"Weighted Average F1 Score: {final_weighted_f1:.4f}\n")
        f.write(f"Weighted Average EM Score: {final_weighted_em:.4f}\n")
        f.write(f"Weighted Average LLM Judge Score: {final_weighted_llm_judge:.4f}\n")
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
