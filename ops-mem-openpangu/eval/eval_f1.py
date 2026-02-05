import json
import string
import re
from collections import defaultdict
import argparse
import sys


def normalize_answer(s):
    """标准化文本：小写、去除标点、去除冠词"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    from collections import Counter
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def analyze_with_category(json_file):
    print(f"Loading data from {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_file} not found.")
        return

    # 初始化计数器
    total_counts = defaultdict(lambda: 0)
    f1_sums = defaultdict(lambda: 0.0)
    em_sums = defaultdict(lambda: 0.0)

    total_f1_sum = 0.0
    total_em_sum = 0.0
    total_samples = 0

    # print("-" * 80)
    # print(f"{'ID':<4} | {'Cat':<3} | {'F1':<5} | {'EM':<5} | {'Question Summary'}")
    # print("-" * 80)

    for item in data:
        idx = item.get('idx', total_samples)
        question = item.get('question', "")
        prediction = item.get('output', "")
        ground_truths = item.get('golden_answers', [])
        category = item.get('category', 'Unknown') 


        f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)

        total_counts[category] += 1
        f1_sums[category] += f1
        em_sums[category] += em
        
        total_f1_sum += f1
        total_em_sum += em
        total_samples += 1

        # 打印单条详情 (将 float 转换为 int 打印 EM 更直观: 1.0 -> 1, 0.0 -> 0)
        # print(f"{idx:<4} | {category:<3} | {f1:.2f}  | {int(em):<5} | {question[:40]}...")

    print("=" * 80)
    print("Metrics by Category:")
    print(f"{'Cat':<10} | {'Count':<6} | {'Avg F1':<8} | {'Avg EM':<8}")
    print("-" * 60)

    # 4. 输出分类统计结果
    sorted_keys = sorted(total_counts.keys())
    
    category_stats = {}

    for k in sorted_keys:
        count = total_counts[k]
        sum_f1 = f1_sums[k]
        sum_em = em_sums[k]
        
        if count == 0:
            print(f"{str(k):<10} | {0:<6} | {'N/A':<8} | {'N/A':<8}")
        else:
            avg_f1 = round(float(sum_f1) / count, 3)
            avg_em = round(float(sum_em) / count, 3)
            print(f"{str(k):<10} | {count:<6} | {avg_f1:<8.3f} | {avg_em:<8.3f}")
            
            category_stats[k] = {
                "count": count,
                "avg_f1": avg_f1,
                "avg_em": avg_em
            }

    # 5. 计算总体平均分
    if total_samples > 0:
        overall_f1 = round(float(total_f1_sum) / total_samples, 3)
        overall_em = round(float(total_em_sum) / total_samples, 3)
    else:
        overall_f1 = 0
        overall_em = 0

    print("-" * 60)
    print(f"Overall Samples: {total_samples}")
    print(f"Overall Avg F1 : {overall_f1}")
    print(f"Overall Avg EM : {overall_em}")

    results_dict = {
        "model_analysis": {
            "overall": {
                "total_samples": total_samples,
                "f1_score": overall_f1,
                "em_score": overall_em
            },
            "by_category": category_stats,
            "raw_counts": dict(total_counts),
            "raw_f1_sums": dict(f1_sums),
            "raw_em_sums": dict(em_sums)
        }
    }
    
    output_file = json_file + 'evaluation_report.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nDetailed stats saved to {output_file}")

if __name__ == "__main__":
    # 初始化参数解析器
    parser = argparse.ArgumentParser(description="Analyze result file categories.")
    
    parser.add_argument('--input_file', '-i', type=str, 
                        default='./result/locomo10_native.json',
                        help='Path to the input JSON file to analyze')

    args = parser.parse_args()
    
    try:
        print(f"Starting analysis on: {args.input_file}")
        analyze_with_category(args.input_file)
    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error executing analysis: {e}")
        sys.exit(1)