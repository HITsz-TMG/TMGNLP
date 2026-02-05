import os
import sys
import json
import argparse
import re
import threading
import concurrent.futures
import numpy as np
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI

MODEL_ZOO = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
    'Qwen3-32B': ('Qwen3-32B', 'local')
}

file_write_lock = threading.Lock()

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    task = str(task)
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session', '1', '4', '3', '5']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task in ['temporal-reasoning', '2']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        else:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        prompt = template.format(question, answer, response)
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response) 
    return prompt

def evaluate_single_entry(entry, client, metric_model):
    """
    处理单个条目的评估逻辑
    """
    try:
        qtype = entry.get('category')
        q = entry.get('question')
        
        # 处理 golden_answers 列表
        golden_answers = entry.get('golden_answers')
        if isinstance(golden_answers, list) and len(golden_answers) > 0:
            ans = golden_answers[0]
        elif golden_answers:
            ans = golden_answers
        else:
            ans = None

        hyp = entry.get('output')
        
        # 1. 关键元数据缺失检查：如果题目或标准答案缺失，无法评估，直接跳过
        if any(v is None for v in [qtype, q, ans]):
            return None, None, f"Skipping ID {entry.get('id', 'N/A')} due to missing critical fields (category/question/answer)."

        # 2. 优化点：如果模型输出为空，直接判定为 False，不调用 API
        if not hyp or (isinstance(hyp, str) and not hyp.strip()):
            entry['autoeval_label'] = {
                'model': 'rule-based-empty-check',
                'label': False,
                'note': 'empty_output'
            }
            return entry, qtype, None

        # 3. 正常 API 评估流程
        abstention_flag = '_abs' in str(entry.get('id', ''))
        prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention=abstention_flag)
        
        kwargs = {
            'model': metric_model,
            'messages': [{"role": "user", "content": prompt}],
            'n': 1,
            'temperature': 0,
            'max_tokens': 4096
        }
        
        completion = chat_completions_with_backoff(client, **kwargs)
        eval_response = completion.choices[0].message.content.strip()

        # 处理 Thinking 过程
        if "</think>" in eval_response:
            final_answer_section = eval_response.split("</think>")[-1]
        else:
            final_answer_section = eval_response
        
        clean_text = final_answer_section.strip().lower()
        
        # 正则判定
        if re.search(r'\byes\b', clean_text): 
            label = True
        else:
            label = False # 默认不通过（包括 no 或无法解析的情况）

        entry['autoeval_label'] = {
            'model': metric_model,
            'label': label,
            'raw_eval': eval_response[:200]
        }
        return entry, qtype, None

    except Exception as e:
        return None, None, f"Error processing ID {entry.get('id', 'N/A')}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Concurrent LLM Evaluation Script")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the hypothesis JSON/JSONL file')
    parser.add_argument('--model', type=str, default='Qwen3-32B', help='Metric model name defined in code')
    parser.add_argument('--workers', type=int, default=32, help='Number of concurrent threads')
    parser.add_argument('--api_base', type=str, default="http://192.168.0.129:1050/v1", help='API Base URL')
    parser.add_argument('--api_key', type=str, default="EMPTY", help='API Key')
    
    args = parser.parse_args()

    if args.model not in MODEL_ZOO:
        print(f"Error: Model {args.model} not found in MODEL_ZOO.")
        sys.exit(1)
    
    metric_model_api_name, _ = MODEL_ZOO[args.model]
    metric_client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    # 加载数据
    print(f"Loading data from {args.input_file}...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines: raise ValueError("File is empty")
            try:
                hypotheses = [json.loads(line) for line in lines]
            except json.JSONDecodeError:
                f.seek(0)
                hypotheses = json.load(f)
    except Exception as e:
        print(f"Failed to load file: {e}")
        sys.exit(1)

    result_file = args.input_file + f'.eval-results-{args.model}'
    qtype2acc = {}
    logs = []
    
    print(f"Starting evaluation (Workers: {args.workers}, Model: {metric_model_api_name})")

    with open(result_file, 'w', encoding='utf-8') as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_entry = {
                executor.submit(evaluate_single_entry, entry, metric_client, metric_model_api_name): entry 
                for entry in hypotheses
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_entry), total=len(hypotheses)):
                processed_entry, qtype, error_msg = future.result() 
                
                if error_msg:
                    print(f"\n[SKIP] {error_msg}")
                    continue
                
                if processed_entry:
                    # 线程安全写入
                    with file_write_lock:
                        out_f.write(json.dumps(processed_entry, ensure_ascii=False) + '\n')
                        out_f.flush()
                    
                    logs.append(processed_entry)
                    label = processed_entry['autoeval_label']['label']
                    if qtype not in qtype2acc:
                        qtype2acc[qtype] = []
                    qtype2acc[qtype].append(1 if label else 0)

    # 统计结果
    if not logs:
        print("No valid results computed.")
        return

    overall_accuracy = np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs])
    
    summary_lines = [
        '\n' + '='*30,
        f'Overall Accuracy: {overall_accuracy:.4f}',
        '='*30
    ]
    
    for k in sorted(qtype2acc.keys()):
        v = qtype2acc[k]
        summary_lines.append(f'\t{k}: {np.mean(v):.4f} ({len(v)})')
    
    summary_content = "\n".join(summary_lines)
    print(summary_content)
    
    # 保存 Summary
    summary_file = result_file + '.summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)

    print(f'\nDetailed results: {result_file}')
    print(f'Summary metrics: {summary_file}')

if __name__ == '__main__':
    main()