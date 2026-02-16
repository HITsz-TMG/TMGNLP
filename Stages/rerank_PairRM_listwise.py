import os
import llm_blender
from datasets import Dataset, load_dataset
from utils.data_utils import load_json, save_jsonl, load_jsonl
from transformers import HfArgumentParser
from dataclasses import dataclass, field

from tqdm import tqdm

@dataclass
class ScriptArguments:
    prev_preference_path: str = field(default='None')
    curr_sampled_path: str = field(default='None')
    blender_model_path: str = field(default="None")
    output_path: str = field(default="none")
    candidates: int = field(default=2)

def load_listwise_data(curr_data, blender, num_candidates):
    prompts, candidates = [], []
    for key, value in curr_data.items():
        prompts.append(key)
        candidates.append([value['output_{}'.format(i)] for i in range(1, num_candidates+1)])
    comparison_results = blender.rank(prompts, candidates, batch_size=8, return_scores=False)
    return comparison_results


if __name__ == '__main__':
    parser = HfArgumentParser((ScriptArguments, ))
    (args,)  = parser.parse_args_into_dataclasses()

    blender = llm_blender.Blender()
    blender.loadranker(args.blender_model_path) # load PairRM
    if 'json' in args.prev_preference_path:
        prev_temp = load_jsonl(args.prev_preference_path)
    else:
        prev_temp = load_dataset(args.prev_preference_path, split="test_prefs")
    if "json" not in args.curr_sampled_path:
        curr_temp = []
        files = os.listdir(args.curr_sampled_path)
        for file in files:
            curr_temp += load_jsonl(os.path.join(args.curr_sampled_path, file))
    else:
        curr_temp = load_jsonl(args.curr_sampled_path)
    curr_data = {}
    for item in curr_temp:
        curr_data[item["instruction"]] = item
    if 'json' not in args.prev_preference_path:
        prev_data = {}
        for i in tqdm(prev_temp["chosen"]):
            prompt = i[0]['content']
            chosen = i[1]['content']
            prev_data[prompt] = {
                "prompt": prompt,
                "chosen": chosen
            }
        for i in tqdm(prev_temp["rejected"]):
            prompt = i[0]['content']
            rejected = i[1]['content']
            prev_data[prompt]['rejected'] = rejected
    else:
        prev_data = prev_temp
    
    for prompt, item in prev_data.items():
        curr_item = curr_data[prompt]
        for i in range(1, args.candidates+1):
            if 'output_{}'.format(i) not in curr_item.keys():
                curr_item['output_{}'.format(i)] = ""
        curr_item['output_{}'.format(args.candidates+1)] = item['chosen']
        curr_item['output_{}'.format(args.candidates+2)] = item['rejected']
        curr_data[prompt] = curr_item
    ranks = load_listwise_data(curr_data, blender, args.candidates+2)
    output_data = []
    idx = 0
    for key, value in curr_data.items():
        value['ranks'] = ranks[idx].tolist()
        output_data.append(value)
        idx += 1
    save_jsonl(output_data, path=args.output_path)