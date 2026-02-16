import os
import llm_blender
from datasets import Dataset, load_dataset
from utils.data_utils import load_json, save_jsonl, load_jsonl
from transformers import HfArgumentParser
from dataclasses import dataclass, field

import random

from tqdm import tqdm

@dataclass
class ScriptArguments:
    prev_preference_path: str = field(default='None')
    curr_sampled_path: str = field(default='None')
    blender_model_path: str = field(default="None")

def load_pairwise_data(prev_data, gen_data, blender):
    conv1 = []
    conv2 = []

    for prev in prev_data:
        prompt = prev["prompt"]
        # prev_chosen = prev['chosen']
        prev_chosen = random.sample([prev['chosen'], prev['rejected']], 1)[0]
        curr_chosen = gen_data[prompt]
        conv1.append([
            {
                "content": prompt,
                "role": "USER"
            },
            {
                "content": curr_chosen,
                "role": "ASSISTANT"
            }
        ])

        conv2.append([
            {
                "content": prompt,
                "role": "USER"
            },
            {
                "content": prev_chosen,
                "role": "ASSISTANT"
            }
        ])

    comparison_results = blender.compare_conversations(conv1, conv2, batch_size=32, return_logits=False)
    return comparison_results


if __name__ == '__main__':
    parser = HfArgumentParser((ScriptArguments, ))
    (args,)  = parser.parse_args_into_dataclasses()

    blender = llm_blender.Blender()
    blender.loadranker(args.blender_model_path) # load PairRM
    prev_temp = load_dataset(args.prev_preference_path, split="train_prefs")
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

    prev_data = list(prev_data.values())

    for item in prev_data:
        prompt = item['prompt']
        curr_item = curr_data[prompt]
        curr_data[prompt] = random.sample([curr_item["output_1"], curr_item['output_2']], 1)[0]
    
    comparison_results = load_pairwise_data(prev_data=prev_data, 
                                            gen_data=curr_data, 
                                            blender=blender)
    results = sum(comparison_results) / len(comparison_results)
    print(results)
    
