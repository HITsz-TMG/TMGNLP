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

def load_pairwise_data(prev_data, gen_data, blender, output_path):
    conv1 = []
    conv2 = []
    for prev in prev_data:
        prompt = prev["prompt"]
        prev_chosen = prev["chosen"]
        curr_chosen = gen_data[prompt]["output_1"]     
        conv1.append([
            {
                "content": prompt,
                "role": "USER"
            },
            {
                "content": prev_chosen,
                "role": "ASSISTANT"
            }
        ])

        conv2.append([
            {
                "content": prompt,
                "role": "USER"
            },
            {
                "content": curr_chosen,
                "role": "ASSISTANT"
            }
        ])

    comparison_results = blender.compare_conversations(conv1, conv2, batch_size=32)

    outputs = []
    for idx, comparison_result in enumerate(comparison_results):
        if comparison_result:
            output_sample = {
                "prompt": conv1[idx][0]['content'],
                "chosen": conv1[idx][1]['content'],
                "rejected": conv2[idx][1]['content']
            }
        else:
            output_sample = {
                "prompt": conv1[idx][0]['content'],
                "chosen": conv2[idx][1]['content'],
                "rejected": conv1[idx][1]['content']
            }
        outputs.append(output_sample)

    save_jsonl(data=outputs, path=output_path)


if __name__ == '__main__':
    parser = HfArgumentParser((ScriptArguments, ))
    (args,)  = parser.parse_args_into_dataclasses()

    blender = llm_blender.Blender()
    blender.loadranker(args.blender_model_path) # load PairRM
    if 'json' in args.prev_preference_path:
        prev_temp = load_jsonl(args.prev_preference_path)
    else:
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
    if 'json' not in args.prev_preference_path:
        prev_data = []
        for i in range(len(prev_temp['chosen'])):
            prev_data.append({
                "prompt": prev_temp['chosen'][i][0]['content'],
                "chosen": prev_temp['chosen'][i][1]['content'],
                "rejected": prev_temp['rejected'][i][1]['content']
            })
    else:
        prev_data = prev_temp

    load_pairwise_data(prev_data=prev_data, gen_data=curr_data, blender=blender, output_path=args.output_path)
