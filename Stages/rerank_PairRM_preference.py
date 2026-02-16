import os
import llm_blender
from datasets import Dataset, load_dataset
from utils.data_utils import load_json, save_jsonl, load_jsonl
from transformers import HfArgumentParser
from dataclasses import dataclass, field

from tqdm import tqdm

@dataclass
class ScriptArguments:
    data_path: str = field(default='None')
    blender_model_path: str = field(default="None")
    output_path: str = field(default="none")

def run(data, blender, output_path):
    conv1 = []
    conv2 = []
    for instruction, item in data.items():
        conv1.append([
            {
                "content": instruction,
                "role": "USER"
            },
            {
                "content": item['chosen'],
                "role": "ASSISTANT"
            }
        ])

        conv2.append([
            {
                "content": instruction,
                "role": "USER"
            },
            {
                "content": item['rejected'],
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

    if "json" not in args.data_path:
        prev_temp = load_dataset(args.data_path, split='train_prefs')
        data = {}
        for i in tqdm(prev_temp["chosen"]):
            prompt = i[0]['content']
            chosen = i[1]['content']
            data[prompt] = {
                "prompt": prompt,
                "chosen": chosen
            }
        for i in tqdm(prev_temp["rejected"]):
            prompt = i[0]['content']
            rejected = i[1]['content']
            data[prompt]['rejected'] = rejected
    else:
        temp = load_jsonl(args.data_path)
        data = {}
        for item in temp:
            data[item['instruction']] = {
                "chosen": item['output_1'],
                "rejected": item['output_2']
            }

    run(data=data, blender=blender, output_path=args.output_path)
