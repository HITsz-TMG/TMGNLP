from vllm import LLM, SamplingParams
from transformers import HfArgumentParser, AutoTokenizer
from dataclasses import dataclass, field
from datasets import load_dataset
import os
from tqdm import tqdm
from utils.data_utils import save_jsonl

@dataclass
class ScriptArguments:
    model_path: str = field(default="none")
    sanity_check: bool = field(default=False, metadata={"help": "only train on 1000 samples"})
    eval_data_path: str = field(default='None')
    split: str = field(default="test_gen")
    output_data_path: str = field(default="none")

def preprocess_function(examples):
    new_examples = {
        "problem": [],
        "gt_answer": []
    }
    for problem, answer in zip(examples['problem'], examples['answer']):
        content = problem
        message = [
            {"role": "user", "content": content}
        ]
        new_examples["problem"].append(message)
        new_examples['gt_answer'].append(answer)
    return new_examples


def get_datasets(data_path, split, sanity_check: bool = False):
    print("load dataset {}".format(data_path))
    dataset = load_dataset(data_path, split=split)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 2)))
    dataset = dataset.map(
        preprocess_function,
        batched=True, 
        num_proc=4,
    )
    return dataset


def sample_paths(model, tokenizer, dataset):
    sampling_params = SamplingParams(
        n=256,
        max_tokens=1536
    )
    inputs = [
        tokenizer.apply_chat_template(
            i['problem'], 
            add_generation_prompt=True, 
            tokenize=False
        ) 
        for i in dataset
    ]
    inputs = inputs[:(len(inputs)//2)]
    print("generating sampled data..")
    outputs = model.generate(inputs, sampling_params, use_tqdm=True)
    sampled_data = []
    for output, input in zip(outputs, dataset):
        sampled_data.append({
            "problem": input["problem"][0]['content'],
            "answers": [item.text for item in output.outputs],
            "gt_answer": input['gt_answer']
        })
    return sampled_data

if __name__ == '__main__':
    parser = HfArgumentParser((ScriptArguments))
    (args,) = parser.parse_args_into_dataclasses()
    model = LLM(model=args.model_path, tokenizer=args.model_path, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = get_datasets(args.eval_data_path, split=args.split, sanity_check=args.sanity_check)
    sampled_data = sample_paths(model, tokenizer, dataset)
    # save_jsonl(sampled_data, path=args.output_data_path)
