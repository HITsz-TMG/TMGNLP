from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import HfArgumentParser, AutoTokenizer

from utils.data_utils import save_jsonl

from vllm import LLM, SamplingParams

import os


@dataclass
class ScriptArguments:
    ckpt: str = field(default="none")
    tensor_parallel_size: int = field(default=1)
    sanity_check: bool = field(default=True, metadata={"help": "only train on 1000 samples"})
    eval_data_path: str = field(default='None')
    split: str = field(default="test_gen")
    output_data_path: str = field(default="none")
    save_name: str = field(default="none")
    response_num: int = field(default=2)


def preprocess_function(examples):
    new_examples = {
        "instruction": [],
        "prompt": [],
    }
    for prompt in examples["prompt"]:
        message = [
            {"role": "user", "content": prompt}
        ]
        new_examples["prompt"].append(message)
        new_examples["instruction"].append(prompt)
    return new_examples


def get_datasets(data_path, split, sanity_check: bool = False):
    print("load dataset {}".format(data_path))
    dataset = load_dataset(data_path, split=split)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 24)))
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    return dataset

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments))
    (args,) = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    ckpt = os.path.basename(args.ckpt)
    
    if args.save_name == 'none':
        save_name = os.path.join(args.output_data_path, "{}.jsonl".format(ckpt))
    else:
        save_name = os.path.join(args.output_data_path, args.save_name)
    if os.path.exists(save_name):
        print("{} exists.".format(save_name))
        exit(0)

    # ################
    # # Dataset
    # ################
    model_ckpt = args.ckpt
    model = LLM(model=model_ckpt, tokenizer=args.ckpt, tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    dataset = get_datasets(args.eval_data_path, split=args.split, sanity_check=args.sanity_check)
    inputs = [tokenizer.apply_chat_template(i["prompt"], add_generation_prompt=True, tokenize=False) for i in dataset]

    sampling_params = SamplingParams(
        max_tokens=2048, 
        n=args.response_num, best_of=args.response_num, 
        top_k=50, top_p=0.9, temperature=0.7,
        detokenize=False
    )

    outputs = model.generate(inputs, sampling_params, use_tqdm=True)
    output_data = []
    for output, input in zip(outputs, dataset):
        data_item = {
            "prompt": output.prompt,
            "instruction": input['instruction']
        }
        for i in range(len(output.outputs)):
            data_item['output_{}'.format(i+1)] = tokenizer.decode(output.outputs[i].token_ids, skip_special_tokens=True)
        output_data.append(data_item)
    save_jsonl(data=output_data, path=save_name)
