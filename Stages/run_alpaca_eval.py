from dataclasses import dataclass, field

from transformers import HfArgumentParser, AutoTokenizer

from utils.data_utils import save_json, load_json

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
    model_name: str = field(default="none")


def preprocess_function(examples):
    new_examples = {
        "instruction": [],
        "prompt": []
    }
    for prompt in examples["instruction"]:
        message = [
            {"role": "user", "content": prompt}
        ]
        new_examples["prompt"].append(message)
        new_examples["instruction"].append(prompt)
    return new_examples


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments))
    (args,) = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    ckpt = os.path.basename(args.ckpt)
    model_name = os.path.basename(args.model_name)
    save_name = os.path.join(args.output_data_path, "{}.json".format(model_name))
    if os.path.exists(save_name):
        print("{} exists.".format(save_name))
        exit(0)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    sampling_params = SamplingParams(max_tokens=4096, best_of=1, top_k=40, top_p=0.9, temperature=0.7, presence_penalty=0.1, frequency_penalty=0.1, detokenize=False)

    # ################
    # # Dataset
    # ################
    eval_dataset = load_json(args.eval_data_path)
    model_ckpt = args.ckpt
    model = LLM(model=model_ckpt, tokenizer=args.ckpt, tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True, enforce_eager=True)
    output_data = []
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": i["instruction"]}], add_generation_prompt=True, tokenize=False
        ) 
        for i in eval_dataset
    ]
    outputs = model.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    for output, input in zip(outputs, eval_dataset):
        output_data.append({
            "instruction": input["instruction"],
            "output": tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True),
            "generator": ckpt
        })
    save_json(data=output_data, path=save_name, indent=4)
