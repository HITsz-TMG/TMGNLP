from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import os

@dataclass
class ScriptArguments:
    train_data_path: str
    eval_data_path: str
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False

def preprocess_function(examples):
    new_examples = {
        "prompt": [tokenizer.apply_chat_template(i[:-1], add_generation_prompt=True, tokenize=False) for i in examples["messages"]],
        "chosen": [i[-1]["content"] for i in examples["chosen"]],
        "rejected": [i[-1]["content"] for i in examples["rejected"]],
    }
    return new_examples

def preprocess_function_2(examples):
    new_examples = {
        "prompt": [tokenizer.apply_chat_template([{'role': 'user', 'content': i}], add_generation_prompt=True, tokenize=False) for i in examples["prompt"]],
        "chosen": examples["chosen"],
        "rejected": examples["rejected"],
    }
    return new_examples


def get_datasets(data_path, split, sanity_check: bool = False):
    dataset = load_dataset(path=data_path, split=split)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    return dataset

def get_datdasets_from_jsonl(data_path, sanity_check: bool = False):
    dataset = load_dataset("json", data_files=[data_path], split='train')
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
    
    dataset = dataset.map(
        preprocess_function_2,
        batched=True,
        num_proc=4,
    )
    return dataset

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    if 'json' in script_args.train_data_path:
        train_dataset = get_datdasets_from_jsonl(script_args.train_data_path)
    else:
        train_dataset = get_datasets(script_args.train_data_path, split="train_prefs")
    eval_dataset = get_datasets(script_args.eval_data_path, split="test_prefs")

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)