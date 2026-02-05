import argparse
import copy

import torch
import os
import sys
sys.path.append('../..')

from utils import seed_everything

from utils import bench_func
from src.utils.cache import StaticCache

def benchmark_e2e(model, tokenizer, args, prefilling_chunk_size, decoding_gen_len):
    model.eval()

    text = ["a\n\n" * args.max_cache_seqlen] * args.batch
    input_len = args.max_cache_seqlen-decoding_gen_len
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")[:, : input_len]
    attention_mask_prefill = inputs["attention_mask"].to("cuda")[:, : input_len]
    print(input_ids.shape)

    print(f"Prefilling chunk size: {prefilling_chunk_size}")

    # pre-filling
    def prefill():
        past_key_values = StaticCache(config=model.config,batch_size=args.batch,max_cache_len=args.max_cache_seqlen,device=model.device,dtype=model.dtype)
        # past_key_values = None
        with torch.no_grad():
            for i in range(0, input_ids.size(1), prefilling_chunk_size):
                input_chunk = input_ids[:, i: i + prefilling_chunk_size]
                attention_mask_chunk = attention_mask_prefill[:, : i + prefilling_chunk_size]
                outputs = model(
                    input_ids=input_chunk,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask_chunk,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        return past_key_values, pred_token_idx


    past_key_values_prefill, pred_token_idx_prefill = prefill()
    print(
        f"Peak memory usage in the pre-filling stage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB"
    )


    def decode():
        past_key_values = copy.deepcopy(past_key_values_prefill)
        pred_token_idx = pred_token_idx_prefill.clone()
        attention_mask = attention_mask_prefill.clone()
        with torch.no_grad():
            for i in range(decoding_gen_len):
                outputs = model(
                    input_ids=pred_token_idx,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)],dim=1)


    gen_latency, gen_memory = bench_func(decode, num_steps=100, num_warmup_steps=10)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, output_file), "w") as f:
            print(f"Average generation time: {gen_latency:.4f} ms", file=f)
            print(f"Average generation time per token: {gen_latency/decoding_gen_len:.4f} ms", file=f)
            print(f"Peak generation memory usage: {gen_memory:.4f} MB", file=f)
            print(f"Context length: {args.max_cache_seqlen}", file=f)
            print(f"Prefilling chunk size: {prefilling_chunk_size}", file=f)
            print(f"Decoding generate length: {decoding_gen_len}", file=f)



def test_lychee(args):
    from transformers import AutoTokenizer
    from src.llama.modeling_llama_block_level import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(args.model,
                                              block_size=args.block_size,
                                              sparse_mode="topk",
                                              mode_value=args.block_num,
                                              batch=args.batch,
                                              max_cache_seqlen=args.max_cache_seqlen,
                                              full_head_path=args.full_head_path,
                                              torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model name or path')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--max_cache_seqlen', type=int, default=32000, help='max sequence length')
    parser.add_argument('--block_size', type=int, default=64, help='block_size')
    parser.add_argument('--block_num', type=int, default=64, help='block_num')
    parser.add_argument("--full_head_path", type=str, default=None)
    args = parser.parse_args()
    seed_everything(42)

    output_dir = "./output"
    output_file = f"batch={args.batch}_context={args.max_cache_seqlen}_block_size={args.block_size}_block_num={args.block_num}.txt"
    model, tokenizer = test_lychee(args)
    benchmark_e2e(model, tokenizer, args, prefilling_chunk_size=512, decoding_gen_len=20)


