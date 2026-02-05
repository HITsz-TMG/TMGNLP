import argparse
import copy
import sys

from torch import nn
from torch.utils.data import DataLoader


sys.path.append('../..')
from src.train.hotpotqa import HotpotQADistractorDataset
from src.train.kuma.kuma import HardKuma


import os
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import wandb


from transformers import AutoConfig, AutoTokenizer


from src.train.passkey_retrieval import get_dataset, MultiplePasskeyRetrievalDataset, get_supervised_dataloader
from src.train.utils import  seed_everything, visualize_2d_tensor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default=None
    )
    parser.add_argument("--config_name", type=str, default=None)

    # train params
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="datasets/booksum.jsonl.zst",
    )
    parser.add_argument("--dataset_format", type=str, default="multiple_passkey")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--context_length_min", type=int, default=1024)
    parser.add_argument("--context_length_max", type=int, default=4096)
    parser.add_argument("--context_lengths_num_intervals", type=int, default=20)
    parser.add_argument("--depth_ratio_num_intervals", type=int, default=10)
    parser.add_argument("--num_passkeys", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--min_needle_depth_ratio", type=float, default=0)
    parser.add_argument("--max_needle_depth_ratio", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--rope_theta", type=float, default=None)

    parser.add_argument("--passkey_length", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=16384)
    parser.add_argument("--generation_length", type=int, default=256)
    parser.add_argument("--prefilling_chunk_size", type=int, default=4096)

    parser.add_argument("--lamda_init_value", type=float, default=0.5)
    parser.add_argument("--lagrange_lr", type=float, default=0.01)
    parser.add_argument("--a_init_value", type=float, default=1.0)
    parser.add_argument("--b_init_value", type=float, default=1.0)
    parser.add_argument("--desired_density", type=float, default=0.5)
    parser.add_argument("--sparse_radio_train", type=float, default=0.7)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args


def train(
    args, model, train_dataloader, optimizer, scheduler, a, b
):
    model.train()

    pbar = tqdm(range(args.num_steps))

    global_step = 0
    chunk_size = args.prefilling_chunk_size


    s_min = torch.Tensor([-0.1])
    s_max = torch.Tensor([1.1])
    support = [s_min, s_max]
    desired_density = args.desired_density
    c0_ma = torch.tensor(0.0)
    lambda0 = torch.tensor(args.lamda_init_value)
    lagrange_alpha = 0.9
    lagrange_lr = args.lagrange_lr
    lambda_min = 1e-12
    lambda_max = 20.0


    while True:
        if global_step >= args.num_steps:
            break
        for step, batch in enumerate(train_dataloader):

            a_clamp = a.clamp(1e-6, 100.0)  # extreme values could result in NaNs
            b_clamp = b.clamp(1e-6, 100.0)  # extreme values could result in NaNs

            sampler = HardKuma([a_clamp, b_clamp], support=[support[0].to(a.device), support[1].to(b.device)])

            sample_z = sampler.sample()

            batch = {k: v.to(f"cuda") for k, v in batch.items()}
            labels = batch["labels"]
            context = batch["input_ids"]

            context_len = context.shape[1]
            labels_len = labels.shape[1]
            if chunk_size > context_len:
                chunk_size = context_len
            past_key_values = None
            for i in range(0, context_len, chunk_size):
                chunk = context[:, i: i + chunk_size]
                output = model(
                    input_ids=chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values

            original_output = model(
                input_ids=labels,
                past_key_values=copy.deepcopy(past_key_values), # deepcopy() for new version transformers
            )

            pruned_output = model(
                input_ids=labels,
                past_key_values=copy.deepcopy(past_key_values),
                full_attention_heads=sample_z,
            )

            original_hidden_states = original_output.last_hidden_state
            pruned_hidden_states = pruned_output.last_hidden_state

            distill_loss = (
                    (original_hidden_states - pruned_hidden_states)
                    .pow(2)
                    .mean(dim=-1)
                    .sum()
                    / labels_len
            )
            pdf0 = sampler.pdf(0.0)
            pdf_nonzero = (1 - pdf0)
            density = pdf_nonzero.mean()

            c0_hat = F.relu(density - desired_density)

            c0_ma = lagrange_alpha * c0_ma + (1 - lagrange_alpha) * c0_hat.item()

            c0 = c0_hat + (c0_ma.detach() - c0_hat.detach())


            lambda0 = lambda0 * torch.exp(lagrange_lr * c0.detach())
            lambda0 = lambda0.clamp(lambda_min, lambda_max)

            reg_loss = lambda0.detach() * c0

            loss = distill_loss + reg_loss

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            p_full = sampler.mean()
            p_full[0, :] = 1

            full_head = (p_full > 0.5).float()

            if not args.disable_wandb:
                fig = visualize_2d_tensor(sample_z)
                fig_p = visualize_2d_tensor(p_full)
                fig_full_heads = visualize_2d_tensor(full_head)

                sample_len = batch["input_ids"].shape[1]
                wandb.log(
                    {
                        "distill_loss": distill_loss.item(),
                        "reg_loss": reg_loss.item(),
                        "density": density.item(),
                        "lamda0": lambda0.item(),
                        "sample_z": fig,
                        "full_head": fig_full_heads,
                        "p_full" : fig_p,
                        "sample_len": sample_len,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

                plt.close(fig)
                plt.close(fig_p)
                plt.close(fig_full_heads)

                pbar.set_description(
                    f"Len={context_len}/{labels_len}|Density={density.item():.3f}|Dloss={distill_loss.item():.3f}|Rloss={reg_loss.item():.3f}|LR={optimizer.param_groups[0]['lr']:.2e}"
                )
                pbar.update(1)

            if args.output_dir is not None and global_step % args.save_steps == 0:

                torch.save(
                    a,
                    os.path.join(
                        args.output_dir,
                        f"a_step={global_step}.pt",
                    ),
                )
                torch.save(
                    b,
                    os.path.join(
                        args.output_dir,
                        f"b_step={global_step}.pt",
                    ),
                )
                torch.save(
                    full_head,
                    os.path.join(
                        args.output_dir,
                        f"full_head_step={global_step}.pt",
                    ),
                )

                os.system(f"rm {args.output_dir}/a_latest.pt")
                os.system(
                    f"cp {args.output_dir}/a_step={global_step}.pt {args.output_dir}/a_latest.pt"
                )

                os.system(f"rm {args.output_dir}/b_latest.pt")
                os.system(
                    f"cp {args.output_dir}/b_step={global_step}.pt {args.output_dir}/b_latest.pt"
                )

                os.system(f"rm {args.output_dir}/full_head_latest.pt")
                os.system(
                    f"cp {args.output_dir}/full_head_step={global_step}.pt {args.output_dir}/full_head_latest.pt"
                )

                # save scheduler and optimizer state
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                    },
                    os.path.join(
                        args.output_dir,
                        f"optimizer_scheduler_state-step={global_step}.pt",
                    ),
                )

                # copy the full_attention_heads and optimizer_scheduler_state to the latest state, replacing the old one
                # remove the previous latest state
                os.system(
                    f"rm {args.output_dir}/optimizer_scheduler_state_latest.pt"
                )
                os.system(
                    f"cp {args.output_dir}/optimizer_scheduler_state-step={global_step}.pt {args.output_dir}/optimizer_scheduler_state_latest.pt"
                )

            if global_step >= args.num_steps:
                break

    pbar.close()

    print("Training finished")
    if args.output_dir is not None:
        torch.save(
            a,
            os.path.join(args.output_dir, "a.pt"),
        )
        torch.save(
            b,
            os.path.join(args.output_dir, "b.pt"),
        )
        torch.save(
            full_head,
            os.path.join(args.output_dir, "full_head.pt"),
        )


def main(args):
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.config_name is not None:
        config = AutoConfig.from_pretrained(args.config_name)
    else:
        config = AutoConfig.from_pretrained(args.model_name)

    if args.rope_theta is not None:
        print(f"Setting rope_theta from {config.rope_theta} to {args.rope_theta}")
        config.rope_theta = args.rope_theta
    if "llama" in args.model_name.lower():
        from src.llama.modeling_llama_train import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(args.model_name, sparse_radio_train=args.sparse_radio_train, torch_dtype=torch.float16).cuda()
    elif "qwen3" in args.model_name.lower():
        from src.qwen3.modeling_qwen3_train import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(args.model_name, sparse_radio_train=args.sparse_radio_train,torch_dtype=torch.float16).cuda()
    elif "qwen" in args.model_name.lower():
        from src.qwen2.modeling_qwen2_train import Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(args.model_name, sparse_radio_train=args.sparse_radio_train,torch_dtype=torch.float16).cuda()
    else:
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = model.model

    for param in model.parameters():
        param.requires_grad = False

    print(model)

    if args.dataset_format == "multiple_passkey":
        haystack_dataset = get_dataset(args.dataset_name, split="train")
        train_dataset = MultiplePasskeyRetrievalDataset(
            haystack_dataset,
            tokenizer,
            max_length=args.max_length,
            min_depth_ratio=args.min_needle_depth_ratio,
            max_depth_ratio=args.max_needle_depth_ratio,
            context_length_min=args.context_length_min,
            context_length_max=args.context_length_max,
            context_lengths_num_intervals=args.context_lengths_num_intervals,
            depth_ratio_num_intervals=args.depth_ratio_num_intervals,
            num_passkeys=args.num_passkeys,
        )
        train_dataloader = get_supervised_dataloader(
            train_dataset, tokenizer, args.batch_size, shuffle=True
        )

    elif args.dataset_format == "hotpotqa":
        train_dataset = HotpotQADistractorDataset(
            args.model_name,
            context_length_min=args.context_length_min,
            context_length_max=args.context_length_max,
            context_lengths_num_intervals=args.context_lengths_num_intervals,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
    else:
        raise ValueError(f"Invalid dataset format: {args.dataset_format}")

    head_num = model.config.num_key_value_heads
    layer_num = model.config.num_hidden_layers
    a = nn.Parameter(torch.ones(layer_num, head_num) * args.a_init_value)
    b = nn.Parameter(torch.ones(layer_num, head_num) * args.b_init_value)

    optimizer = torch.optim.AdamW([a,b], lr=args.lr, weight_decay=0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1,
            max((step + 1) / (args.num_steps // 5), 0.1),
            max((args.num_steps - step) / (args.num_steps // 5), 0.1),
        ),
    )

    experiment_config = vars(args)
    if not args.disable_wandb:
        wandb.init(project="KumaTrain", config=experiment_config)
        if args.exp_name is not None:
            wandb.run.name = args.exp_name

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(experiment_config, f)


    train(
        args,
        model,
        train_dataloader,
        optimizer,
        scheduler,
        a,
        b
    )


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)