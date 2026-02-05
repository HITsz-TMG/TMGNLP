import argparse
import os

import numpy as np

import matplotlib.pyplot as plt


def visualize_2d_tensor(tensor):
    tensor = tensor.clone().detach().cpu().float()
    img = np.array(tensor)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04)
    # scale the color to 0-1
    plt.clim(0, 1)
    plt.tight_layout()
    return fig

def parse_device(device: str):
    if "," in device:
        return [int(d) for d in device.split(",")]
    elif device in ["auto", "cpu"]:
        return device
    return f"cuda:{device}"

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="kv_reduction")

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--config_name", type=str, default=None)

    # train params
    parser.add_argument("--dataset_name",type=str,default="",)
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
    parser.add_argument("--sink_size", type=int, default=64)
    parser.add_argument("--recent_size", type=int, default=256)
    parser.add_argument("--deploy_sink_size", type=int, default=None)
    parser.add_argument("--deploy_recent_size", type=int, default=None)
    parser.add_argument("--reg_weight", type=float, default=0.05)
    parser.add_argument("--corr_weight", type=float, default=0.05)
    parser.add_argument("--initial_value", type=float, default=0.5)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--enable_pp", action="store_true")
    parser.add_argument("--enable_tp", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--min_needle_depth_ratio", type=float, default=0)
    parser.add_argument("--max_needle_depth_ratio", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rope_theta", type=float, default=None)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--streaming_attn_implementation", type=str, default="blocksparse"
    )

    parser.add_argument(
        "--supervision",
        type=str,
        default="distill",
        choices=["classify", "distill"],
    )

    # Eval params
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--task", type=str, default="default")
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--passkey_length", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=16384)
    parser.add_argument("--generation_length", type=int, default=256)
    parser.add_argument("--stride_length", type=int, default=256)
    parser.add_argument("--prefilling_chunk_size", type=int, default=4096)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.device = parse_device(args.device)
    return args