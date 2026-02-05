#!/bin/bash

cd "$(dirname "$0")" || exit

export CUDA_VISIBLE_DEVICES=0

model="/data/share/Model/Llama-3-8B-Instruct-Gradient-1048k"
full_head_path="../../src/train/outputs/Llama-3-8B-Instruct-Gradient-1048k/lr=0.01-ctx=1000_10000_lamda_init=2.0_lagrange_lr=0.001_a_init=1.0_b_init=1.0_desired_density=0.09375_sparse_radio_train=0.7_kuma_multi_passkey/full_head.pt"

block_size=64
block_num=64

python benchmark_e2e.py \
    --model $model \
    --full_head_path $full_head_path \
    --batch 1 \
    --max_cache_seqlen 32000 \
    --block_size $block_size \
    --block_num $block_num