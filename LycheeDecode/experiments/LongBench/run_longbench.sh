#!/bin/bash

cd "$(dirname "$0")" || exit

export CUDA_VISIBLE_DEVICES=3

mkdir -p longbench_logs
model="/your/model/path/Qwen3-8B"
full_head_path="../../src/train/outputs/Qwen3-8B/lr=0.01-ctx=1000_10000_lamda_init=2.0_lagrange_lr=0.001_a_init=1.0_b_init=1.0_desired_density=0.08333_sparse_radio_train=0.7_kuma_multi_passkey/full_head.pt"

sparse_mode=topk
mode_value=4096

setting="${sparse_mode}=${mode_value}"
for task in "qmsum" "narrativeqa" "qasper" "multifieldqa_en" "triviaqa" "passage_retrieval_en" "hotpotqa" "2wikimqa"
do
    python -u pred.py \
        --model $model --task $task \
        --sparse_mode ${sparse_mode}\
        --mode_value ${mode_value} \
        --setting ${setting} \
        --full_head_path ${full_head_path} \
        2>&1 | tee "longbench_logs/${task}_${setting}_output.log"
done

python -u eval.py --model $model --setting ${setting}