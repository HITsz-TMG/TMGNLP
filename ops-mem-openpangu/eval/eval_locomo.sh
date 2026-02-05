#!/bin/bash

ori_dir='locomo/locomo10.json'  #原始locomo10.json 存放位置
prefix="outputs" #save_dir
result_dir="${prefix}/locomo_"
out_file='results/locomo_result.json'  #输出文件
model='Qwen3-32B'
api_base='http://192.168.0.129:1050/v1'
api_key='EMPTY'

python eval/extra_locomo.py --ori_dir "$ori_dir" --result_dir "$result_dir" --output "$out_file"
python eval/eval_f1.py --input_file "$out_file"
python eval/eval_llm_judge.py --input_file "$out_file" --model "$model" --api_base "$api_base" --api_key "$api_key"