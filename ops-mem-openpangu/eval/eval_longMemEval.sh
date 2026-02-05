#!/bin/bash

ori_dir='longmemeval_s'  
result_dir="outputs_longmemeval_s"  #save_dir
out_file='results/longmemeval_s_result.json'  #输出文件
model='Qwen3-32B'
api_base='http://192.168.0.129:1050/v1'
api_key='EMPTY'

python eval/extra_longMem.py  --data_dir "$ori_dir" --result_dir "$result_dir" --save_path "$out_file"
python eval/eval_f1.py --input_file "$out_file"
python eval/eval_llm_judge.py --input_file "$out_file" --model "$model" --api_base "$api_base" --api_key "$api_key"