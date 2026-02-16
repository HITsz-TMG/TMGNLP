EVAL=""
OUTPUT_DIR=""
MODEL=""
python run_alpaca_eval.py \
    --ckpt $MODEL \
    --model_name $MODEL \
    --output_data_path $OUTPUT_DIR \
    --tensor_parallel_size 1 \
    --sanity_check False \
    --split eval \
    --eval_data_path $EVAL
