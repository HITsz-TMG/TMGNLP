GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=42512
ds_config_path="config/deepspeed_config/ds_config.json"

MODEL=""
DATA="path_to_ultrachat_200k"
torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    -m sft \
    --deepspeed ${ds_config_path} \
    --model_name_or_path $MODEL \
    --bf16 True \
    --output_dir "" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --logging_steps 20 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --save_only_model True \
    --log_level info \
    --model_max_length 2048 \
    --dataset_name $DATA \
    --dataset_train_split train_sft \
    --report_to "none" \
    --lr_scheduler_type cosine \
    --no_remove_unused_columns
