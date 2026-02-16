GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=34672
ds_config_path="config/deepspeed_config/ds_config_zero3.json"

MODEL=""
DATA="path_to_ultrafeedback_relabeled"
EVAL="path_to_ultrafeedback_binarized"
torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    -m dpo \
    --deepspeed ${ds_config_path} \
    --model_name_or_path $MODEL \
    --bf16 True \
    --output_dir "" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --learning_rate 5e-7 \
    --logging_steps 20 \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --log_level info \
    --max_length 2048 \
    --train_data_path $DATA \
    --eval_data_path $EVAL \
    --lr_scheduler_type cosine \
    --beta 0.01 \
    --save_only_model True \
    --report_to "none"