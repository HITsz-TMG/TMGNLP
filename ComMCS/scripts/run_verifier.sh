GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=34672
ds_config_path="config/ds_config_zero3.json"

# V_mse
torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    -m categorical_training.train_rm \
    --deepspeed "config/ds_config_zero3.json" \
    --model_name_or_path "" \
    --dataset_name "" \
    --output_dir "" \
    --dataset_train_split train \
    --dataset_test_split test \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_eval False \
    --eval_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 2.0e-6 \
    --gradient_accumulation_steps 8 \
    --logging_steps 50 \
    --max_seq_length 1024 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --loss_type "mse" \
    --attn_implementation flash_attention_2 \
    --save_only_model True

# V_ce
torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    -m categorical_training.train_classifier \
    --deepspeed ${ds_config_path} \
    --model_name_or_path "" \
    --dataset_name "" \
    --output_dir "" \
    --dataset_train_split train \
    --dataset_test_split test \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_eval True \
    --eval_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 2.0e-6 \
    --gradient_accumulation_steps 8 \
    --logging_steps 50 \
    --max_seq_length 1024 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --loss_type "weighted_cross_entropy" \
    --distribution_type "td" \
    --scale 1 \
    --attn_implementation flash_attention_2 \
    --save_only_model True \
    --remove_unused_columns False


# V_ce + variance reduction
torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    -m categorical_training.train_classifier \
    --deepspeed ${ds_config_path} \
    --model_name_or_path "" \
    --dataset_name "" \
    --output_dir "" \
    --dataset_train_split train \
    --dataset_test_split test \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_eval True \
    --eval_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 2.0e-6 \
    --gradient_accumulation_steps 8 \
    --logging_steps 50 \
    --max_seq_length 1024 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --loss_type "dv1" \
    --distribution_type "td" \
    --scale 1 \
    --attn_implementation flash_attention_2 \
    --save_only_model True \
    --remove_unused_columns False


# V_bce
torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    -m categorical_training.train_classifier \
    --deepspeed ${ds_config_path} \
    --model_name_or_path "" \
    --dataset_name "" \
    --output_dir "" \
    --dataset_train_split train \
    --dataset_test_split test \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_eval True \
    --eval_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 2.0e-6 \
    --gradient_accumulation_steps 8 \
    --logging_steps 50 \
    --max_seq_length 1024 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --loss_type "weighted_cross_entropy" \
    --distribution_type "binary" \
    --scale 1 \
    --num_labels 2 \
    --attn_implementation flash_attention_2 \
    --save_only_model True \
    --remove_unused_columns False