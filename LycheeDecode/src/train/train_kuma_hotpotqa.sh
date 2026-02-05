export CUDA_VISIBLE_DEVICES=0

model_name=Qwen3-8B
ctx_len_min=1000
ctx_len_max=20000
prefill_chunk_size=1000
lr=0.01
num_steps=3000

lamda_init_value=2.0
lagrange_lr=0.001
a_init_value=1.0
b_init_value=1.0
desired_density=0.08333

setting="lr=${lr}-ctx=${ctx_len_min}_${ctx_len_max}_lamda_init=${lamda_init_value}_lagrange_lr=${lagrange_lr}_a_init=${a_init_value}_b_init=${b_init_value}_desired_density=${desired_density}_kuma_hotpotqa"

exp_name=${model_name}/${setting}

python  train_kuma.py \
    --model_name /your/model/path/${model_name} \
    --batch_size 1 \
    --num_steps ${num_steps} \
    --lr ${lr} \
    --prefilling_chunk_size ${prefill_chunk_size} \
    --exp_name $exp_name \
    --context_length_min ${ctx_len_min} \
    --context_length_max ${ctx_len_max} \
    --context_lengths_num_intervals 50 \
    --dataset_format "hotpotqa" \
    --output_dir outputs/${exp_name} \
    --lamda_init_value ${lamda_init_value} \
    --lagrange_lr ${lagrange_lr} \
    --a_init_value ${a_init_value} \
    --b_init_value ${b_init_value} \
    --desired_density ${desired_density}
#    --disable_wandb