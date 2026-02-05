#!/bin/bash

cd "$(dirname "$0")" || exit

export CUDA_VISIBLE_DEVICES=1
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

FULL_HEAD_NUM_LIST="0 1 2 3 4 5 6 7 8"   
BATCH_LIST="1 2 4 8"      
CONTETX_LIST="16000 32000 64000 128000"

start_time=$(date +%s)
echo "🚀 Starting experiments..."
echo "Logs will be saved in the '$LOG_DIR' directory."
echo "=================================================="

for fhn in $FULL_HEAD_NUM_LIST; do
  for batch_size in $BATCH_LIST; do
      for context_len in $CONTETX_LIST; do
        LOG_FILE="${LOG_DIR}/run_fhn_${fhn}_batch_${batch_size}_context_${context_len}.log"

        echo "Running: full_head_num=${fhn}, batch=${batch_size}, context_len=${context_len} ... Log: ${LOG_FILE}"

        python ../../src/kernels/tilelang_sparse_gqa_decode_varlen_indice_hybrid_head.py \
          --batch "$batch_size" \
          --max_cache_seqlen "$context_len" \
          --full_head_num "$fhn" > "$LOG_FILE" 2>&1
    done
  done
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "=================================================="
echo "✅ All experiments completed!"
echo "Total duration: ${duration} seconds."
echo "Please check the log files in '$LOG_DIR'."