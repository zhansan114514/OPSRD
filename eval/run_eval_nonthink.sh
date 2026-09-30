#!/bin/bash

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"

# Evaluate the both-nonthink 4B model (student & teacher both non-thinking during training)
# at checkpoint-100 on AIME24, in non-thinking inference mode.
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 4 \
    --no_thinking \
    --checkpoint_dir "${CHECKPOINT_DIR:-./runs/qwen34b_both_nonthink/checkpoint-100}"
wait
