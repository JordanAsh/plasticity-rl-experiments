#!/bin/bash
set -e
cd /home/joash/plasticity
source /home/joash/anaconda3/etc/profile.d/conda.sh
conda activate plasticity-rl

LOG_DIR=generation_logs/qwen2.5_1.5b_grpo_countdown_20260527_1932

echo "========== SFT shuffled lr=1e-5 cosine | countdown =========="
torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir "$LOG_DIR" \
    --output_dir sft_outputs/countdown_shuffled_lr1e5 \
    --lr 1e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule cosine --warmup_ratio 0.03 \
    --score_threshold 0.5 \
    --max_seq_length 1024

echo "========== SFT ordered lr=1e-5 cosine | countdown =========="
torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir "$LOG_DIR" \
    --output_dir sft_outputs/countdown_ordered_lr1e5 \
    --lr 1e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule cosine --warmup_ratio 0.03 \
    --score_threshold 0.5 \
    --max_seq_length 1024 \
    --ordered

echo "========== Both countdown SFT runs done at $(date) =========="
