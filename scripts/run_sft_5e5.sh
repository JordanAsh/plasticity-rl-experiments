#!/bin/bash
set -e
cd /home/joash/plasticity

echo "========== SFT shuffled lr=5e-5 =========="
conda run -n plasticity-rl --no-capture-output torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir generation_logs/qwen2.5_1.5b_grpo_seed42_20260519_2017 \
    --output_dir sft_outputs/seed42_shuffled_lr5e5 \
    --lr 5e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule cosine --warmup_ratio 0.03

echo ""
echo "========== SFT ordered lr=5e-5 =========="
conda run -n plasticity-rl --no-capture-output torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir generation_logs/qwen2.5_1.5b_grpo_seed42_20260519_2017 \
    --output_dir sft_outputs/seed42_ordered_lr5e5 \
    --lr 5e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule cosine --warmup_ratio 0.03 \
    --ordered

echo ""
echo "========== Both SFT lr=5e-5 runs done =========="
