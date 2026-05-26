#!/bin/bash
set -e
cd /home/joash/plasticity

echo "========== SFT shuffled lr=5e-5 constant =========="
conda run -n plasticity-rl --no-capture-output torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir generation_logs/qwen2.5_1.5b_grpo_seed42_20260519_2017 \
    --output_dir sft_outputs/seed42_shuffled_lr5e5_const \
    --lr 5e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule constant --warmup_ratio 0.0

echo "========== SFT ordered lr=5e-5 constant =========="
conda run -n plasticity-rl --no-capture-output torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir generation_logs/qwen2.5_1.5b_grpo_seed42_20260519_2017 \
    --output_dir sft_outputs/seed42_ordered_lr5e5_const \
    --lr 5e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule constant --warmup_ratio 0.0 \
    --ordered

# Greedy evals
for variant in shuffled ordered; do
    echo "========== Greedy: seed42_${variant}_lr5e5_const =========="
    conda run -n plasticity-rl --no-capture-output python eval_model.py \
        --model_path sft_outputs/seed42_${variant}_lr5e5_const \
        --output_dir eval_results/sft_seed42_${variant}_lr5e5_const \
        --tensor_parallel_size 4
done

# pass@16 evals
for variant in shuffled ordered; do
    echo "========== pass@16: seed42_${variant}_lr5e5_const =========="
    conda run -n plasticity-rl --no-capture-output python eval_pass_at_k.py \
        --model_path sft_outputs/seed42_${variant}_lr5e5_const \
        --output_dir eval_results_pass16/sft_seed42_${variant}_lr5e5_const \
        --k 16 --n 16 --temperature 0.8 --top_p 0.95 \
        --tensor_parallel_size 4
done

echo "=========================================="
echo "ALL CONSTANT LR RUNS COMPLETE at $(date)"
echo "=========================================="
