#!/bin/bash
set -e
cd /home/joash/plasticity

# Wait for lr=5e-5 evals to finish
echo "Waiting for lr=5e-5 evals to finish..."
while pgrep -f "eval_pass_at_k.*lr5e5\|run_evals_5e5" > /dev/null 2>&1; do
    sleep 60
done
echo "Previous runs done at $(date)"

# SFT training
echo "========== SFT shuffled lr=1e-4 =========="
conda run -n plasticity-rl --no-capture-output torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir generation_logs/qwen2.5_1.5b_grpo_seed42_20260519_2017 \
    --output_dir sft_outputs/seed42_shuffled_lr1e4 \
    --lr 1e-4 --batch_size 4 --effective_batch_size 128 \
    --schedule cosine --warmup_ratio 0.03

echo ""
echo "========== SFT ordered lr=1e-4 =========="
conda run -n plasticity-rl --no-capture-output torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir generation_logs/qwen2.5_1.5b_grpo_seed42_20260519_2017 \
    --output_dir sft_outputs/seed42_ordered_lr1e4 \
    --lr 1e-4 --batch_size 4 --effective_batch_size 128 \
    --schedule cosine --warmup_ratio 0.03 \
    --ordered

echo "SFT lr=1e-4 training done at $(date)"

# Greedy evals
for variant in shuffled ordered; do
    echo ""
    echo "========== Greedy eval: seed42_${variant}_lr1e4 =========="
    conda run -n plasticity-rl --no-capture-output python eval_model.py \
        --model_path sft_outputs/seed42_${variant}_lr1e4 \
        --output_dir eval_results/sft_seed42_${variant}_lr1e4 \
        --tensor_parallel_size 4
done

# pass@16 evals
for variant in shuffled ordered; do
    echo ""
    echo "========== pass@16 eval: seed42_${variant}_lr1e4 =========="
    conda run -n plasticity-rl --no-capture-output python eval_pass_at_k.py \
        --model_path sft_outputs/seed42_${variant}_lr1e4 \
        --output_dir eval_results_pass16/sft_seed42_${variant}_lr1e4 \
        --k 16 --n 16 \
        --temperature 0.8 --top_p 0.95 \
        --tensor_parallel_size 4
done

echo ""
echo "=========================================="
echo "ALL lr=1e-4 RUNS COMPLETE"
echo "=========================================="
echo "--- Greedy ---"
for d in eval_results/sft_seed42_*_lr1e4/; do
    echo "$d:"; cat "$d/summary.json"; echo ""
done
echo "--- pass@16 ---"
for d in eval_results_pass16/sft_seed42_*_lr1e4/; do
    echo "$d:"; cat "$d/summary.json"; echo ""
done
