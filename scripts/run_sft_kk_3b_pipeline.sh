#!/bin/bash
set -e
cd /home/joash/plasticity
source /home/joash/anaconda3/etc/profile.d/conda.sh
conda activate plasticity-rl

LOG_DIR=generation_logs/qwen2.5_3b_grpo_kk_20260601_1521

echo "========== SFT shuffled lr=1e-5 cosine | KK 3B =========="
torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir "$LOG_DIR" \
    --model_path Qwen/Qwen2.5-3B \
    --output_dir sft_outputs/kk_3b_shuffled_lr1e5 \
    --lr 1e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule cosine --warmup_ratio 0.03 \
    --score_threshold 2.5 \
    --max_seq_length 2048

echo "========== SFT ordered lr=1e-5 cosine | KK 3B =========="
torchrun --nproc_per_node=4 run_sft.py \
    --generation_logs_dir "$LOG_DIR" \
    --model_path Qwen/Qwen2.5-3B \
    --output_dir sft_outputs/kk_3b_ordered_lr1e5 \
    --lr 1e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule cosine --warmup_ratio 0.03 \
    --score_threshold 2.5 \
    --max_seq_length 2048 \
    --ordered

echo "========== SFT training done. Launching evals at $(date) =========="

MODELS=(
    "sft_outputs/kk_3b_shuffled_lr1e5|kk_3b_sft_shuffled_lr1e5"
    "sft_outputs/kk_3b_ordered_lr1e5|kk_3b_sft_ordered_lr1e5"
)

# Greedy
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path name <<< "$entry"
    echo "========== Greedy: $name =========="
    python eval_kk.py --model_path "$model_path" --output_dir eval_results/$name --tensor_parallel_size 4
done

# pass@16
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path name <<< "$entry"
    echo "========== pass@16: $name =========="
    python eval_kk_pass_at_k.py --model_path "$model_path" --output_dir eval_results_pass16/$name \
        --k 16 --n 16 --temperature 0.8 --top_p 0.95 --tensor_parallel_size 4
done

echo "========== ALL 3B KK SFT + EVALS DONE at $(date) =========="
for d in eval_results/kk_3b_sft*/ eval_results_pass16/kk_3b_sft*/; do
    echo "--- $d ---"; cat "$d/summary.json"; echo ""
done
