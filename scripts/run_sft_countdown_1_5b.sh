#!/bin/bash

# Run from the repo root so bare paths (run_sft.py, eval_kk.py, sft_outputs/...) resolve.
cd "$(dirname "$0")/.."

LOG_DIR=/home/t-jinshen/plasticity_data/qwen2.5_1.5b_grpo_countdown_20260527_1932
SHUFFLED_OUT=sft_outputs/cd_1.5b_shuffled_lr1e5_plain
ORDERED_OUT=sft_outputs/cd_1.5b_ordered_lr1e5_plain

# Reduce CUDA fragmentation; helps prevent OOM at optimizer.step.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========== SFT shuffled lr=1e-5 cosine | KK 1.5B =========="
torchrun --nproc_per_node=4 --master_port=29501 run_sft.py \
    --generation_logs_dir "$LOG_DIR" \
    --model_path Qwen/Qwen2.5-1.5B \
    --output_dir "$SHUFFLED_OUT" \
    --lr 1e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule constant \
    --score_threshold 0.5 \
    --max_seq_length 1024 \
    --num_checkpoints 10 \
    --snapshot_dtype bfloat16 \
    --no-save_optimizer_state \
    --grad_clip inf \
    --metrics_log_every 10

echo "========== SFT ordered lr=1e-5 cosine | KK 1.5B =========="
torchrun --nproc_per_node=4 --master_port=29502 run_sft.py \
    --generation_logs_dir "$LOG_DIR" \
    --model_path Qwen/Qwen2.5-1.5B \
    --output_dir "$ORDERED_OUT" \
    --lr 1e-5 --batch_size 4 --effective_batch_size 128 \
    --schedule constant \
    --score_threshold 0.5 \
    --max_seq_length 1024 \
    --num_checkpoints 10 \
    --snapshot_dtype bfloat16 \
    --no-save_optimizer_state \
    --grad_clip inf \
    --metrics_log_every 10 \
    --ordered

echo "========== SFT training done. Launching evals at $(date) =========="

# MODELS=(
#     "${SHUFFLED_OUT}|cd_1.5b_sft_shuffled_lr1e5"
#     "${ORDERED_OUT}|cd_1.5b_sft_ordered_lr1e5"
# )

# # Greedy
# for entry in "${MODELS[@]}"; do
#     IFS='|' read -r model_path name <<< "$entry"
#     echo "========== Greedy: $name =========="
#     python eval_kk.py --model_path "$model_path/final" --output_dir eval_results/$name --tensor_parallel_size 4
# done

# # pass@16
# for entry in "${MODELS[@]}"; do
#     IFS='|' read -r model_path name <<< "$entry"
#     echo "========== pass@16: $name =========="
#     python eval_kk_pass_at_k.py --model_path "$model_path/final" --output_dir eval_results_pass16/$name \
#         --k 16 --n 16 --temperature 0.8 --top_p 0.95 --tensor_parallel_size 4
# done

# echo "========== ALL 1.5B KK SFT + EVALS DONE at $(date) =========="
# for d in eval_results/kk_1.5b_sft*/ eval_results_pass16/kk_1.5b_sft*/; do
#     echo "--- $d ---"; cat "$d/summary.json"; echo ""
# done
