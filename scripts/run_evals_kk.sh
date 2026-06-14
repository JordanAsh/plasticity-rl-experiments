#!/bin/bash

# Run from the repo root so bare paths resolve.
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Existing 1.5B runs (already trained). Use absolute paths so HF/vLLM does not
# mistake them for hub repo ids.
MODELS=(
    "${REPO_ROOT}/sft_outputs/kk_3b_shuffled_lr1e5_plain|kk_3b_sft_shuffled_lr1e5_plain"
    "${REPO_ROOT}/sft_outputs/kk_3b_ordered_lr1e5_plain|kk_3b_sft_ordered_lr1e5_plain"
)

echo "========== Eval start at $(date) =========="

# Greedy
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_root name <<< "$entry"
    echo "========== Greedy: $name =========="
    python eval_kk.py \
        --model_path "$model_root/final/" \
        --output_dir eval_results/$name \
        --tensor_parallel_size 4
done

# pass@{1,8,16,32} from a single n=32 sampling run per model
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_root name <<< "$entry"
    echo "========== pass@1,8,16,32 (n=32): $name =========="
    python eval_kk_pass_at_k.py \
        --model_path "$model_root/final" \
        --output_dir eval_results_pass32/$name \
        --ks 1,8,16,32,64 --n 64 \
        --temperature 0.8 --top_p 0.95 \
        --tensor_parallel_size 4
done

echo "========== ALL 3B KK EVALS DONE at $(date) =========="
for d in eval_results/kk_3b_sft*/ eval_results_pass32/kk_3b_sft*/; do
    echo "--- $d ---"; cat "$d/summary.json" 2>/dev/null; echo ""
done
