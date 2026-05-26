#!/bin/bash
set -e
cd /home/joash/plasticity

MODELS=(
    "checkpoints/rl_seed42_step435_merged|eval_results_pass16/rl_seed42_step435"
    "sft_outputs/seed42_shuffled|eval_results_pass16/sft_seed42_shuffled"
    "sft_outputs/seed42_ordered|eval_results_pass16/sft_seed42_ordered"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path output_dir <<< "$entry"
    echo ""
    echo "=========================================="
    echo "pass@16: $model_path"
    echo "=========================================="
    conda run -n plasticity-rl --no-capture-output python eval_pass_at_k.py \
        --model_path "$model_path" \
        --output_dir "$output_dir" \
        --k 16 --n 16 \
        --temperature 0.8 --top_p 0.95 \
        --tensor_parallel_size 4
done

echo ""
echo "=========================================="
echo "ALL pass@16 EVALUATIONS COMPLETE"
echo "=========================================="
for d in eval_results_pass16/*/; do
    echo "--- $d ---"
    cat "$d/summary.json"
    echo ""
done
