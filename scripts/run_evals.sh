#!/bin/bash
set -e
cd /home/joash/plasticity

echo "=========================================="
echo "Evaluation 1/3: RL checkpoint (seed42, step 435)"
echo "=========================================="
conda run -n plasticity-rl --no-capture-output python eval_model.py \
    --model_path checkpoints/rl_seed42_step435_merged \
    --output_dir eval_results/rl_seed42_step435 \
    --tensor_parallel_size 4

echo ""
echo "=========================================="
echo "Evaluation 2/3: SFT shuffled"
echo "=========================================="
conda run -n plasticity-rl --no-capture-output python eval_model.py \
    --model_path sft_outputs/seed42_shuffled \
    --output_dir eval_results/sft_seed42_shuffled \
    --tensor_parallel_size 4

echo ""
echo "=========================================="
echo "Evaluation 3/3: SFT ordered"
echo "=========================================="
conda run -n plasticity-rl --no-capture-output python eval_model.py \
    --model_path sft_outputs/seed42_ordered \
    --output_dir eval_results/sft_seed42_ordered \
    --tensor_parallel_size 4

echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE"
echo "=========================================="
for d in eval_results/*/; do
    echo "--- $d ---"
    cat "$d/summary.json"
    echo ""
done
