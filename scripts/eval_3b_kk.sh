cd ~/plasticity-rl-experiments
mkdir -p logs

RUN=sft_outputs/kk_3b_ordered_lr1e5_plain   # change to kk_1.5b_shuffled_lr1e5 for the other run
TAG=$(basename "$RUN")

for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r python eval_checkpoints.py \
    --run_dir "$RUN" \
    --bf16 --batch_size 4 \
    --rank $r --world_size 4 \
    --skip_probe \
    --old_data_step_stride 25 \
    --include_final \
    > logs/eval_${TAG}_r${r}.log 2>&1 &
done
wait
echo "done: $RUN"

RUN=sft_outputs/kk_3b_shuffled_lr1e5_plain   # change to kk_1.5b_shuffled_lr1e5 for the other run
TAG=$(basename "$RUN")

for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r python eval_checkpoints.py \
    --run_dir "$RUN" \
    --bf16 --batch_size 4 \
    --rank $r --world_size 4 \
    --skip_probe \
    --old_data_step_stride 25 \
    --include_final \
    > logs/eval_${TAG}_r${r}.log 2>&1 &
done
wait
echo "done: $RUN"