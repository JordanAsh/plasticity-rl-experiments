cd ~/plasticity-rl-experiments
mkdir -p logs

# CKPT_DIR=/home/t-jinshen/amlt/qwen3b_cd_noformat/qwen3b_cd_noformat/qwen3b_cd_noformat/checkpoints_hf_format
# TAG=qwen3b_cd_noformat

# for r in 0 1 2 3; do
#   CUDA_VISIBLE_DEVICES=$r python eval_rl_checkpoints.py \
#     --checkpoint_dir "$CKPT_DIR" \
#     --init_model_path Qwen/Qwen2.5-3B \
#     --step_interval 50 \
#     --history_step_stride 10 \
#     --rank $r --world_size 4 \
#     --overwrite \
#     > logs/eval_${TAG}_r${r}.log 2>&1 &
# done
# wait
# echo "done: $TAG"

RUN=/home/t-jinshen/amlt/llama3b_cd_sft_sweep_v5/cd3b_sft_ordered_lr1e-05_seed42_lr_1e-05_ord_ordered_see_42   # change to kk_1.5b_shuffled_lr1e5 for the other run
TAG=$(basename "$RUN")

for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r python eval_checkpoints.py \
    --run_dir "$RUN" \
    --bf16 --batch_size 4 \
    --rank $r --world_size 4 \
    --skip_probe \
    --old_data_step_stride 25 \
    --overwrite \
    --include_final \
    > logs/eval_${TAG}_r${r}.log 2>&1 &
done
wait
echo "done: $RUN"

RUN=/home/t-jinshen/amlt/llama3b_cd_sft_sweep_v5/cd3b_sft_shuffled_lr1e-05_seed42_lr_1e-05_ord_shuffled_see_42   # change to kk_1.5b_shuffled_lr1e5 for the other run
TAG=$(basename "$RUN")

for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r python eval_checkpoints.py \
    --run_dir "$RUN" \
    --bf16 --batch_size 4 \
    --rank $r --world_size 4 \
    --skip_probe \
    --old_data_step_stride 25 \
    --overwrite \
    --include_final \
    > logs/eval_${TAG}_r${r}.log 2>&1 &
done
wait
echo "done: $RUN"