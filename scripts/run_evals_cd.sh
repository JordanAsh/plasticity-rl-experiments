cd ..
# python eval_countdown_pass_at_k.py \
#   --model_path sft_outputs/cd_1.5b_ordered_lr1e5_plain/final \
#   --ks 1,8,16,32 \
#   --n 32 \
#   --output_dir eval_results_pass32/cd_1.5b_sft_ordered_lr1e5_plain


python eval_countdown_pass_at_k.py \
  --model_path /home/t-jinshen/amlt/qwen3b_cd_noformat/qwen3b_cd_noformat/qwen3b_cd_noformat/checkpoints_hf_format/global_step_200 \
  --ks 1,8,16,32,64 \
  --n 64 \
  --output_dir eval_results_pass32/cd_3b_rl_steps200