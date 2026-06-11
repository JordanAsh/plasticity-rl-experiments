cd ..
# python eval_countdown_pass_at_k.py \
#   --model_path sft_outputs/cd_1.5b_ordered_lr1e5_plain/final \
#   --ks 1,8,16,32 \
#   --n 32 \
#   --output_dir eval_results_pass32/cd_1.5b_sft_ordered_lr1e5_plain


python eval_countdown_pass_at_k.py \
  --model_path sft_outputs/cd_1.5b_shuffled_lr1e5_plain/final \
  --ks 1,8,16,32 \
  --n 32 \
  --output_dir eval_results_pass32/cd_1.5b_sft_shuffled_lr1e5_plain