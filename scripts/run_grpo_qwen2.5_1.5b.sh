#!/usr/bin/env bash
# GRPO | Qwen2.5-1.5B | FSDP training | MATH + GSM8K
#
# Plasticity-loss research baseline.
# Adapted from verl/examples/grpo_trainer/run_qwen3_8b_fsdp.sh,
# scaled down for Qwen2.5-1.5B on a single node (e.g. 4×A100 or 8×A100).
#
# Usage:
#   conda activate plasticity-rl
#   cd /home/joash/plasticity
#   bash run_grpo_qwen2.5_1.5b.sh
#
# Override knobs on the command line:
#   NGPUS=4 ROLLOUT_TP=1 bash run_grpo_qwen2.5_1.5b.sh

set -xeuo pipefail

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B}
NGPUS=${NGPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
INFER_BACKEND=${INFER_BACKEND:-vllm}

# Training hyperparams (GRPO defaults from verl, tuned for 1.5B)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
ROLLOUT_N=${ROLLOUT_N:-5}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-150}
TEST_FREQ=${TEST_FREQ:-25}

PROJECT_NAME=${PROJECT_NAME:-plasticity_grpo_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_grpo_$(date +%Y%m%d_%H%M)}
GENERATION_LOG_DIR=${GENERATION_LOG_DIR:-/home/joash/plasticity/generation_logs/${EXPERIMENT_NAME}}
########################### end user-adjustable ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$HOME/data/gsm8k/train.parquet','$HOME/data/math/train.parquet']"
    data.val_files="['$HOME/data/gsm8k/test.parquet','$HOME/data/math/test.parquet']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.rollout_data_dir=${GENERATION_LOG_DIR}
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
