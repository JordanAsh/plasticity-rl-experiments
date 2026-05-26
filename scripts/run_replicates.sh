#!/usr/bin/env bash
# Run N replicates of GRPO training sequentially with different seeds.
# Each replicate gets its own generation log directory.
#
# Usage:
#   bash run_replicates.sh          # 3 replicates (default)
#   N_REPLICATES=5 bash run_replicates.sh

set -euo pipefail

N_REPLICATES=${N_REPLICATES:-3}
SEEDS=(42 137 2024)  # extend if N_REPLICATES > 3

for i in $(seq 0 $((N_REPLICATES - 1))); do
    SEED=${SEEDS[$i]:-$((42 + i * 100))}
    EXPERIMENT="qwen2.5_1.5b_grpo_seed${SEED}_$(date +%Y%m%d_%H%M)"

    echo "========================================"
    echo "Replicate $((i + 1))/${N_REPLICATES}  seed=${SEED}  experiment=${EXPERIMENT}"
    echo "========================================"

    ray stop --force 2>/dev/null || true

    VLLM_USE_V1=0 EXPERIMENT_NAME="${EXPERIMENT}" \
        bash run_grpo_qwen2.5_1.5b.sh \
        trainer.logger='["console"]' \
        +actor_rollout_ref.rollout.seed=${SEED} \
        trainer.val_before_train=False

    echo "Replicate $((i + 1)) done. Logs: generation_logs/${EXPERIMENT}/"
    echo ""
done

echo "All ${N_REPLICATES} replicates complete."
