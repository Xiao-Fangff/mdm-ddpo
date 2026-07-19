#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MDM_DDPO_ENABLE_STEP_REWARD=1
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/step_k8_soft_counterfactual}"
export MDM_DDPO_STEP_REWARD_CALIBRATION_PATH="${MDM_DDPO_STEP_REWARD_CALIBRATION_PATH:-${PROJECT_ROOT}/step_reward_k8_soft_huber_calibration.json}"
export MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH="${MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH:-${PROJECT_ROOT}/artifacts/step_val_fixed_eval_pool_k8_soft.pt}"
export MDM_DDPO_SWANLAB_RUN_NAME="${MDM_DDPO_SWANLAB_RUN_NAME:-step-k8-soft-counterfactual}"

if [[ ! -f "${MDM_DDPO_STEP_REWARD_CALIBRATION_PATH}" ]]; then
  echo "Missing K8 soft-Huber calibration: ${MDM_DDPO_STEP_REWARD_CALIBRATION_PATH}" >&2
  echo "Run tools/calibrate_step_reward_stats.py with synthetic K8 soft_huber_exact first." >&2
  exit 1
fi

exec bash "${PROJECT_ROOT}/scripts/train_humanml_step.sh" \
  --epochs 30 \
  --sample-steps 50 \
  --rollout-batch-size 64 \
  --rollout-batches-per-epoch 4 \
  --samples-per-prompt 4 \
  --step-samples-per-prompt 8 \
  --step-data-ratio 0.5 \
  --step-rollout-source synthetic \
  --step-balanced-sampling \
  --train-batch-size 64 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --clip-range 1e-4 \
  --advantage-retrieval-weight 0.5 \
  --advantage-m2m-weight 0.5 \
  --step-advantage-retrieval-weight 0.0 \
  --step-advantage-m2m-weight 0.0 \
  --step-advantage-step-weight 1.0 \
  --no-step-use-m2m-reward \
  --step-reward-weight 1.0 \
  --step-reward-mode soft_huber_exact \
  --fixed-step-eval-samples-per-prompt 8 \
  --fixed-eval-every 2 \
  --early-stop-patience 0 \
  --save-every 2 \
  "$@"
