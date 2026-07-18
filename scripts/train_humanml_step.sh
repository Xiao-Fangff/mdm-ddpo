#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MDM_DDPO_ENABLE_STEP_REWARD=1
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/humanml_step_ddpo}"
export MDM_DDPO_STEP_REWARD_CALIBRATION_PATH="${MDM_DDPO_STEP_REWARD_CALIBRATION_PATH:-${PROJECT_ROOT}/step_reward_calibration.json}"
export MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH="${MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH:-${PROJECT_ROOT}/artifacts/step_val_fixed_eval_pool.pt}"

if [[ ! -f "${MDM_DDPO_STEP_REWARD_CALIBRATION_PATH}" ]]; then
  echo "Missing step reward calibration: ${MDM_DDPO_STEP_REWARD_CALIBRATION_PATH}" >&2
  echo "Run tools/calibrate_step_reward_stats.py first." >&2
  exit 1
fi

exec bash "${PROJECT_ROOT}/scripts/train_humanml.sh" \
  --advantage-mode component_shrink \
  --advantage-retrieval-weight 0.375 \
  --advantage-m2m-weight 0.375 \
  --advantage-step-weight 0.25 \
  --step-data-ratio 0.25 \
  --step-reward-weight 0.5 \
  "$@"
