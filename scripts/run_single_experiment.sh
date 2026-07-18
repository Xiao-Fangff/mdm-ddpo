#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "Usage: $0 RUN_NAME ADVANTAGE_MODE FLOOR LR CLIP SEED ANCHOR_RATIO EPOCHS" >&2
  exit 2
fi

RUN_NAME="$1"
ADVANTAGE_MODE="$2"
FLOOR_QUANTILE="$3"
LEARNING_RATE="$4"
CLIP_RANGE="$5"
SEED="$6"
ANCHOR_RATIO="$7"
EPOCHS="$8"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/stability_experiments}"
REWARD_CALIBRATION_PATH="${REWARD_CALIBRATION_PATH:-${PROJECT_ROOT}/reward_calibration.json}"
FIXED_EVAL_POOL_PATH="${FIXED_EVAL_POOL_PATH:-${PROJECT_ROOT}/artifacts/humanml_val_fixed_eval_pool.pt}"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
REWARD_CALIBRATION_PATH="$(realpath -m "${REWARD_CALIBRATION_PATH}")"
FIXED_EVAL_POOL_PATH="$(realpath -m "${FIXED_EVAL_POOL_PATH}")"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

if [[ ! -f "${REWARD_CALIBRATION_PATH}" ]]; then
  echo "Missing reward calibration: ${REWARD_CALIBRATION_PATH}" >&2
  echo "Run tools/calibrate_reward_stats.py first." >&2
  exit 1
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse experiment output: ${RUN_DIR}" >&2
  echo "Every run must start from the original MDM with zero LoRA." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${FIXED_EVAL_POOL_PATH}")"
export OUTPUT_DIR="${RUN_DIR}"
export MDM_DDPO_SWANLAB_RUN_NAME="${MDM_DDPO_SWANLAB_RUN_NAME:-${RUN_NAME}}"

bash "${PROJECT_ROOT}/scripts/train_humanml.sh" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --sample-steps 50 \
  --rollout-batch-size 32 \
  --rollout-batches-per-epoch 4 \
  --samples-per-prompt 4 \
  --train-batch-size 32 \
  --gradient-accumulation-steps 2 \
  --inner-epochs 1 \
  --timestep-fraction 0.5 \
  --learning-rate "${LEARNING_RATE}" \
  --clip-range "${CLIP_RANGE}" \
  --advantage-mode "${ADVANTAGE_MODE}" \
  --advantage-std-floor-quantile "${FLOOR_QUANTILE}" \
  --advantage-retrieval-weight 0.5 \
  --advantage-m2m-weight 0.5 \
  --reward-calibration-path "${REWARD_CALIBRATION_PATH}" \
  --fixed-eval-pool-path "${FIXED_EVAL_POOL_PATH}" \
  --fixed-eval-every 5 \
  --fixed-eval-prompts 128 \
  --fixed-eval-samples-per-prompt 4 \
  --early-stop-patience 0 \
  --anchor-lambda 0 \
  --anchor-auto-grad-ratio "${ANCHOR_RATIO}" \
  --anchor-batch-size 32 \
  --save-every 5 \
  --log-every 1
