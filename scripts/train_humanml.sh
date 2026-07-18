#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/zhiwei/anaconda3/envs/motionrft/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
REWARD_DEVICE="${REWARD_DEVICE:-same}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/humanml_retrieval_m2m}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${PROJECT_ROOT}/.cache/mdm}"
REWARD_CALIBRATION_PATH="${MDM_DDPO_REWARD_CALIBRATION_PATH:-${REWARD_CALIBRATION_PATH:-${PROJECT_ROOT}/reward_calibration.json}}"
FIXED_EVAL_POOL_PATH="${MDM_DDPO_FIXED_EVAL_POOL_PATH:-${FIXED_EVAL_POOL_PATH:-${PROJECT_ROOT}/artifacts/humanml_val_fixed_eval_pool.pt}}"
DDPO_ENABLE_STEP_REWARD="${MDM_DDPO_ENABLE_STEP_REWARD:-${ENABLE_STEP_REWARD:-0}}"
STEP_REWARD_CALIBRATION_PATH="${MDM_DDPO_STEP_REWARD_CALIBRATION_PATH:-${STEP_REWARD_CALIBRATION_PATH:-${PROJECT_ROOT}/step_reward_calibration.json}}"
FIXED_STEP_EVAL_POOL_PATH="${MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH:-${FIXED_STEP_EVAL_POOL_PATH:-${PROJECT_ROOT}/artifacts/step_val_fixed_eval_pool.pt}}"
STEP_DATA_MANIFEST="${MDM_DDPO_STEP_DATA_MANIFEST:-${STEP_DATA_MANIFEST:-/home/zhiwei/projects/MotionRFT/RFT_MLD/third_party/motion-rule-data/offline_reward_validation/walk_step_five_manifests_0_6_random400/sample_manifest.jsonl}}"
STEP_MOTION_ROOT="${MDM_DDPO_STEP_MOTION_ROOT:-${STEP_MOTION_ROOT:-/home/zhiwei/projects/MotionRFT/RFT_MLD/third_party/motion-rule}}"
STEP_DETECTOR_ROOT="${MDM_DDPO_STEP_DETECTOR_ROOT:-${STEP_DETECTOR_ROOT:-${STEP_MOTION_ROOT}}}"
DDPO_USE_SWANLAB="${MDM_DDPO_USE_SWANLAB:-${USE_SWANLAB:-0}}"
DDPO_SWANLAB_PROJECT="${MDM_DDPO_SWANLAB_PROJECT:-${SWANLAB_PROJECT:-mdm-ddpo}}"
DDPO_SWANLAB_RUN_NAME="${MDM_DDPO_SWANLAB_RUN_NAME:-${SWANLAB_RUN_NAME:-}}"
DDPO_SWANLAB_WORKSPACE="${MDM_DDPO_SWANLAB_WORKSPACE:-${SWANLAB_WORKSPACE:-}}"
DDPO_SWANLAB_MODE="${MDM_DDPO_SWANLAB_MODE:-${SWANLAB_MODE:-online}}"
DDPO_SWANLAB_LOG_DIR="${MDM_DDPO_SWANLAB_LOG_DIR:-${SWANLAB_LOG_DIR:-}}"

# SwanLab 0.9 reserves several SWANLAB_* names for nested SDK settings.
# Values have already been copied to DDPO_* variables and are passed as CLI
# arguments below, so do not leak the convenience aliases into the SDK import.
unset USE_SWANLAB SWANLAB_PROJECT SWANLAB_RUN_NAME SWANLAB_WORKSPACE
unset SWANLAB_MODE SWANLAB_LOG_DIR
unset ENABLE_STEP_REWARD STEP_REWARD_CALIBRATION_PATH FIXED_STEP_EVAL_POOL_PATH
unset STEP_DATA_MANIFEST STEP_MOTION_ROOT STEP_DETECTOR_ROOT

SWANLAB_FLAG="--no-use-swanlab"
case "${DDPO_USE_SWANLAB,,}" in
  1|true|yes|on) SWANLAB_FLAG="--use-swanlab" ;;
esac

STEP_ARGS=(--no-enable-step-reward)
case "${DDPO_ENABLE_STEP_REWARD,,}" in
  1|true|yes|on)
    STEP_ARGS=(
      --enable-step-reward
      --step-reward-calibration-path "${STEP_REWARD_CALIBRATION_PATH}"
      --fixed-step-eval-pool-path "${FIXED_STEP_EVAL_POOL_PATH}"
      --step-data-manifest "${STEP_DATA_MANIFEST}"
      --step-motion-root "${STEP_MOTION_ROOT}"
      --step-detector-root "${STEP_DETECTOR_ROOT}"
      --step-data-ratio 0.25
      --step-targets 1,2,3,4,5,6
      --step-eval-samples-per-target 8
      --step-detector-backend progressive
      --step-detector-lead-threshold 0.138
      --step-reward-mode exp
      --step-reward-temperature 1.0
      --step-reward-weight 0.5
      --advantage-step-weight 0.25
    )
  ;;
esac

exec "${PYTHON}" "${PROJECT_ROOT}/train_ddpo.py" \
  --device "${DEVICE}" \
  --reward-device "${REWARD_DEVICE}" \
  --precision bf16 \
  --output-dir "${OUTPUT_DIR}" \
  --data-cache-dir "${DATA_CACHE_DIR}" \
  --reward-calibration-path "${REWARD_CALIBRATION_PATH}" \
  --sample-steps 50 \
  --guidance-scale 2.5 \
  --ddim-eta 1.0 \
  --rollout-batch-size 32 \
  --rollout-batches-per-epoch 4 \
  --samples-per-prompt 4 \
  --train-batch-size 32 \
  --inner-epochs 1 \
  --timestep-fraction 0.5 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-4 \
  --advantage-mode component_shrink \
  --advantage-std-floor-quantile p25 \
  --advantage-retrieval-weight 0.5 \
  --advantage-m2m-weight 0.5 \
  --train-mode lora \
  --lora-rank 8 \
  --lora-alpha 8 \
  --retrieval-weight 1.0 \
  --m2m-weight 1.0 \
  --reward-embedding-mode mean \
  --fixed-eval-every 5 \
  --fixed-eval-prompts 128 \
  --fixed-eval-samples-per-prompt 4 \
  --fixed-eval-pool-path "${FIXED_EVAL_POOL_PATH}" \
  "${STEP_ARGS[@]}" \
  --early-stop-patience 8 \
  --early-stop-min-delta-mode auto \
  --anchor-lambda 0 \
  --anchor-auto-grad-ratio 0 \
  "${SWANLAB_FLAG}" \
  --swanlab-project "${DDPO_SWANLAB_PROJECT}" \
  --swanlab-run-name "${DDPO_SWANLAB_RUN_NAME}" \
  --swanlab-workspace "${DDPO_SWANLAB_WORKSPACE}" \
  --swanlab-mode "${DDPO_SWANLAB_MODE}" \
  --swanlab-log-dir "${DDPO_SWANLAB_LOG_DIR}" \
  "$@"
