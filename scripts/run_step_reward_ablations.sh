#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/step_reward_ablations}"
REWARD_CALIBRATION_PATH="${REWARD_CALIBRATION_PATH:-${PROJECT_ROOT}/reward_calibration.json}"
STEP_REWARD_CALIBRATION_PATH="${STEP_REWARD_CALIBRATION_PATH:-${PROJECT_ROOT}/step_reward_k16_calibration.json}"
FIXED_EVAL_POOL_PATH="${FIXED_EVAL_POOL_PATH:-${PROJECT_ROOT}/artifacts/humanml_val_fixed_eval_pool.pt}"
FIXED_STEP_EVAL_POOL_PATH="${FIXED_STEP_EVAL_POOL_PATH:-${PROJECT_ROOT}/artifacts/step_val_fixed_eval_pool_k16.pt}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-30}"

for path in "${REWARD_CALIBRATION_PATH}" "${STEP_REWARD_CALIBRATION_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing calibration: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${FIXED_EVAL_POOL_PATH}")" \
  "$(dirname "${FIXED_STEP_EVAL_POOL_PATH}")"

run_one() {
  local name="$1"
  local step_reward_weight="$2"
  local step_advantage_weight="$3"
  local run_dir="${OUTPUT_ROOT}/${name}"
  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to reuse output directory: ${run_dir}" >&2
    exit 1
  fi
  OUTPUT_DIR="${run_dir}" \
  MDM_DDPO_ENABLE_STEP_REWARD=1 \
  MDM_DDPO_REWARD_CALIBRATION_PATH="${REWARD_CALIBRATION_PATH}" \
  MDM_DDPO_STEP_REWARD_CALIBRATION_PATH="${STEP_REWARD_CALIBRATION_PATH}" \
  MDM_DDPO_FIXED_EVAL_POOL_PATH="${FIXED_EVAL_POOL_PATH}" \
  MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH="${FIXED_STEP_EVAL_POOL_PATH}" \
  MDM_DDPO_SWANLAB_RUN_NAME="${name}" \
  bash "${PROJECT_ROOT}/scripts/train_humanml_step.sh" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --step-reward-weight "${step_reward_weight}" \
    --advantage-step-weight "${step_advantage_weight}" \
    --fixed-eval-every 5 \
    --early-stop-patience 0 \
    --save-every 5
}

# S0 controls for the effect of adding step-labelled prompts without a step
# advantage. S1/S2 then increase only the calibrated step component weight.
run_one S0_mixed_no_step_adv 0.0 0.0
run_one S1_step_adv_0125 0.5 0.125
run_one S2_step_adv_0250 0.5 0.25

python "${PROJECT_ROOT}/tools/summarize_experiments.py" \
  "${OUTPUT_ROOT}/S0_mixed_no_step_adv" \
  "${OUTPUT_ROOT}/S1_step_adv_0125" \
  "${OUTPUT_ROOT}/S2_step_adv_0250" \
  --output-prefix "${OUTPUT_ROOT}/step_ablation_comparison"
