#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="/home/zhiwei/projects/motion-diffusion-model/save/humanml_trans_dec_512_bert_epsilon_linear50_minsnr5_xstart1_vel01"

export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/epsilon_linear50_minsnr5_150k_step_k8_short}"
export MDM_DDPO_REWARD_CALIBRATION_PATH="${MDM_DDPO_REWARD_CALIBRATION_PATH:-${PROJECT_ROOT}/reward_calibration_epsilon_linear50_minsnr5_150k.json}"
export MDM_DDPO_STEP_REWARD_CALIBRATION_PATH="${MDM_DDPO_STEP_REWARD_CALIBRATION_PATH:-${PROJECT_ROOT}/step_reward_k8_soft_huber_epsilon_linear50_minsnr5_150k.json}"
export MDM_DDPO_FIXED_EVAL_POOL_PATH="${MDM_DDPO_FIXED_EVAL_POOL_PATH:-${PROJECT_ROOT}/artifacts/humanml_val_fixed_eval_pool.pt}"
export MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH="${MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH:-${PROJECT_ROOT}/artifacts/step_val_fixed_eval_pool_k8_soft.pt}"
export MDM_DDPO_USE_SWANLAB="${MDM_DDPO_USE_SWANLAB:-0}"

exec bash "${PROJECT_ROOT}/scripts/train_step_soft_counterfactual.sh" \
  --epochs 6 \
  --seed 42 \
  --model-path "${MODEL_DIR}/model000150000.pt" \
  --model-args-path "${MODEL_DIR}/args.json" \
  --prediction-type epsilon \
  --learning-rate 3e-5 \
  --clip-range 1e-3 \
  --fixed-eval-every 1 \
  --early-stop-patience 0 \
  --save-every 1 \
  "$@"
