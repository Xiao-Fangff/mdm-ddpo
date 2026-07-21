#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPSILON_DIR="/home/zhiwei/projects/motion-diffusion-model/save/humanml_trans_dec_512_bert_epsilon_linear50_minsnr5_xstart1_vel01"

exec /home/zhiwei/anaconda3/envs/motionrft/bin/python \
  "${PROJECT_ROOT}/tools/train_count_conditioning_sft.py" \
  --model-path "${MODEL_PATH:-${EPSILON_DIR}/model000150000.pt}" \
  --model-args-path "${MODEL_ARGS_PATH:-${EPSILON_DIR}/args.json}" \
  --prediction-type auto \
  --output-dir "${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/epsilon_count_sft}" \
  --device "${DEVICE:-cuda:0}" \
  --precision "${PRECISION:-bf16}" \
  --epochs 3 \
  --steps-per-epoch 0 \
  --human-batch-size 32 \
  --step-batch-size 32 \
  --human-loss-weight 0.5 \
  --step-loss-weight 0.5 \
  --lora-learning-rate 3e-5 \
  --count-learning-rate 1e-3 \
  --length-bins 8 \
  --anti-jitter-auto-grad-ratio 0.1 \
  "$@"
