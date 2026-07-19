#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-${PROJECT_ROOT}/outputs/step_k8_soft_counterfactual}"
POOL_PATH="${COUNTERFACTUAL_POOL_PATH:-${PROJECT_ROOT}/artifacts/step_counterfactual_number_pool.pt}"
OUTPUT_PATH="${COUNTERFACTUAL_OUTPUT_PATH:-${RUN_DIR}/counterfactual_number_probe.json}"
SAMPLES_PATH="${COUNTERFACTUAL_SAMPLES_PATH:-${RUN_DIR}/counterfactual_number_probe_samples.pt}"

shopt -s nullglob
checkpoints=("${RUN_DIR}"/checkpoint_*.pt)
if (( ${#checkpoints[@]} == 0 )); then
  echo "No checkpoint_*.pt files found under ${RUN_DIR}" >&2
  exit 1
fi

exec "${PYTHON:-python}" "${PROJECT_ROOT}/tools/probe_step_number_conditioning.py" \
  --output "${OUTPUT_PATH}" \
  --pool-path "${POOL_PATH}" \
  --samples-output "${SAMPLES_PATH}" \
  --conditions 24 \
  --samples-per-condition 2 \
  --batch-size 48 \
  --sample-steps 50 \
  --step-targets 1,2,3,4,5,6 \
  --number-style words \
  --device "${DEVICE:-cuda:0}" \
  --precision "${PRECISION:-bf16}" \
  --checkpoints "${checkpoints[@]}"
