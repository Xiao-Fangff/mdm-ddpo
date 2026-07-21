#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/epsilon_count_only_step_k8_short}"

exec bash "${PROJECT_ROOT}/scripts/train_epsilon_count_sft_step_k8_short.sh" \
  --no-train-lora \
  --learning-rate 1e-4 \
  --clip-range 1e-3 \
  "$@"
