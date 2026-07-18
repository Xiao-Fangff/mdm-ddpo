#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/zhiwei/anaconda3/envs/motionrft/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

exec "${PYTHON}" "${PROJECT_ROOT}/train_ddpo.py" \
  --preflight \
  --device "${DEVICE}" \
  --reward-device same \
  --output-dir "${PROJECT_ROOT}/outputs/preflight" \
  --fixed-eval-every 0 \
  "$@"
