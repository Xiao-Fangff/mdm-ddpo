#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DDPO_CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MDM_ROOT="${MDM_ROOT:-/home/zhiwei/projects/motion-diffusion-model}"
MOTIONRFT_ROOT="${MOTIONRFT_ROOT:-/home/zhiwei/projects/MotionRFT}"
PYTHON="${PYTHON:-/home/zhiwei/anaconda3/envs/motionrft/bin/python}"
CHECKPOINT="$(realpath "$1")"
OUTPUT_DIR="$(realpath -m "$2")"
BASELINE_DIR="${OUTPUT_DIR}/baseline"
CANDIDATE_DIR="${OUTPUT_DIR}/candidate"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite evaluation directory: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${BASELINE_DIR}" "${CANDIDATE_DIR}"

cp "${MDM_ROOT}/save/humanml_trans_dec_512_bert/model000600000.pt" \
  "${BASELINE_DIR}/model000600000.pt"
cp "${MDM_ROOT}/save/humanml_trans_dec_512_bert/args.json" \
  "${BASELINE_DIR}/args.json"

"${PYTHON}" "${PROJECT_ROOT}/export_ddpo.py" \
  --checkpoint "${CHECKPOINT}" \
  --output "${CANDIDATE_DIR}/model_ddpo.pt"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mdm-ddpo}"
export PYTHON
cd "${MOTIONRFT_ROOT}"
bash RFT_MDM_v2/run_eval_mdm.sh \
  --model-path "${BASELINE_DIR}/model000600000.pt" \
  --device "${EVAL_DEVICE:-0}" \
  --eval-mode "${EVAL_MODE:-debug}" \
  --guidance-param 2.5 \
  --cache-dir "${OUTPUT_DIR}/cache_baseline"
bash RFT_MDM_v2/run_eval_mdm.sh \
  --model-path "${CANDIDATE_DIR}/model_ddpo.pt" \
  --device "${EVAL_DEVICE:-0}" \
  --eval-mode "${EVAL_MODE:-debug}" \
  --guidance-param 2.5 \
  --cache-dir "${OUTPUT_DIR}/cache_candidate"
