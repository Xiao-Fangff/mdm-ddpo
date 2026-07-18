#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 BEST_RUN_DIR" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/zhiwei/anaconda3/envs/motionrft/bin/python}"
BEST_RUN="$(realpath "$1")"
if [[ ! -f "${BEST_RUN}/config.json" ]]; then
  echo "Missing ${BEST_RUN}/config.json" >&2
  exit 1
fi

mapfile -t CONFIG_VALUES < <(
  "${PYTHON}" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["advantage_mode"]); print(c["advantage_std_floor_quantile"]); print(c["learning_rate"]); print(c["clip_range"])' \
    "${BEST_RUN}/config.json"
)
ADVANTAGE_MODE="${CONFIG_VALUES[0]}"
FLOOR_QUANTILE="${CONFIG_VALUES[1]}"
LEARNING_RATE="${CONFIG_VALUES[2]}"
CLIP_RANGE="${CONFIG_VALUES[3]}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/anchor_seed_sweep}"
export OUTPUT_ROOT
RUN_DIRS=()
for ANCHOR_RATIO in 0 0.1 0.2; do
  RATIO_TOKEN="${ANCHOR_RATIO/./p}"
  for SEED in 42 43 44; do
    RUN_NAME="anchor${RATIO_TOKEN}_seed${SEED}"
    bash "${PROJECT_ROOT}/scripts/run_single_experiment.sh" \
      "${RUN_NAME}" \
      "${ADVANTAGE_MODE}" \
      "${FLOOR_QUANTILE}" \
      "${LEARNING_RATE}" \
      "${CLIP_RANGE}" \
      "${SEED}" \
      "${ANCHOR_RATIO}" \
      30
    RUN_DIRS+=("${OUTPUT_ROOT}/${RUN_NAME}")
  done
done

"${PYTHON}" "${PROJECT_ROOT}/tools/summarize_experiments.py" \
  "${RUN_DIRS[@]}" \
  --output-prefix "${OUTPUT_ROOT}/anchor_seed_comparison" \
  --aggregate-seeds
