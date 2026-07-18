#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/stability_ablations}"
export OUTPUT_ROOT

RUNNER="${PROJECT_ROOT}/scripts/run_single_experiment.sh"

bash "${RUNNER}" A0_group_whiten group_whiten p25 3e-4 1e-4 42 0 30
bash "${RUNNER}" A1_group_centered group_centered p25 1e-4 1e-4 42 0 30
bash "${RUNNER}" A2_group_shrink_p25 group_shrink p25 1e-4 1e-4 42 0 30
bash "${RUNNER}" A3_group_shrink_p50 group_shrink p50 1e-4 1e-4 42 0 30
bash "${RUNNER}" A4_component_shrink_p25 component_shrink p25 1e-4 1e-4 42 0 30

"${PYTHON:-/home/zhiwei/anaconda3/envs/motionrft/bin/python}" \
  "${PROJECT_ROOT}/tools/summarize_experiments.py" \
  "${OUTPUT_ROOT}/A0_group_whiten" \
  "${OUTPUT_ROOT}/A1_group_centered" \
  "${OUTPUT_ROOT}/A2_group_shrink_p25" \
  "${OUTPUT_ROOT}/A3_group_shrink_p50" \
  "${OUTPUT_ROOT}/A4_component_shrink_p25" \
  --output-prefix "${OUTPUT_ROOT}/ablation_comparison"
