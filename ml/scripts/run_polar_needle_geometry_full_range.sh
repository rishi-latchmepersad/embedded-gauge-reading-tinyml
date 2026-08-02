#!/usr/bin/env bash
if [[ "${WSL_GUARDED:-0}" != "1" ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/run_wsl_guarded.sh" env WSL_GUARDED=1 bash "${BASH_SOURCE[0]}" "$@"
fi
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="artifacts/training/polar_needle_geometry_full_range_v1"
INIT_MODEL="artifacts/training/polar_v3_geometry_smoke/model.keras"

poetry run python scripts/train_polar_needle.py \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 80 \
  --batch-size 8 \
  --learning-rate 0.0002 \
  --mask-loss-weight 2.0 \
  --value-loss-weight 0.1 \
  --weak-pseudo-labels \
  --weak-label-weight 0.35 \
  --weak-mask-sigma-multiplier 1.5 \
  --init-model "${INIT_MODEL}"
