#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the compact CNN geometry model on the real hard-case pool.
#
# This keeps the explicit keypoint/angle formulation, uses the hard-case
# manifests that actually contain the cold and hot ends, and leaves the split
# random so those examples can land in training instead of being filtered out.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/compact_geometry_hardcases_full_range_v1.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/compact_geometry_full_range/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/compact_geometry_hardcases_full_range_v1.model.keras"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${BASE_MODEL_LOCAL}")"
cp -f "${BASE_MODEL_SRC}" "${BASE_MODEL_LOCAL}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting compact geometry hard-case full-range training."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family compact_geometry \
  --device gpu \
  --epochs 20 \
  --learning-rate 5e-7 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30.csv \
  --hard-case-repeat 3 \
  --range-aware-sampling \
  --cold-tail-fraction 0.20 \
  --hot-tail-fraction 0.20 \
  --oversampling-factor 2.0 \
  --edge-focus-strength 1.5 \
  --keypoint-heatmap-size 56 \
  --keypoint-heatmap-loss-weight 3.0 \
  --keypoint-coord-loss-weight 4.0 \
  --geometry-value-loss-weight 0.0 \
  --run-name compact_geometry_hardcases_full_range_v1 \
  2>&1 | tee "${LOG_FILE}"
