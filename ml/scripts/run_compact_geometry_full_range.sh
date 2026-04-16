#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the compact CNN geometry-first model on the full board-style mix.
#
# This run starts from the strongest compact CNN baseline we have, then lets
# the geometry head learn explicit keypoints and a deterministic gauge value
# over the full sweep.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/compact_geometry_full_range.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/compact_256_baseline/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/compact_geometry_base.model.keras"

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

echo "[WRAPPER] Starting compact geometry full-range training."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family compact_geometry \
  --device gpu \
  --epochs 60 \
  --learning-rate 3e-5 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 6 \
  --edge-focus-strength 1.0 \
  --keypoint-heatmap-size 28 \
  --keypoint-heatmap-loss-weight 1.0 \
  --keypoint-coord-loss-weight 1.0 \
  --geometry-value-loss-weight 1.0 \
  --run-name compact_geometry_full_range \
  2>&1 | tee "${LOG_FILE}"
