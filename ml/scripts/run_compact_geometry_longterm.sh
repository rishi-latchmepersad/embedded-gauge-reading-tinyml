#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the compact CNN localizer as the tiny detector-style front end.
#
# This keeps the backbone lightweight, pins the board-style validation and test
# manifests, and focuses the run on sharper localization rather than the final
# scalar value head.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/compact_geometry_longterm.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/compact_geometry_full_range/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/compact_geometry_longterm.model.keras"

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

echo "[WRAPPER] Starting compact geometry long-term training."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family compact_geometry \
  --device gpu \
  --no-gpu-memory-growth \
  --epochs 20 \
  --learning-rate 5e-7 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/full_labelled_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 1 \
  --val-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
  --test-manifest data/board_rectified_probe_20260422.csv \
  --edge-focus-strength 1.0 \
  --keypoint-heatmap-size 56 \
  --keypoint-heatmap-loss-weight 3.0 \
  --keypoint-coord-loss-weight 4.0 \
  --geometry-value-loss-weight 0.0 \
  --run-name compact_geometry_longterm \
  2>&1 | tee "${LOG_FILE}"
