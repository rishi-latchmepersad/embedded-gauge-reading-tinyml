#!/usr/bin/env bash
set -euo pipefail

# Train the uncertainty-aware MobileNetV2 geometry reader on the full board-style mix.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_geometry_uncertainty_full_range.log"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting MobileNetV2 geometry uncertainty training."
echo "[WRAPPER] Log file: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="" "${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_geometry_uncertainty \
  --device gpu \
  --no-gpu-memory-growth \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 4 \
  --edge-focus-strength 1.0 \
  --epochs 16 \
  --learning-rate 5e-7 \
  --mobilenet-warmup-epochs 4 \
  --keypoint-heatmap-size 28 \
  --keypoint-heatmap-loss-weight 1.0 \
  --keypoint-coord-loss-weight 1.0 \
  --geometry-value-loss-weight 1.0 \
  --geometry-uncertainty-loss-weight 0.25 \
  --geometry-uncertainty-low-quantile 0.1 \
  --geometry-uncertainty-high-quantile 0.9 \
  --run-name mobilenetv2_geometry_uncertainty_full_range \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
