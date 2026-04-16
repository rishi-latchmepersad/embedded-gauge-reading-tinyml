#!/usr/bin/env bash
set -euo pipefail

# Fine-tune a geometry-first MobileNetV2 localizer for the keypoint-gated
# cascade. The goal is to make the heatmaps sharper and the coordinate
# estimates more stable, while leaving the scalar readout to the separate
# board-ready reader model.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_geometry_cascade_localizer.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting geometry cascade localizer fine-tune."
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_geometry \
  --device gpu \
  --no-gpu-memory-growth \
  --no-mobilenet-pretrained \
  --epochs 16 \
  --learning-rate 5e-7 \
  --mobilenet-warmup-epochs 2 \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 8 \
  --edge-focus-strength 1.0 \
  --keypoint-heatmap-size 56 \
  --keypoint-heatmap-loss-weight 3.0 \
  --keypoint-coord-loss-weight 4.0 \
  --geometry-value-loss-weight 0.0 \
  --run-name mobilenetv2_geometry_cascade_localizer \
  2>&1 | tee "${LOG_FILE}"
