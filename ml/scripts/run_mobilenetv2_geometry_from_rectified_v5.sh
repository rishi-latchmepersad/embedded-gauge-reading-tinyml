#!/usr/bin/env bash
set -euo pipefail

# Fine-tune a geometry-first MobileNetV2 from the strong rectified v5 backbone.
#
# This is the first paper-inspired experiment in the v5 improvement plan:
# keep the proven scalar backbone weights, then force the model to learn
# explicit keypoint geometry plus the derived gauge value.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_geometry_from_rectified_v5.log"

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

echo "[WRAPPER] Starting geometry-from-rectified-v5 fine-tune."
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_geometry \
  --device gpu \
  --no-gpu-memory-growth \
  --no-mobilenet-pretrained \
  --epochs 12 \
  --learning-rate 1e-4 \
  --mobilenet-warmup-epochs 2 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-backbone-trainable \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
  --hard-case-repeat 6 \
  --edge-focus-strength 1.0 \
  --keypoint-heatmap-size 28 \
  --keypoint-heatmap-loss-weight 1.0 \
  --keypoint-coord-loss-weight 1.0 \
  --geometry-value-loss-weight 1.0 \
  --run-name mobilenetv2_geometry_from_rectified_v5 \
  2>&1 | tee "${LOG_FILE}"
