#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the detector-first MobileNetV2 model on the clean labels.
#
# This run keeps the paper-aligned structure:
# - predict the keypoint heatmaps
# - derive the gauge value from the detected geometry
# - let hard cases reinforce the same geometry-to-value mapping
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_detector_geometry.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/mobilenetv2_keypoint_geometry_clean/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_detector_geometry.model.keras"

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

echo "[WRAPPER] Starting detector-first MobileNetV2 fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_detector \
  --device gpu \
  --epochs 8 \
  --learning-rate 1e-6 \
  --mobilenet-warmup-epochs 1 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 4 \
  --edge-focus-strength 1.0 \
  --keypoint-heatmap-size 28 \
  --keypoint-heatmap-loss-weight 1.0 \
  --run-name mobilenetv2_detector_geometry \
  2>&1 | tee "${LOG_FILE}"
