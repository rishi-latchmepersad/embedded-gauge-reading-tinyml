#!/usr/bin/env bash
set -euo pipefail

# Train a MobileNetV2 geometry-first model on the full range of gauge data.
#
# This run keeps the angle/keypoint problem explicit, lets hard cases land in
# the actual training split, and preserves symmetric tail sampling so the cold
# and hot ends both stay in play.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_geometry_full_range_v1.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/mobilenetv2_keypoint_geometry_clean/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_geometry_full_range_v1.model.keras"

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

echo "[WRAPPER] Starting MobileNetV2 geometry full-range training."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_geometry \
  --device gpu \
  --epochs 24 \
  --learning-rate 1e-6 \
  --seed 21 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30.csv \
  --hard-case-repeat 2 \
  --val-fraction 0.15 \
  --test-fraction 0.15 \
  --range-aware-sampling \
  --cold-tail-fraction 0.20 \
  --hot-tail-fraction 0.20 \
  --oversampling-factor 2.0 \
  --edge-focus-strength 1.5 \
  --mobilenet-warmup-epochs 1 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-alpha 1.0 \
  --mobilenet-head-units 128 \
  --mobilenet-head-dropout 0.2 \
  --keypoint-heatmap-size 28 \
  --keypoint-heatmap-loss-weight 3.0 \
  --keypoint-coord-loss-weight 4.0 \
  --geometry-value-loss-weight 0.0 \
  --run-name mobilenetv2_geometry_full_range_v1 \
  2>&1 | tee "${LOG_FILE}"
