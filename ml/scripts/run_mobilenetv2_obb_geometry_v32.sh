#!/usr/bin/env bash
set -euo pipefail

# Train the literature-style OBB + geometry model on the labeled gauge data.
# The run uses MobileNetV2 as a shared backbone, an OBB localizer branch, and
# an explicit keypoint/geometry branch so the model learns structure instead of
# only a scalar shortcut.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_obb_geometry_v32.log"
RUN_NAME="${RUN_NAME:-mobilenetv2_obb_geometry_v32}"

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
export CUDA_VISIBLE_DEVICES="-1"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

echo "[WRAPPER] Starting OBB-geometry training."
echo "[WRAPPER] Run name: ${RUN_NAME}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --run-name "${RUN_NAME}" \
  --model-family mobilenet_v2_obb_geometry \
  --gauge-id littlegood_home_temp_gauge_c \
  --image-height 224 \
  --image-width 224 \
  --batch-size 4 \
  --epochs 16 \
  --mobilenet-warmup-epochs 4 \
  --learning-rate 5e-5 \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --mobilenet-backbone-trainable \
  --mobilenet-unfreeze-last-n 12 \
  --geometry-value-loss-weight 1.0 \
  --keypoint-heatmap-loss-weight 0.5 \
  --keypoint-coord-loss-weight 1.0 \
  --edge-focus-strength 0.75 \
  --no-gpu-memory-growth \
  2>&1 | tee "${LOG_FILE}"
