#!/usr/bin/env bash
set -euo pipefail

# Train a blur-aware OBB-geometry MobileNetV2 variant using raw plus fixed
# unsharp-masked views so we can test whether low-contrast hard cases improve.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_bluraware_obb_geometry_v34.log"
RUN_NAME="${RUN_NAME:-mobilenetv2_bluraware_obb_geometry_v34}"

INIT_MODEL_PATH="${INIT_MODEL_PATH:-${REPO_ROOT}/artifacts/training/mobilenetv2_obb_geometry_v32/model.keras}"

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

echo "[WRAPPER] Starting blur-aware OBB-geometry training." | tee "${LOG_FILE}"
echo "[WRAPPER] Init model: ${INIT_MODEL_PATH}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:   ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --run-name "${RUN_NAME}" \
  --model-family mobilenet_v2_bluraware_obb_geometry \
  --gauge-id littlegood_home_temp_gauge_c \
  --batch-size 4 \
  --epochs 16 \
  --learning-rate 5e-5 \
  --seed 21 \
  --mobilenet-backbone-trainable \
  --mobilenet-warmup-epochs 4 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --geometry-value-loss-weight 1.0 \
  --keypoint-heatmap-loss-weight 0.5 \
  --keypoint-coord-loss-weight 1.0 \
  --init-model "${INIT_MODEL_PATH}" \
  2>&1 | tee -a "${LOG_FILE}"
