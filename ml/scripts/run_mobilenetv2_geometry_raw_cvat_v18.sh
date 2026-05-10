#!/usr/bin/env bash
set -euo pipefail

# Train a geometry-first MobileNetV2 on the raw CVAT-labelled gauge images.
#
# The raw CVAT exports contain the center/tip/dial annotations we need for a
# geometry-aware model. This run is intended to learn better alignment and
# keypoint supervision directly from that source, rather than forcing the same
# data through the scalar-only head again.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_geometry_raw_cvat_v18.log"
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

cd "${ROOT_DIR}"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

ARTIFACTS_DIR="artifacts/training"
RUN_NAME="mobilenetv2_geometry_raw_cvat_v18"

echo "[WRAPPER] Starting geometry-first training on raw CVAT data v18." | tee "${LOG_FILE}"
echo "[WRAPPER] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Launching geometry training..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_geometry \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --run-name "${RUN_NAME}" \
  --epochs 24 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --mobilenet-warmup-epochs 2 \
  --mobilenet-unfreeze-last-n 6 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --keypoint-heatmap-size 28 \
  --keypoint-heatmap-loss-weight 1.0 \
  --keypoint-coord-loss-weight 1.0 \
  --geometry-value-loss-weight 1.0 \
  --device gpu \
  2>&1 | tee -a "${LOG_FILE}"
