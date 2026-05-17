#!/usr/bin/env bash
set -euo pipefail

# Train the needle-direction geometry bottleneck model with hard-case emphasis.
#
# The goal is to force the CNN to learn a stable needle vector first, then map
# that vector back to Celsius through the calibrated gauge geometry layer.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
RUN_NAME="${RUN_NAME:-mobilenetv2_direction_geometry_hardtail_v1}"
TRAIN_LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
HARD_MANIFEST="${HARD_MANIFEST:-${REPO_ROOT}/data/hard_cases_extreme_weighted_v4.csv}"
VAL_MANIFEST="${VAL_MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid.csv}"
TEST_MANIFEST="${TEST_MANIFEST:-${REPO_ROOT}/data/board_rectified_probe_20260422.csv}"
PRECOMPUTED_CROP_BOXES="${PRECOMPUTED_CROP_BOXES:-data/rectified_crop_boxes_v5_all.csv}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting direction-geometry hard-tail training." | tee "${TRAIN_LOG_FILE}"
echo "[WRAPPER] Log file: ${TRAIN_LOG_FILE}" | tee -a "${TRAIN_LOG_FILE}"

RECTIFY_ALL=1 "${POETRY_BIN}" run python -u scripts/run_training.py \
  --run-name "${RUN_NAME}" \
  --model-family mobilenet_v2_direction_geometry \
  --gauge-id littlegood_home_temp_gauge_c \
  --device gpu \
  --no-gpu-memory-growth \
  --batch-size 8 \
  --epochs 80 \
  --learning-rate 5e-5 \
  --seed 21 \
  --mobilenet-backbone-trainable \
  --mobilenet-warmup-epochs 8 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --hard-case-manifest "${HARD_MANIFEST}" \
  --hard-case-repeat 12 \
  --val-manifest "${VAL_MANIFEST}" \
  --test-manifest "${TEST_MANIFEST}" \
  --precomputed-crop-boxes "${PRECOMPUTED_CROP_BOXES}" \
  --range-aware-sampling \
  --cold-tail-fraction 0.20 \
  --hot-tail-fraction 0.20 \
  --oversampling-factor 4.0 \
  2>&1 | tee -a "${TRAIN_LOG_FILE}"
