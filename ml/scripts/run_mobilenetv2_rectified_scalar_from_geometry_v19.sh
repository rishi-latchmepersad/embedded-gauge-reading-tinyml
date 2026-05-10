#!/usr/bin/env bash
set -euo pipefail

# Retrain the strict rectified scalar model using geometry-derived crop boxes.
#
# This keeps the successful strict MobileNetV2 scalar recipe but replaces the
# crop source with the geometry model's tighter keypoint-informed boxes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_from_geometry_v19.log"
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

TRAIN_MANIFEST="data/full_labelled_plus_board30_valid_with_new5.csv"
HELD_OUT_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"
BOXES_CSV="data/geometry_crop_boxes_v18.csv"
INIT_MODEL="artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras"
OUTPUT_DIR="artifacts/training/mobilenetv2_rectified_scalar_from_geometry_v19"

echo "[WRAPPER] Starting geometry-derived rectified scalar fine-tune v19." | tee "${LOG_FILE}"
echo "[WRAPPER] Train manifest: ${TRAIN_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Held-out eval:  ${HELD_OUT_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:      ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init model:     ${INIT_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:       ${LOG_FILE}" | tee -a "${LOG_FILE}"

if [[ ! -f "${BOXES_CSV}" ]]; then
  echo "[WRAPPER] Geometry crop boxes not found; generating them first..." | tee -a "${LOG_FILE}"
  bash scripts/run_precompute_geometry_boxes.sh 2>&1 | tee -a "${LOG_FILE}"
fi

echo "[WRAPPER] Launching scalar fine-tune..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${TRAIN_MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${INIT_MODEL}" \
  --batch-size 4 \
  --epochs 16 \
  --warmup-epochs 6 \
  --learning-rate 1e-4 \
  --fine-tune-lr 2e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --seed 21 \
  2>&1 | tee -a "${LOG_FILE}"
