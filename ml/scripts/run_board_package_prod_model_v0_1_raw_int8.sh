#!/usr/bin/env bash
set -euo pipefail

# Package the board-ready raw int8 production candidate and refresh the
# canonical xSPI2 blob used by the STM32 firmware loader.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/prod_model_v0_1_raw_int8_board_package.log"
MODEL_IN="${MODEL_IN:-artifacts/deployment/prod_model_v0.1_raw_int8/model_int8.tflite}"
CALIBRATION_JSON="${CALIBRATION_JSON:-artifacts/deployment/prod_model_v0.1_raw_int8/calibration_weighted.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runtime/prod_model_v0.1_raw_int8_reloc}"
WORK_ROOT="${WORK_ROOT:-${HOME}/prod_model_v0_1_raw_int8_board_package}"
BASE_MODEL_LOCAL="${WORK_ROOT}/model_int8.tflite"
PACKAGE_NAME="${PACKAGE_NAME:-scalar_full_finetune_from_best_piecewise_calibrated_int8}"

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
rm -rf "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}"

echo "[WRAPPER] Staging raw int8 model into ${WORK_ROOT}."
cp "${REPO_ROOT}/${MODEL_IN}" "${BASE_MODEL_LOCAL}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting board package for prod_model_v0.1_raw_int8."
echo "[WRAPPER] Model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Package name: ${PACKAGE_NAME}"
echo "[WRAPPER] Output dir: ${OUTPUT_DIR}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/package_scalar_model_for_n6.py \
  --model "${BASE_MODEL_LOCAL}" \
  --output-dir "${OUTPUT_DIR}" \
  --name "${PACKAGE_NAME}" \
  --compression high \
  --optimization balanced \
  2>&1 | tee "${LOG_FILE}"

echo "[WRAPPER] Re-scoring raw int8 model with weighted calibration on the original hard-case set." | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/eval_tflite_with_piecewise_calibration.py \
  --model "${REPO_ROOT}/${MODEL_IN}" \
  --manifest data/hard_cases.csv \
  --calibration-json "${REPO_ROOT}/${CALIBRATION_JSON}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Re-scoring raw int8 model with weighted calibration on the expanded board30-valid set." | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/eval_tflite_with_piecewise_calibration.py \
  --model "${REPO_ROOT}/${MODEL_IN}" \
  --manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --calibration-json "${REPO_ROOT}/${CALIBRATION_JSON}" \
  2>&1 | tee -a "${LOG_FILE}"
