#!/usr/bin/env bash
set -euo pipefail

# Export the prod-v0.4 scalar winner to board-ready TFLite artifacts.
# The winning model is staged into WSL-local storage so TensorFlow does not
# stall on /mnt/d during export.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/prod_model_v0_4_scalar_board_export.log"
MODEL_IN="${MODEL_IN:-artifacts/deployment/prod_model_v0.4_scalar_int8/model.keras}"
MANIFEST_IN="${MANIFEST_IN:-data/hard_cases_plus_board30.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/deployment/prod_model_v0.4_scalar_int8}"
WORK_ROOT="${WORK_ROOT:-${HOME}/prod_model_v0_4_scalar_board_export}"
BASE_MODEL_LOCAL="${WORK_ROOT}/model.keras"

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

echo "[WRAPPER] Staging prod-v0.4 scalar clean model into ${WORK_ROOT}."
cp "${REPO_ROOT}/${MODEL_IN}" "${BASE_MODEL_LOCAL}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting board export for prod_model_v0.4_scalar_int8."
echo "[WRAPPER] Model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Manifest: ${MANIFEST_IN}"
echo "[WRAPPER] Output dir: ${OUTPUT_DIR}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/export_board_artifacts.py \
  --model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest "${MANIFEST_IN}" \
  --output-dir "${OUTPUT_DIR}" \
  --representative-count 64 \
  2>&1 | tee "${LOG_FILE}"

echo "[WRAPPER] Evaluating exported int8 model on the original hard-case set." | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/eval_scalar_tflite_on_manifest.py \
  --model "${REPO_ROOT}/${OUTPUT_DIR}/model_int8.tflite" \
  --manifest data/hard_cases.csv \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating exported int8 model on the expanded board30 hard-case set." | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/eval_scalar_tflite_on_manifest.py \
  --model "${REPO_ROOT}/${OUTPUT_DIR}/model_int8.tflite" \
  --manifest data/hard_cases_plus_board30.csv \
  2>&1 | tee -a "${LOG_FILE}"
