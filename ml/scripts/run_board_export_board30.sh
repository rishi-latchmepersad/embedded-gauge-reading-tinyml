#!/usr/bin/env bash
set -euo pipefail

# Export the board30-calibrated scalar model to TFLite with a richer representative set.
# The model is staged into WSL-local storage so TensorFlow does not stall on /mnt/d.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/board_export_board30.log"
MODEL_IN="${MODEL_IN:-artifacts/training/scalar_full_finetune_from_best_board30_piecewise_calibrated/model.keras}"
MANIFEST_IN="${MANIFEST_IN:-data/hard_cases_plus_board30.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/deployment/scalar_full_finetune_from_best_board30_piecewise_calibrated_int8_board30}"
WORK_ROOT="${WORK_ROOT:-${HOME}/board_export_board30}"
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

echo "[WRAPPER] Staging model into ${WORK_ROOT}."
cp "${REPO_ROOT}/${MODEL_IN}" "${BASE_MODEL_LOCAL}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting board30 export for the calibrated scalar model."
echo "[WRAPPER] Model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Manifest: ${MANIFEST_IN}"
echo "[WRAPPER] Output dir: ${OUTPUT_DIR}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/export_board_artifacts.py \
  --model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest "${MANIFEST_IN}" \
  --output-dir "${OUTPUT_DIR}" \
  --representative-count 64 \
  2>&1 | tee "${LOG_FILE}"

echo "[WRAPPER] Evaluating exported int8 model on the original hard-case set." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/eval_scalar_tflite_on_manifest.py \
  --model "${REPO_ROOT}/${OUTPUT_DIR}/model_int8.tflite" \
  --manifest data/hard_cases.csv \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating exported int8 model on the expanded board30 hard-case set." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/eval_scalar_tflite_on_manifest.py \
  --model "${REPO_ROOT}/${OUTPUT_DIR}/model_int8.tflite" \
  --manifest data/hard_cases_plus_board30.csv \
  2>&1 | tee -a "${LOG_FILE}"
