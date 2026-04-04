#!/usr/bin/env bash
set -euo pipefail

# Calibrate the best scalar MobileNetV2 model with a small affine head fit.
# The model is staged into the WSL Linux filesystem first so TensorFlow does
# not spend extra time reading the large Keras artifact from the mounted drive.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
MODEL_IN="${MODEL_IN:-artifacts/training/scalar_full_finetune_from_best/model.keras}"
MANIFEST_IN="${MANIFEST_IN:-data/hard_cases.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/training/scalar_full_finetune_from_best_calibrated}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_full_finetune_from_best_calibration.log"
WORK_ROOT="${WORK_ROOT:-${HOME}/calibration_scalar_full}"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[CAL] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
rm -rf "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}"

echo "[CAL] Staging inputs into ${WORK_ROOT}."
cp "${REPO_ROOT}/${MODEL_IN}" "${WORK_ROOT}/model.keras"
cp "${REPO_ROOT}/${MANIFEST_IN}" "${WORK_ROOT}/hard_cases.csv"

cd "${REPO_ROOT}"

echo "[CAL] Loading from staged model: ${WORK_ROOT}/model.keras"
echo "[CAL] Writing output to: ${REPO_ROOT}/${OUTPUT_DIR}"
echo "[CAL] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/calibrate_scalar_model.py \
  --model "${WORK_ROOT}/model.keras" \
  --manifest "${WORK_ROOT}/hard_cases.csv" \
  --output-dir "${WORK_ROOT}/output" \
  2>&1 | tee "${LOG_FILE}"

rm -rf "${REPO_ROOT}/${OUTPUT_DIR}"
cp -r "${WORK_ROOT}/output" "${REPO_ROOT}/${OUTPUT_DIR}"
echo "[CAL] Copied calibrated artifact to ${REPO_ROOT}/${OUTPUT_DIR}"
