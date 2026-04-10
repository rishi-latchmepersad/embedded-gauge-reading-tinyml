#!/usr/bin/env bash
set -euo pipefail

# Fit a piecewise calibration layer using the training crop pipeline plus the
# clean board-crop manifest. This keeps the calibration in the same image
# domain the deployed model actually sees.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
MODEL_IN="${MODEL_IN:-artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras}"
BOARD_MANIFEST="${BOARD_MANIFEST:-data/hard_cases_plus_board30_valid_with_new5.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_crop_q8}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_full_finetune_from_best_board30_clean_plus_new5_calibrated_crop_q8.log"
WORK_ROOT="${WORK_ROOT:-${HOME}/calibration_scalar_full_board30_crop_domain}"

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
cp "${REPO_ROOT}/${BOARD_MANIFEST}" "${WORK_ROOT}/board_manifest.csv"

cd "${REPO_ROOT}"

echo "[CAL] Loading from staged model: ${WORK_ROOT}/model.keras"
echo "[CAL] Writing output to: ${REPO_ROOT}/${OUTPUT_DIR}"
echo "[CAL] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/calibrate_scalar_model_crop_domain.py \
  --mode spline \
  --spline-knot-mode quantile \
  --spline-knot-count 8 \
  --model "${WORK_ROOT}/model.keras" \
  --board-manifest "${WORK_ROOT}/board_manifest.csv" \
  --output-dir "${WORK_ROOT}/output" \
  2>&1 | tee "${LOG_FILE}"

rm -rf "${REPO_ROOT}/${OUTPUT_DIR}"
cp -r "${WORK_ROOT}/output" "${REPO_ROOT}/${OUTPUT_DIR}"
echo "[CAL] Copied calibrated artifact to ${REPO_ROOT}/${OUTPUT_DIR}"

echo "[CAL] Evaluating calibrated model on original hard cases."
"${POETRY_BIN}" run python -u scripts/eval_scalar_model_on_manifest.py \
  --model "${REPO_ROOT}/${OUTPUT_DIR}/model.keras" \
  --manifest data/hard_cases.csv \
  2>&1 | tee -a "${LOG_FILE}"

echo "[CAL] Evaluating calibrated model on expanded board30 hard cases."
"${POETRY_BIN}" run python -u scripts/eval_scalar_model_on_manifest.py \
  --model "${REPO_ROOT}/${OUTPUT_DIR}/model.keras" \
  --manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  2>&1 | tee -a "${LOG_FILE}"
