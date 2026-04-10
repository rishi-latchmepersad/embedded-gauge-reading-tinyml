#!/usr/bin/env bash
set -euo pipefail

# Fit a piecewise-linear calibration layer on top of the best board30 scalar model.
# The source model and manifest are staged into WSL-local storage to keep the fit
# responsive and avoid repeated reads from the Windows-mounted drive.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
MODEL_IN="${MODEL_IN:-artifacts/training/scalar_full_finetune_from_best_board30_piecewise_calibrated/model.keras}"
MANIFEST_IN="${MANIFEST_IN:-data/hard_cases_plus_board30.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/training/scalar_full_finetune_from_best_board30_piecewise_calibrated_plus}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_full_finetune_from_best_board30_piecewise_calibration_plus.log"
WORK_ROOT="${WORK_ROOT:-${HOME}/calibration_scalar_full_board30_piecewise}"

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
cp "${REPO_ROOT}/${MANIFEST_IN}" "${WORK_ROOT}/hard_cases_plus_board30.csv"

cd "${REPO_ROOT}"

echo "[CAL] Loading from staged model: ${WORK_ROOT}/model.keras"
echo "[CAL] Writing output to: ${REPO_ROOT}/${OUTPUT_DIR}"
echo "[CAL] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/calibrate_scalar_model.py \
  --mode spline \
  --spline-knot-mode all \
  --model "${WORK_ROOT}/model.keras" \
  --manifest "${WORK_ROOT}/hard_cases_plus_board30.csv" \
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
  --manifest data/hard_cases_plus_board30.csv \
  2>&1 | tee -a "${LOG_FILE}"
