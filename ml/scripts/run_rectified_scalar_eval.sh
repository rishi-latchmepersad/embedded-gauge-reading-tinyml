#!/usr/bin/env bash
set -euo pipefail

# Evaluate the rectifier + scalar-reader chain on a labeled manifest.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/rectified_scalar_eval.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
RECTIFIER_MODEL="${RECTIFIER_MODEL:-artifacts/training/mobilenetv2_rectifier_hardcase_finetune_v3/model.keras}"
SCALAR_MODEL="${SCALAR_MODEL:-artifacts/deployment/mobilenetv2_rectified_scalar_finetune_v2_int8/model_int8.tflite}"
MANIFEST="${MANIFEST:-data/hard_cases_plus_board30_valid_with_new5.csv}"
RECTIFIER_CROP_SCALE="${RECTIFIER_CROP_SCALE:-1.80}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[RECT-EVAL] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

cd "${ROOT_DIR}"
echo "[RECT-EVAL] Starting rectified scalar evaluation." | tee "${LOG_FILE}"
echo "[RECT-EVAL] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# The rectified chain is evaluated on CPU to avoid WSL GPU driver stalls.
CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/eval_rectified_scalar_on_manifest.py \
  --rectifier-model "${RECTIFIER_MODEL}" \
  --scalar-model "${SCALAR_MODEL}" \
  --manifest "${MANIFEST}" \
  --rectifier-crop-scale "${RECTIFIER_CROP_SCALE}" \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
