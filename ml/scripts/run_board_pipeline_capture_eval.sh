#!/usr/bin/env bash
set -euo pipefail

# Replay the current STM32 board pipeline on the newest raw captures.
# This keeps the laptop-side run close to the firmware logs without requiring
# any of the board hardware to be attached.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/board_pipeline_capture_eval.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
OBB_MODEL="${OBB_MODEL:-artifacts/deployment/prod_model_v0.3_obb_int8/model_int8.tflite}"
RECTIFIER_MODEL="${RECTIFIER_MODEL:-artifacts/deployment/mobilenetv2_rectifier_hardcase_finetune_v3_int8/model_int8.tflite}"
SCALAR_MODEL="${SCALAR_MODEL:-artifacts/deployment/scalar_full_finetune_from_best_piecewise_calibrated_int8/model_int8.tflite}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[PIPE] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

cd "${ROOT_DIR}"
echo "[PIPE] Starting board-pipeline replay." | tee "${LOG_FILE}"
echo "[PIPE] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# Run on CPU so the replay stays stable on WSL and matches the board's tensor
# ordering and quantization behaviour more closely.
CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/eval_board_pipeline_on_captures.py \
  --obb-model "${OBB_MODEL}" \
  --rectifier-model "${RECTIFIER_MODEL}" \
  --scalar-model "${SCALAR_MODEL}" \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"

