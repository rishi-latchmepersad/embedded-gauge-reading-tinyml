#!/usr/bin/env bash
set -euo pipefail

# Sweep rectifier crop expansion factors on the board-probe manifest.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/rectified_scalar_crop_scale_board_probe_sweep.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
RECTIFIER_MODEL="${RECTIFIER_MODEL:-artifacts/deployment/mobilenetv2_rectifier_hardcase_finetune_v3_int8/model_int8.tflite}"
SCALAR_MODEL="${SCALAR_MODEL:-artifacts/deployment/mobilenetv2_rectified_scalar_finetune_v2_int8/model_int8.tflite}"
MANIFEST="${MANIFEST:-data/board_rectified_probe_20260422.csv}"
SCALES="${SCALES:-1.75 1.80 1.85 1.90 1.95}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[SWEEP] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

cd "${ROOT_DIR}"
echo "[SWEEP] Starting rectified scalar crop-scale sweep on board probe." | tee "${LOG_FILE}"
echo "[SWEEP] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "[SWEEP] Rectifier: ${RECTIFIER_MODEL}" | tee -a "${LOG_FILE}"
echo "[SWEEP] Scalar: ${SCALAR_MODEL}" | tee -a "${LOG_FILE}"
echo "[SWEEP] Manifest: ${MANIFEST}" | tee -a "${LOG_FILE}"

for scale in ${SCALES}; do
  echo "[SWEEP] scale=${scale} start" | tee -a "${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/eval_rectified_scalar_on_manifest.py \
    --rectifier-model "${RECTIFIER_MODEL}" \
    --scalar-model "${SCALAR_MODEL}" \
    --manifest "${MANIFEST}" \
    --rectifier-crop-scale "${scale}" \
    2>&1 | tee -a "${LOG_FILE}"
  echo "[SWEEP] scale=${scale} done" | tee -a "${LOG_FILE}"
done
