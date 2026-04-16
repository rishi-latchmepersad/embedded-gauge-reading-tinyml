#!/usr/bin/env bash
set -euo pipefail

# Warm-start the rectifier-first MobileNetV2 model from the current best checkpoint.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectifier_finetune.log"
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
echo "[WRAPPER] Starting MobileNetV2 rectifier fine-tune." | tee "${LOG_FILE}"
echo "[WRAPPER] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_rectifier \
  --init-model artifacts/training/mobilenetv2_rectifier_gpu_nopretrained/model.keras \
  --device gpu \
  --no-gpu-memory-growth \
  --no-mobilenet-pretrained \
  --mobilenet-backbone-trainable \
  --mobilenet-warmup-epochs 0 \
  --learning-rate 1e-5 \
  --epochs 12 \
  --run-name mobilenetv2_rectifier_finetune \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
