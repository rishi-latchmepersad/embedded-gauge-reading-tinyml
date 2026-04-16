#!/usr/bin/env bash
set -euo pipefail

# Train the rectifier-first MobileNetV2 model in WSL with a fresh session.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectifier_full.log"
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
echo "[WRAPPER] Starting MobileNetV2 rectifier training." | tee "${LOG_FILE}"
echo "[WRAPPER] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_rectifier \
  --device gpu \
  --no-gpu-memory-growth \
  --mobilenet-warmup-epochs 4 \
  --mobilenet-backbone-trainable \
  --learning-rate 5e-5 \
  --epochs 24 \
  --run-name mobilenetv2_rectifier_full \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
