#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the rectifier model with the labeled hard-case mix upweighted.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectifier_hardcase_finetune.log"
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
echo "[WRAPPER] Starting MobileNetV2 rectifier hard-case fine-tune." | tee "${LOG_FILE}"
echo "[WRAPPER] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_rectifier \
  --init-model artifacts/training/mobilenetv2_rectifier_finetune/model.keras \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 8 \
  --edge-focus-strength 1.75 \
  --device gpu \
  --no-gpu-memory-growth \
  --no-mobilenet-pretrained \
  --mobilenet-backbone-trainable \
  --mobilenet-warmup-epochs 0 \
  --learning-rate 3e-6 \
  --epochs 10 \
  --run-name mobilenetv2_rectifier_hardcase_finetune_v2 \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
