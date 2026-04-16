#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the scalar reader on rectifier-generated crops so stage 2 learns
# the same crop distribution it will see at inference time.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_finetune_v2.log"
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
echo "[WRAPPER] Starting MobileNetV2 rectified-scalar fine-tune v2." | tee "${LOG_FILE}"
echo "[WRAPPER] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --init-model artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras \
  --rectifier-model-path artifacts/training/mobilenetv2_rectifier_hardcase_finetune_v3/model.keras \
  --rectifier-crop-scale 1.50 \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 4 \
  --edge-focus-strength 0.75 \
  --device gpu \
  --no-gpu-memory-growth \
  --no-mobilenet-pretrained \
  --mobilenet-backbone-trainable \
  --mobilenet-warmup-epochs 0 \
  --learning-rate 3e-6 \
  --epochs 10 \
  --run-name mobilenetv2_rectified_scalar_finetune_v2 \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
