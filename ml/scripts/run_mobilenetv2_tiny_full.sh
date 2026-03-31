#!/usr/bin/env bash
set -euo pipefail

# Train the compressed MobileNetV2 variant intended to fit the STM32N6 memory map.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_tiny_224_full.log"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "[WRAPPER] Starting tiny MobileNetV2 training baseline."
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --device gpu \
  --model-family mobilenet_v2_tiny \
  --image-height 224 \
  --image-width 224 \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --hard-case-manifest data/hard_cases.csv \
  --hard-case-repeat 2 \
  --epochs 40 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --seed 21 \
  --run-name mobilenetv2_tiny_224_full \
  2>&1 | tee "${LOG_FILE}"
