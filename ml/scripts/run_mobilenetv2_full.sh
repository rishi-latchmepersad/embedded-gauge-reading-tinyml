#!/usr/bin/env bash
set -euo pipefail

# Run the strongest known MobileNetV2 preset from the WSL Poetry environment and tee logs.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_224_full.log"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "[WRAPPER] Starting MobileNetV2 full training baseline."
echo "[WRAPPER] Log file: ${LOG_FILE}"

# Unbuffered Python keeps the epoch logs visible in PowerShell while tee captures them.
~/.local/bin/poetry run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --device gpu \
  --run-name mobilenetv2_224_full \
  2>&1 | tee "${LOG_FILE}"
