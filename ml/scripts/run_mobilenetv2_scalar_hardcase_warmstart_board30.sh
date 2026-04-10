#!/usr/bin/env bash
set -euo pipefail

# Warm-start MobileNetV2 on the full labelled dataset plus the expanded hard-case set.
# This keeps the launch command simple and makes the run easy to repeat.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_scalar_hardcase_warmstart_board30.log"
BASE_MODEL="artifacts/training/wsl_mnv2_finetune_seed21/model.keras"

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

echo "[WRAPPER] Starting scalar MobileNetV2 warm-start hard-case fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --device gpu \
  --epochs 20 \
  --learning-rate 3e-6 \
  --hard-case-manifest data/hard_cases_plus_board30.csv \
  --hard-case-repeat 8 \
  --edge-focus-strength 1.5 \
  --init-model "${BASE_MODEL}" \
  --run-name mobilenetv2_scalar_hardcase_warmstart_board30 \
  2>&1 | tee "${LOG_FILE}"
