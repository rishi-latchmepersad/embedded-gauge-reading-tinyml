#!/usr/bin/env bash
set -euo pipefail

# Train a compact CNN on the board30 hard-case mix so we can compare
# a smaller, potentially more quantization-friendly architecture against
# the MobileNetV2 scalar runs.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/compact_board30_hardcase.log"
RUN_NAME="${RUN_NAME:-compact_board30_hardcase_v1}"

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

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting compact board30 training run."
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family compact \
  --hard-case-manifest data/hard_cases_plus_board30.csv \
  --hard-case-repeat 8 \
  --edge-focus-strength 1.5 \
  --epochs 60 \
  --run-name "${RUN_NAME}" \
  2>&1 | tee "${LOG_FILE}"
