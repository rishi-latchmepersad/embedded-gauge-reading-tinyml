#!/usr/bin/env bash
set -euo pipefail

# Train the compact CNN coarse-to-fine interval model on the full board-style
# sweep so we can compare a small CNN backbone against the current prod model.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/compact_interval_full_range.log"
RUN_NAME="${RUN_NAME:-compact_interval_full_range_v1}"

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

echo "[WRAPPER] Starting compact interval full-range training run."
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family compact_interval \
  --batch-size 8 \
  --learning-rate 3e-4 \
  --epochs 60 \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 6 \
  --edge-focus-strength 1.0 \
  --interval-bin-width 5.0 \
  --run-name "${RUN_NAME}" \
  2>&1 | tee "${LOG_FILE}"
