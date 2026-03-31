#!/usr/bin/env bash
set -euo pipefail

# Probe a range of MobileNetV2 widths and report the largest candidate that
# stays inside the STM32N6 internal relocatable memory pools.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/fit_search_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_fit_search.log"

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

echo "[WRAPPER] Starting MobileNetV2 fit search."
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/search_mobilenetv2_fit.py \
  "$@" 2>&1 | tee "${LOG_FILE}"
