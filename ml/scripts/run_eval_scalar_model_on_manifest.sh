#!/usr/bin/env bash
set -euo pipefail

# Evaluate one saved scalar model on a CSV manifest and tee the results to a log.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/eval_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_scalar_hardcase_finetune_board30.eval.log"

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

echo "[WRAPPER] Evaluating manifest hard cases."
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/eval_scalar_model_on_manifest.py \
  --model artifacts/training/mobilenetv2_scalar_hardcase_finetune_board30/model.keras \
  --manifest data/hard_cases_plus_board30.csv \
  2>&1 | tee "${LOG_FILE}"
