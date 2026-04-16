#!/usr/bin/env bash
set -euo pipefail

# Regression-focused fine-tune from the current full-range checkpoint.
#
# The previous full-range pass was close overall, but it still drifted on a few
# stubborn cases around 30C, 35C, 50C, and some low-end samples. This pass
# keeps the whole board-style sweep in mind while upweighting those regressions.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_full_finetune_from_best_full_range_regression_fix.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_full_range/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_full_finetune_from_best_full_range_regression_fix.model.keras"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${BASE_MODEL_LOCAL}")"
cp -f "${BASE_MODEL_SRC}" "${BASE_MODEL_LOCAL}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting full-range regression-fix fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --device gpu \
  --no-gpu-memory-growth \
  --mobilenet-backbone-trainable \
  --epochs 4 \
  --learning-rate 2e-7 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/full_range_regression_focus.csv \
  --hard-case-repeat 1 \
  --edge-focus-strength 0.5 \
  --run-name scalar_full_finetune_from_best_full_range_regression_fix \
  2>&1 | tee "${LOG_FILE}"
