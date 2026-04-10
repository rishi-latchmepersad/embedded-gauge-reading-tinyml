#!/usr/bin/env bash
set -euo pipefail

# Focused warm-start on the remaining hard misses, keeping the repeat count
# moderate so we improve the weak cases without overfitting a single sample.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_full_finetune_from_best_board30_focus_remaining.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new4/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_full_finetune_from_best_board30_focus_remaining.model.keras"

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

echo "[WRAPPER] Starting focus-remaining board30 MobileNetV2 fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --device gpu \
  --epochs 5 \
  --learning-rate 1.5e-7 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_remaining_focus.csv \
  --hard-case-repeat 8 \
  --edge-focus-strength 1.25 \
  --run-name scalar_full_finetune_from_best_board30_focus_remaining \
  2>&1 | tee "${LOG_FILE}"
