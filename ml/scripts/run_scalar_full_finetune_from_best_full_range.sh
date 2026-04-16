#!/usr/bin/env bash
set -euo pipefail

# Full-range fine-tune from the current best scalar checkpoint.
#
# This pass keeps the broad board-style coverage in the loop while using a
# modest edge bias so we improve the stubborn temperatures without turning the
# run into another narrow hard-case overfit.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_full_finetune_from_best_full_range.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_full_finetune_from_best_full_range.model.keras"

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

echo "[WRAPPER] Starting full-range MobileNetV2 fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --device gpu \
  --no-gpu-memory-growth \
  --mobilenet-backbone-trainable \
  --epochs 6 \
  --learning-rate 3e-7 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 2 \
  --edge-focus-strength 0.75 \
  --run-name scalar_full_finetune_from_best_full_range \
  2>&1 | tee "${LOG_FILE}"
