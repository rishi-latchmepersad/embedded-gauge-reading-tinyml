#!/usr/bin/env bash
set -euo pipefail

# Head-only fine-tune from the current prod scalar checkpoint on the worst
# board-style misses. This keeps the run small while we push the large-error
# cases without disturbing the live board behavior too much.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_head_finetune_from_best_board_weak_focus.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_head_finetune_from_best_board_weak_focus.model.keras"

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

echo "[WRAPPER] Starting board weak-focus head-only fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/finetune_scalar_from_best.py \
  --base-model "${BASE_MODEL_LOCAL}" \
  --device gpu \
  --no-gpu-memory-growth \
  --hard-case-manifest data/board_weak_focus.csv \
  --hard-case-repeat 8 \
  --edge-focus-strength 1.5 \
  --epochs 6 \
  --learning-rate 5e-6 \
  --run-name scalar_head_finetune_from_best_board_weak_focus \
  2>&1 | tee "${LOG_FILE}"
