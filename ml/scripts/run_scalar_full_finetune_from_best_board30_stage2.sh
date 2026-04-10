#!/usr/bin/env bash
set -euo pipefail

# Run a tighter second-stage fine-tune from the first board30 checkpoint.
# We copy the base model to the Linux filesystem first so WSL does not stall
# while loading it from the Windows mount.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_full_finetune_from_best_board30_stage2.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_full_finetune_from_best_board30.model.keras"

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

echo "[WRAPPER] Starting second-stage scalar MobileNetV2 fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/finetune_scalar_from_best.py \
  --base-model "${BASE_MODEL_LOCAL}" \
  --device gpu \
  --no-freeze-backbone \
  --hard-case-manifest data/hard_cases_plus_board30.csv \
  --hard-case-repeat 16 \
  --edge-focus-strength 2.0 \
  --epochs 6 \
  --learning-rate 1e-6 \
  --run-name scalar_full_finetune_from_best_board30_stage2 \
  2>&1 | tee "${LOG_FILE}"
