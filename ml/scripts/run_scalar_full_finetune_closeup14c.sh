#!/usr/bin/env bash
set -euo pipefail

# Fine-tune from the current best scalar checkpoint with close-up 14C captures
# added to the training manifest.
#
# The 13 new captures (2026-04-18) were taken at close camera distance with the
# gauge at 14C. The board was inferring -8C on this framing because the fixed
# training crop was designed for far-camera images. Adding these close-up
# captures forces the model to generalise across camera distances.
#
# hard-case-repeat 3 upweights the close-up captures relative to the rest of
# the manifest so the model actually shifts on this new distribution.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_full_finetune_closeup14c.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_full_finetune_closeup14c.model.keras"

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

echo "[WRAPPER] Starting close-up 14C fine-tune."
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
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5_closeup14c.csv \
  --hard-case-repeat 3 \
  --edge-focus-strength 0.5 \
  --run-name scalar_full_finetune_closeup14c \
  2>&1 | tee "${LOG_FILE}"
