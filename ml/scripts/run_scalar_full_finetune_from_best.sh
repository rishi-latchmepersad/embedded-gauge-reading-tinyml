#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the best scalar MobileNetV2 model with the backbone trainable.
# This is a lighter hard-case strategy than the head-only pass: fewer repeats,
# lower learning rate, and a short run so we can see whether the hard frames move.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_full_finetune_from_best.log"

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

echo "[WRAPPER] Starting scalar full-backbone fine-tune from the best MobileNetV2 model."
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/finetune_scalar_from_best.py \
  --device gpu \
  --no-freeze-backbone \
  --hard-case-repeat 4 \
  --edge-focus-strength 1.25 \
  --epochs 6 \
  --learning-rate 5e-6 \
  --run-name scalar_full_finetune_from_best \
  2>&1 | tee "${LOG_FILE}"
