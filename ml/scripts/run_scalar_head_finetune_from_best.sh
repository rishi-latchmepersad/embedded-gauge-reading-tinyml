#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the best scalar MobileNetV2 model with extra weight on the newest hard board captures.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_head_finetune_from_best.log"

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

echo "[WRAPPER] Starting scalar head-only fine-tune from the best MobileNetV2 model."
echo "[WRAPPER] Log file: ${LOG_FILE}"

# Keep the job chatty so it does not look stuck while the first epoch is running.
"${POETRY_BIN}" run python -u scripts/finetune_scalar_from_best.py \
  --device gpu \
  --hard-case-repeat 4 \
  --epochs 8 \
  2>&1 | tee "${LOG_FILE}"
