#!/usr/bin/env bash
set -euo pipefail

# Iteration 3: scalar fine-tune from prod with HEAVY photometric augmentation.
# No rectifier. Fixed crop. Targets prod's biggest failures (extreme values,
# under/overexposed board captures).
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_aug_heavy_v3.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_aug_heavy_v3.model.keras"

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

echo "[WRAPPER] scalar_aug_heavy_v3 \u2014 prod base + AUG_HEAVY=1, fixed crop, no rectifier"
echo "[WRAPPER] Log file: ${LOG_FILE}"

AUG_HEAVY=1 PYTHONDONTWRITEBYTECODE=1 \
  "${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --device gpu \
  --batch-size 4 \
  --epochs 25 \
  --learning-rate 1e-6 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 4 \
  --val-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --edge-focus-strength 1.0 \
  --run-name scalar_aug_heavy_v3 \
  2>&1 | tee "${LOG_FILE}"
