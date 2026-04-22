#!/usr/bin/env bash
set -euo pipefail

# Iteration 4: extreme-value upweighting from prod base.
# Hypothesis: residual error is concentrated at temperature extremes
# (m30, m19, p50). The extreme-weighted manifest gives extremes (|val|>=25)
# a 4x in-manifest multiplier on top of --hard-case-repeat=4 = effective 16x.
# Mid hard cases stay at 4x. No heavy aug (v3 showed it hurt val_mae).
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_extremes_v4.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_extremes_v4.model.keras"

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

echo "[WRAPPER] scalar_extremes_v4 - prod base + extreme-weighted hard cases"
echo "[WRAPPER] Log file: ${LOG_FILE}"

PYTHONDONTWRITEBYTECODE=1 \
  "${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --device gpu \
  --batch-size 4 \
  --epochs 30 \
  --learning-rate 5e-7 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_extreme_weighted_v4.csv \
  --hard-case-repeat 4 \
  --val-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --edge-focus-strength 1.0 \
  --run-name scalar_extremes_v4 \
  2>&1 | tee "${LOG_FILE}"
