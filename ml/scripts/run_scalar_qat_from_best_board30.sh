#!/usr/bin/env bash
set -euo pipefail

# Quantization-aware fine-tune the strongest scalar MobileNetV2 model on the
# expanded hard-case set. The source model is copied into Linux-local storage
# first so the quantize/load path stays responsive under WSL.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_qat_from_best_board30.log"
RUN_NAME="scalar_qat_from_best_board30"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_piecewise_calibrated/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_qat_cache/scalar_full_finetune_from_best_board30_piecewise_calibrated.model.keras"
MODEL_OUT="${REPO_ROOT}/artifacts/training/${RUN_NAME}/model.keras"

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

echo "[WRAPPER] Starting scalar MobileNetV2 quantization-aware fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Output model: ${MODEL_OUT}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/train_scalar_qat_from_best.py \
  --base-model "${BASE_MODEL_LOCAL}" \
  --device gpu \
  --hard-case-manifest data/hard_cases_plus_board30.csv \
  --hard-case-repeat 8 \
  --edge-focus-strength 1.5 \
  --epochs 6 \
  --learning-rate 1e-6 \
  --run-name "${RUN_NAME}" \
  2>&1 | tee "${LOG_FILE}"

echo "[WRAPPER] Evaluating QAT model on the original hard-case set." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/eval_scalar_model_on_manifest.py \
  --model "${MODEL_OUT}" \
  --manifest data/hard_cases.csv \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating QAT model on the expanded board30 hard-case set." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/eval_scalar_model_on_manifest.py \
  --model "${MODEL_OUT}" \
  --manifest data/hard_cases_plus_board30.csv \
  2>&1 | tee -a "${LOG_FILE}"
