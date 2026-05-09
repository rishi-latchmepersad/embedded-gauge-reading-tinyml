#!/usr/bin/env bash
set -euo pipefail

# Adapt the preview-heavy rectified model to the hard-case manifest.
#
# This is a separate stage on top of the stronger v14 base so the tail samples
# can influence the backbone without being mixed into the main rectified fit.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_adapt_hardcases_v16.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
BASE_MODEL_SRC="${ROOT_DIR}/artifacts/training/mobilenetv2_rectified_scalar_from_hard_synth_previewaug_v14/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_rectified_scalar_from_hard_synth_previewaug_v14.model.keras"

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
: > "${LOG_FILE}"

cd "${ROOT_DIR}"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

echo "[WRAPPER] Starting hard-case adaptation from preview-heavy v14." | tee "${LOG_FILE}"
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:   ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --device gpu \
  --linear-output \
  --epochs 6 \
  --learning-rate 1e-6 \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 8 \
  --edge-focus-strength 1.5 \
  --run-name mobilenetv2_rectified_scalar_adapt_hardcases_v16 \
  2>&1 | tee -a "${LOG_FILE}"
