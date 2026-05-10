#!/usr/bin/env bash
set -euo pipefail

# Train a pure OBB localizer for the first stage of the two-stage cascade.
#
# The goal here is not to predict the gauge value directly. We only want the
# model to learn a stable oriented crop proposal that can feed the separate
# scalar reader stage.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_obb_localizer_v31.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/mobilenetv2_obb_longterm/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_obb_localizer_v31.model.keras"

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
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

echo "[WRAPPER] Starting OBB localizer fine-tune."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_obb \
  --device gpu \
  --no-gpu-memory-growth \
  --epochs 16 \
  --batch-size 4 \
  --learning-rate 5e-6 \
  --mobilenet-warmup-epochs 2 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-backbone-trainable \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --val-fraction 0.15 \
  --test-fraction 0.15 \
  --run-name mobilenetv2_obb_localizer_v31 \
  2>&1 | tee "${LOG_FILE}"
