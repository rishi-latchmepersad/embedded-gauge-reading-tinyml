#!/usr/bin/env bash
set -euo pipefail

# Evaluate the OBB localizer plus the strict scalar reader on the hard-case mix.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/obb_localizer_strict_reader_cascade_v31.log"
OUTPUT_DIR="${REPO_ROOT}/artifacts/cascade_eval/obb_localizer_strict_reader_v31"

OBB_MODEL="${OBB_MODEL:-${REPO_ROOT}/artifacts/deployment/prod_model_v0.3_obb_int8/model_int8.tflite}"
SCALAR_MODEL="${SCALAR_MODEL:-${REPO_ROOT}/artifacts/deployment/mobilenetv2_rectified_scalar_finetune_v2_int8/model_int8.tflite}"
SCALAR_MODEL_KIND="${SCALAR_MODEL_KIND:-auto}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid_with_new5.csv}"
OBB_CROP_SCALE="${OBB_CROP_SCALE:-1.30}"
SCALAR_MODEL_LOCAL="${SCALAR_MODEL_LOCAL:-${HOME}/ml_eval_cache/mobilenetv2_rectified_scalar_reader_v31.tflite}"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"
: > "${LOG_FILE}"

# Keep the reader local when we are using a large model file from /mnt/d, but
# TFLite is already cheap to load so the copy is mostly a guardrail.
mkdir -p "$(dirname "${SCALAR_MODEL_LOCAL}")"
cp -f "${SCALAR_MODEL}" "${SCALAR_MODEL_LOCAL}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="-1"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

echo "[WRAPPER] Starting OBB localizer + strict scalar reader cascade evaluation."
echo "[WRAPPER] OBB model: ${OBB_MODEL}"
echo "[WRAPPER] Scalar model: ${SCALAR_MODEL_LOCAL}"
echo "[WRAPPER] Scalar model kind: ${SCALAR_MODEL_KIND}"
echo "[WRAPPER] Manifest: ${MANIFEST}"
echo "[WRAPPER] OBB crop scale: ${OBB_CROP_SCALE}"
echo "[WRAPPER] Output dir: ${OUTPUT_DIR}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/eval_obb_scalar_on_manifest.py \
  --obb-model "${OBB_MODEL}" \
  --scalar-model "${SCALAR_MODEL_LOCAL}" \
  --scalar-model-kind "${SCALAR_MODEL_KIND}" \
  --manifest "${MANIFEST}" \
  --obb-crop-scale "${OBB_CROP_SCALE}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"
