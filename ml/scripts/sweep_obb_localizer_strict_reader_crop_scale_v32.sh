#!/usr/bin/env bash
set -euo pipefail

# Sweep OBB crop scaling factors against the hard-case manifest.
#
# This is a small tuning pass over the two-stage OBB -> scalar cascade. The
# localizer stays fixed; we only change how aggressively we expand the OBB box
# before handing it to the reader.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/obb_localizer_strict_reader_crop_scale_sweep_v32.log"
MANIFEST="${MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid_with_new5.csv}"
OBB_MODEL="${OBB_MODEL:-${REPO_ROOT}/artifacts/deployment/prod_model_v0.3_obb_int8/model_int8.tflite}"
SCALAR_MODEL="${SCALAR_MODEL:-${REPO_ROOT}/artifacts/deployment/mobilenetv2_rectified_scalar_finetune_v2_int8/model_int8.tflite}"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[SWEEP] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="-1"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

SCALES=(1.05 1.10 1.20 1.30 1.40)
MIN_CROP_SIZE=48.0

echo "[SWEEP] Starting OBB crop-scale sweep." | tee -a "${LOG_FILE}"
echo "[SWEEP] OBB model: ${OBB_MODEL}" | tee -a "${LOG_FILE}"
echo "[SWEEP] Scalar model: ${SCALAR_MODEL}" | tee -a "${LOG_FILE}"
echo "[SWEEP] Manifest: ${MANIFEST}" | tee -a "${LOG_FILE}"
echo "[SWEEP] Scales: ${SCALES[*]}" | tee -a "${LOG_FILE}"

for scale in "${SCALES[@]}"; do
  OUT_DIR="${REPO_ROOT}/artifacts/cascade_eval/obb_localizer_strict_reader_v32_scale_${scale}"
  echo "[SWEEP] Running scale=${scale}" | tee -a "${LOG_FILE}"
  "${POETRY_BIN}" run python -u scripts/eval_obb_scalar_on_manifest.py \
    --obb-model "${OBB_MODEL}" \
    --obb-model-kind tflite \
    --scalar-model "${SCALAR_MODEL}" \
    --scalar-model-kind tflite \
    --manifest "${MANIFEST}" \
    --output-dir "${OUT_DIR}" \
    --obb-crop-scale "${scale}" \
    --min-crop-size "${MIN_CROP_SIZE}" \
    2>&1 | tee -a "${LOG_FILE}"
done

echo "[SWEEP] Sweep complete." | tee -a "${LOG_FILE}"
