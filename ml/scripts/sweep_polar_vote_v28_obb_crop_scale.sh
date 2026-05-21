#!/usr/bin/env bash
set -euo pipefail

# Sweep OBB crop scales for the exact polar-v28 replay.
#
# This keeps the deployed OBB localizer fixed and only tunes how much context
# the exact V28 reader sees after decoding the OBB box. The goal is to find the
# best crop scale for the current exported board artifact before we flash.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/polar_vote_v28_obb_crop_scale_sweep.log"
MANIFEST="${MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid_with_new6.csv}"
OBB_MODEL="${OBB_MODEL:-${REPO_ROOT}/artifacts/deployment/prod_model_v0.3_obb_int8/model_int8.tflite}"
RECTIFIER_MODEL="${RECTIFIER_MODEL:-${REPO_ROOT}/artifacts/deployment/mobilenetv2_rectifier_hardcase_finetune_v3_int8/model_int8.tflite}"
SCALES=(${SCALES:-0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90})
MIN_CROP_SIZE="${MIN_CROP_SIZE:-48.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/d/Projects/embedded-gauge-reading-tinyml/tmp/polar_vote_v28_obb_crop_scale_sweep}"

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
echo "[SWEEP] Starting exact V28 OBB crop-scale sweep." | tee -a "${LOG_FILE}"
echo "[SWEEP] OBB model: ${OBB_MODEL}" | tee -a "${LOG_FILE}"
echo "[SWEEP] Manifest: ${MANIFEST}" | tee -a "${LOG_FILE}"
echo "[SWEEP] Scales: ${SCALES[*]}" | tee -a "${LOG_FILE}"

for scale in "${SCALES[@]}"; do
  OUT_DIR="${OUTPUT_ROOT}/polar_vote_v28_obb_scale_${scale}"
  echo "[SWEEP] Running scale=${scale}" | tee -a "${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/compare_polar_vote_v28_crop_sources.py \
    --manifest "${MANIFEST}" \
    --obb-model "${OBB_MODEL}" \
    --obb-model-kind tflite \
    --rectifier-model "${RECTIFIER_MODEL}" \
    --rectifier-model-kind tflite \
    --output-dir "${OUT_DIR}" \
    --obb-crop-scale "${scale}" \
    --obb-min-crop-size "${MIN_CROP_SIZE}" \
    2>&1 | tee -a "${LOG_FILE}"
done

echo "[SWEEP] Sweep complete." | tee -a "${LOG_FILE}"
