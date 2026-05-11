#!/usr/bin/env bash
set -euo pipefail

# Replay the current blur-aware OBB sequence reader with the widened crop
# acceptance window. This lets us measure whether the reader improves once the
# OBB crop is no longer forced into rectifier fallback on the hard tail.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
HARD_MANIFEST="${HARD_MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid_with_new5.csv}"
OBB_MODEL="${OBB_MODEL:-${REPO_ROOT}/artifacts/deployment/prod_model_v0.3_obb_int8/model_int8.tflite}"
RECTIFIER_MODEL="${RECTIFIER_MODEL:-${REPO_ROOT}/artifacts/deployment/mobilenetv2_rectifier_hardcase_finetune_v3_int8/model_int8.tflite}"
SCALAR_MODEL="${SCALAR_MODEL:-${REPO_ROOT}/artifacts/training/mobilenetv2_bluraware_obb_sequence_geometry_v43/model.keras}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/cascade_eval/mobilenetv2_bluraware_obb_sequence_geometry_v43_cropwindow_v45_hardcase}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/artifacts/training_logs/mobilenetv2_bluraware_obb_sequence_geometry_v43_cropwindow_v45_hardcase.log}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[REPLAY] Poetry was not found in WSL." >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

echo "[REPLAY] Starting widened-crop hard-case replay." | tee "${LOG_FILE}"
echo "[REPLAY] Scalar model: ${SCALAR_MODEL}" | tee -a "${LOG_FILE}"
echo "[REPLAY] Output dir:   ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/eval_board_pipeline_on_captures.py \
  --manifest "${HARD_MANIFEST}" \
  --obb-model "${OBB_MODEL}" \
  --rectifier-model "${RECTIFIER_MODEL}" \
  --scalar-model "${SCALAR_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
