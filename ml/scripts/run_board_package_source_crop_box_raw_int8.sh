#!/usr/bin/env bash
set -euo pipefail

# Package the source-crop-box localizer for the STM32N6 relocatable flow.
#
# This wrapper keeps the repeatable package settings in one place and makes the
# board-facing source-crop-box build use an int8 input tensor, which avoids the
# fragile float-input QuantizeLinear startup path on the board.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_source_crop_box_raw_int8_board_package.log"
MODEL_IN="${MODEL_IN:-artifacts/deployment/mobilenetv2_source_crop_box_v1_int8/model_int8.tflite}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runtime/mobilenetv2_source_crop_box_v1_stripped_int8_reloc}"
WORKSPACE_DIR="${WORKSPACE_DIR:-../st_ai_output/packages/mobilenetv2_source_crop_box_v1_stripped_int8/st_ai_ws}"
STAI_OUTPUT_DIR="${STAI_OUTPUT_DIR:-../st_ai_output/packages/mobilenetv2_source_crop_box_v1_stripped_int8/st_ai_output}"
WORK_ROOT="${WORK_ROOT:-${HOME}/mobilenetv2_source_crop_box_raw_int8_board_package}"
BASE_MODEL_LOCAL="${WORK_ROOT}/model_int8.tflite"
CANONICAL_RAW_PATH="${CANONICAL_RAW_PATH:-${STAI_OUTPUT_DIR}/network_atonbuf.xSPI2.raw}"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"
rm -rf "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}"

echo "[WRAPPER] Staging source-crop-box int8 model into ${WORK_ROOT}."
cp "${REPO_ROOT}/${MODEL_IN}" "${BASE_MODEL_LOCAL}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting board package for mobilenetv2_source_crop_box_v1_stripped_int8."
echo "[WRAPPER] Model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Output dir: ${OUTPUT_DIR}"
echo "[WRAPPER] Workspace: ${WORKSPACE_DIR}"
echo "[WRAPPER] ST AI output: ${STAI_OUTPUT_DIR}"
echo "[WRAPPER] Canonical raw: ${CANONICAL_RAW_PATH}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

# The source-crop-box head is only useful on the board if the generated
# package exposes an int8 input tensor, so we force the relocatable pack to
# keep the input in int8 and let the model handle its own final dequantize.
CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/package_scalar_model_for_n6.py \
  --model "${BASE_MODEL_LOCAL}" \
  --output-dir "${OUTPUT_DIR}" \
  --workspace-dir "${WORKSPACE_DIR}" \
  --stai-output-dir "${STAI_OUTPUT_DIR}" \
  --name mobilenetv2_source_crop_box_v1_stripped_int8 \
  --canonical-raw-path "${CANONICAL_RAW_PATH}" \
  --compression high \
  --optimization balanced \
  --input-data-type int8 \
  --output-data-type float32 \
  2>&1 | tee "${LOG_FILE}"

echo "[WRAPPER] Source-crop-box package refreshed."
