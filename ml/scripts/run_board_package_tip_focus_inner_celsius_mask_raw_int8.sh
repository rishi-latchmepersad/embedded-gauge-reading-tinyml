#!/usr/bin/env bash
set -euo pipefail

# Package the inner-Celsius tip-focus int8 model into the STM32N6 relocatable
# flow, then refresh the firmware-local NPU package payload in place.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
PROJECT_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml"
FIRMWARE_PACKAGE_ROOT="${PROJECT_ROOT}/firmware/stm32/n657/st_ai_output/packages/tip_focus_v4_112_int8_n6_npu"
FIRMWARE_PACKAGE_STAI_OUTPUT="${FIRMWARE_PACKAGE_ROOT}/st_ai_output"
TMP_ROOT="${PROJECT_ROOT}/tmp/tip_focus_v4_112_inner_celsius_mask_npu_export"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/tip_focus_v4_112_inner_celsius_mask_board_package.log"
MODEL_IN="${MODEL_IN:-artifacts/deployment/geometry_heatmap_v4_112_inner_celsius_mask/model_v4_112_int8.tflite}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runtime/tip_focus_v4_112_inner_celsius_mask_reloc}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${TMP_ROOT}/st_ai_ws}"
STAI_OUTPUT_DIR="${STAI_OUTPUT_DIR:-${TMP_ROOT}/st_ai_output}"
WORK_ROOT="${WORK_ROOT:-${TMP_ROOT}/work}"
BASE_MODEL_LOCAL="${WORK_ROOT}/model_v4_112_int8.tflite"
CANONICAL_RAW_PATH="${CANONICAL_RAW_PATH:-${FIRMWARE_PACKAGE_STAI_OUTPUT}/network_atonbuf.xSPI2.raw}"

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
rm -rf "${TMP_ROOT}"
mkdir -p "${WORK_ROOT}"

echo "[WRAPPER] Staging tip-focus int8 model into ${WORK_ROOT}."
cp "${REPO_ROOT}/${MODEL_IN}" "${BASE_MODEL_LOCAL}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting board package for tip_focus_v4_112_int8."
echo "[WRAPPER] Model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Output dir: ${OUTPUT_DIR}"
echo "[WRAPPER] Workspace: ${WORKSPACE_DIR}"
echo "[WRAPPER] ST AI output: ${STAI_OUTPUT_DIR}"
echo "[WRAPPER] Firmware package ST AI output: ${FIRMWARE_PACKAGE_STAI_OUTPUT}"
echo "[WRAPPER] Canonical raw: ${CANONICAL_RAW_PATH}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/package_scalar_model_for_n6.py \
  --model "${BASE_MODEL_LOCAL}" \
  --output-dir "${OUTPUT_DIR}" \
  --workspace-dir "${WORKSPACE_DIR}" \
  --stai-output-dir "${STAI_OUTPUT_DIR}" \
  --name tip_focus_v4_112_int8 \
  --canonical-raw-path "${CANONICAL_RAW_PATH}" \
  --compression high \
  --optimization balanced \
  --input-data-type int8 \
  --output-data-type int8 \
  2>&1 | tee "${LOG_FILE}"

# Keep the firmware package layout stable: only refresh the generated NPU
# payload underneath the existing package root.
mkdir -p "${FIRMWARE_PACKAGE_STAI_OUTPUT}"
rsync -a --delete "${STAI_OUTPUT_DIR}/" "${FIRMWARE_PACKAGE_STAI_OUTPUT}/"

TIP_FOCUS_C="${FIRMWARE_PACKAGE_STAI_OUTPUT}/tip_focus_v4_112_int8.c"
TIP_FOCUS_H="${FIRMWARE_PACKAGE_STAI_OUTPUT}/tip_focus_v4_112_int8.h"
TIP_FOCUS_RAW="${FIRMWARE_PACKAGE_STAI_OUTPUT}/tip_focus_v4_112_int8_atonbuf.xSPI2.raw"
NETWORK_C="${FIRMWARE_PACKAGE_STAI_OUTPUT}/network.c"
NETWORK_H="${FIRMWARE_PACKAGE_STAI_OUTPUT}/network.h"
NETWORK_RAW="${FIRMWARE_PACKAGE_STAI_OUTPUT}/network_atonbuf.xSPI2.raw"

if [[ ! -f "${TIP_FOCUS_C}" || ! -f "${TIP_FOCUS_H}" || ! -f "${TIP_FOCUS_RAW}" ]]; then
  echo "[WRAPPER] ERROR: Tip-focus export did not produce the expected files." >&2
  exit 1
fi

cp -f "${TIP_FOCUS_C}" "${NETWORK_C}"
cp -f "${TIP_FOCUS_H}" "${NETWORK_H}"
cp -f "${TIP_FOCUS_RAW}" "${NETWORK_RAW}"

echo "[WRAPPER] Compatibility aliases refreshed:"
echo "[WRAPPER]   ${NETWORK_C}"
echo "[WRAPPER]   ${NETWORK_H}"
echo "[WRAPPER]   ${NETWORK_RAW}"

echo "[WRAPPER] Tip-focus package refreshed."
stat -c '[WRAPPER] Raw blob size: %n (%s bytes)' \
  "${NETWORK_RAW}"
