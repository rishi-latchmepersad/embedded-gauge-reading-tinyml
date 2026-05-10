#!/usr/bin/env bash
set -euo pipefail

# Evaluate the blur-aware OBB-geometry model directly on the hard-case mix.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_bluraware_obb_geometry_self_cascade_v34.log"
OUTPUT_DIR="${REPO_ROOT}/artifacts/cascade_eval/mobilenetv2_bluraware_obb_geometry_self_v34"

LOCALIZER_MODEL="${LOCALIZER_MODEL:-${REPO_ROOT}/artifacts/training/mobilenetv2_bluraware_obb_geometry_v34/model.keras}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid_with_new5.csv}"

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

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting blur-aware self-cascade evaluation." | tee "${LOG_FILE}"
echo "[WRAPPER] Localizer: ${LOCALIZER_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Manifest:  ${MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:  ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/eval_keypoint_reader_cascade_on_manifest.py \
  --localizer-model "${LOCALIZER_MODEL}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
