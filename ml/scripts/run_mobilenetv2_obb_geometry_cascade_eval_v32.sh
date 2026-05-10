#!/usr/bin/env bash
set -euo pipefail

# Evaluate the new OBB-geometry model as a cascade front-end on the hard-case mix.
#
# This keeps the geometry-aware `mobilenetv2_obb_geometry_v32` model in the
# localization slot and uses the strict rectified scalar reader as the second
# stage so we can see whether the new geometry supervision helps the downstream
# reader more than the older OBB front-end.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_obb_geometry_cascade_eval_v32.log"
OUTPUT_DIR="${REPO_ROOT}/artifacts/cascade_eval/mobilenetv2_obb_geometry_v32_to_strict_v5_hardcase_v32"

LOCALIZER_MODEL="${LOCALIZER_MODEL:-${REPO_ROOT}/artifacts/training/mobilenetv2_obb_geometry_v32/model.keras}"
READER_MODEL="${READER_MODEL:-${REPO_ROOT}/artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras}"
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

echo "[WRAPPER] Starting OBB-geometry cascade evaluation." | tee "${LOG_FILE}"
echo "[WRAPPER] Localizer: ${LOCALIZER_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Reader:    ${READER_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Manifest:  ${MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:  ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/eval_keypoint_reader_cascade_on_manifest.py \
  --localizer-model "${LOCALIZER_MODEL}" \
  --reader-model "${READER_MODEL}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
