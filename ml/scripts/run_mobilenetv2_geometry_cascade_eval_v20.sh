#!/usr/bin/env bash
set -euo pipefail

# Evaluate the geometry-model front-end as a deployable cascade on the hard-case mix.
#
# This keeps the raw-CVAT geometry model in the localization slot and uses the
# current strict rectified scalar model as the reader so we can measure whether
# the geometry stage is a useful front-end for deployment.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_geometry_cascade_eval_v20.log"
OUTPUT_DIR="${REPO_ROOT}/artifacts/cascade_eval/mobilenetv2_geometry_v18_to_strict_v5_hardcase_v20"

LOCALIZER_MODEL="${LOCALIZER_MODEL:-${REPO_ROOT}/artifacts/training/mobilenetv2_geometry_raw_cvat_v18/model.keras}"
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

echo "[WRAPPER] Starting geometry-cascade evaluation." | tee "${LOG_FILE}"
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
