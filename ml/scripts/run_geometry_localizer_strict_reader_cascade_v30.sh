#!/usr/bin/env bash
set -euo pipefail

# Evaluate the geometry localizer + strict scalar reader two-stage pipeline.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/geometry_localizer_strict_reader_cascade_v30.log"
OUTPUT_DIR="${REPO_ROOT}/artifacts/cascade_eval/geometry_localizer_strict_reader_v30"

LOCALIZER_MODEL="${LOCALIZER_MODEL:-${REPO_ROOT}/artifacts/training/mobilenetv2_geometry_localizer_only_v30/model.keras}"
READER_MODEL="${READER_MODEL:-${REPO_ROOT}/artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid_with_new5.csv}"

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

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting geometry-localizer + strict-reader cascade evaluation."
echo "[WRAPPER] Localizer: ${LOCALIZER_MODEL}"
echo "[WRAPPER] Reader: ${READER_MODEL}"
echo "[WRAPPER] Manifest: ${MANIFEST}"
echo "[WRAPPER] Output dir: ${OUTPUT_DIR}"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/eval_keypoint_reader_cascade_on_manifest.py \
  --localizer-model "${LOCALIZER_MODEL}" \
  --reader-model "${READER_MODEL}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"
