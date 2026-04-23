#!/usr/bin/env bash
set -euo pipefail

# Fit firmware calibration for the OBB + scalar cascade and tee the results.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/calibrate_obb_scalar_firmware.log"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "[WRAPPER] Fitting OBB + scalar firmware calibration."
echo "[WRAPPER] Fit manifest: data/mid_band_focus_18_42.csv"
echo "[WRAPPER] Test manifests:"
echo "[WRAPPER]   - data/hard_cases_plus_board30_valid_with_new6.csv"
echo "[WRAPPER]   - data/board_weak_focus.csv"
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/calibrate_obb_scalar_firmware.py \
  --obb-crop-scale 1.20 \
  --fit-manifest data/mid_band_focus_18_42.csv \
  --test-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
  --test-manifest data/board_weak_focus.csv \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
