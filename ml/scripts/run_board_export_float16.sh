#!/usr/bin/env bash
set -euo pipefail

# Export the calibrated scalar CNN to a float16 TFLite artifact.
# The source model is staged into WSL-local storage first so the Keras load and
# conversion steps do not stall on the Windows-mounted drive.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
MODEL_IN="${MODEL_IN:-artifacts/training/scalar_full_finetune_from_best_board30_piecewise_calibrated/model.keras}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/deployment/scalar_full_finetune_from_best_board30_piecewise_calibrated_float16}"
WORK_ROOT="${WORK_ROOT:-${HOME}/float16_export_scalar_full_board30}"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[EXPORT] Poetry was not found in WSL." >&2
  exit 1
fi

rm -rf "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}"

echo "[EXPORT] Staging model into ${WORK_ROOT}."
cp "${REPO_ROOT}/${MODEL_IN}" "${WORK_ROOT}/model.keras"

cd "${REPO_ROOT}"

echo "[EXPORT] Loading from staged model: ${WORK_ROOT}/model.keras"
echo "[EXPORT] Writing output to: ${REPO_ROOT}/${OUTPUT_DIR}"

"${POETRY_BIN}" run python -u scripts/export_board_artifacts_float16.py \
  --model "${WORK_ROOT}/model.keras" \
  --output-dir "${REPO_ROOT}/${OUTPUT_DIR}" \
  2>&1 | tee "${REPO_ROOT}/artifacts/training_logs/board_export_float16.log"

