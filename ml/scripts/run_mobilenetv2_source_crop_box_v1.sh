#!/usr/bin/env bash
set -euo pipefail

# Train a direct source-space crop-box localizer for the exact V28 replay path.
#
# The target is the rectified crop box in source pixels, which makes the crop
# prediction closer to the offline oracle than the normalized center/size
# rectifier. This wrapper keeps the experiment repeatable and easy to rerun.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_source_crop_box_v1.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/mobilenetv2_rectifier_hardcase_finetune_v3/model.keras"
BASE_MODEL_LOCAL="${REPO_ROOT}/../tmp/source_crop_box_v1.model.keras"
BOXES_CSV="${REPO_ROOT}/data/rectified_crop_boxes_v5_all.csv"
HARD_CASE_MANIFEST="${HARD_CASE_MANIFEST:-data/hard_cases_plus_board30_valid_with_new6.csv}"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${BASE_MODEL_LOCAL}")"
cp -f "${BASE_MODEL_SRC}" "${BASE_MODEL_LOCAL}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting source crop-box fine-tune v1."
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Boxes CSV:  ${BOXES_CSV}"
echo "[WRAPPER] Hard set:   ${HARD_CASE_MANIFEST}"
echo "[WRAPPER] Log file:   ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_source_crop_box \
  --device gpu \
  --no-gpu-memory-growth \
  --batch-size 4 \
  --epochs 12 \
  --learning-rate 3e-6 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --val-manifest "${HARD_CASE_MANIFEST}" \
  --hard-case-manifest "${HARD_CASE_MANIFEST}" \
  --hard-case-repeat 4 \
  --edge-focus-strength 1.0 \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --run-name mobilenetv2_source_crop_box_v1 \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
