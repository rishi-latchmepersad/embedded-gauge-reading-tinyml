#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the scalar model using rectifier-predicted crop boxes instead of
# the fixed training crop. This makes the scalar model invariant to camera
# placement: at inference time the board runs the same rectifier -> crop ->
# scalar pipeline that was used during training.
#
# Warm-starts from the current prod best (clean_plus_new5) so we preserve all
# existing accuracy and only adapt to the rectifier framing.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_rectified_crop_finetune_v1.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_rectified_crop_finetune_v1.model.keras"
RECTIFIER_MODEL_SRC="${REPO_ROOT}/artifacts/training/mobilenetv2_rectifier_zoom_aug_v4/model.keras"
RECTIFIER_MODEL_LOCAL="${HOME}/ml_eval_cache/rectifier_zoom_aug_v4.model.keras"

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
cp -f "${RECTIFIER_MODEL_SRC}" "${RECTIFIER_MODEL_LOCAL}"

cd "${REPO_ROOT}"

BOXES_CSV="${REPO_ROOT}/data/rectified_crop_boxes_v4_all.csv"

echo "[WRAPPER] Starting rectified-crop scalar fine-tune v1."
echo "[WRAPPER] Base model:    ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Rectifier:     ${RECTIFIER_MODEL_LOCAL}"
echo "[WRAPPER] Boxes CSV:     ${BOXES_CSV}"
echo "[WRAPPER] Log file:      ${LOG_FILE}"

# Step 1: precompute rectifier crop boxes (CPU-only, separate process)
if [[ ! -f "${BOXES_CSV}" ]]; then
  echo "[WRAPPER] Precomputing rectifier boxes..."
  bash scripts/run_precompute_rectifier_boxes.sh
  echo "[WRAPPER] Precomputation done."
else
  echo "[WRAPPER] Using existing boxes CSV: ${BOXES_CSV}"
fi

# Step 2: train scalar using precomputed boxes (GPU, no rectifier in memory)
"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --device gpu \
  --batch-size 4 \
  --epochs 40 \
  --learning-rate 5e-6 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 4 \
  --val-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --edge-focus-strength 1.0 \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --run-name scalar_rectified_crop_finetune_v1 \
  2>&1 | tee "${LOG_FILE}"
