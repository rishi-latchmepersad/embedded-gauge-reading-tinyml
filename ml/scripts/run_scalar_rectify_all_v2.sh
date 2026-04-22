#!/usr/bin/env bash
set -euo pipefail

# Iteration 2: True 2-stage scalar training.
# Apply rectifier-predicted crop boxes to ALL training examples (RECTIFY_ALL=1),
# so the scalar learns one unified "rectified-crop" framing distribution that
# matches what the board sees at inference time.
#
# Warm-starts from clean_plus_new5 (prod) with very low LR to preserve the
# accuracy already learned and only adapt to the rectifier framing.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/scalar_rectify_all_v2.log"
BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new5/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_rectify_all_v2.model.keras"
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

echo "[WRAPPER] Starting rectify-all (true 2-stage) scalar fine-tune v2."
echo "[WRAPPER] Base model:    ${BASE_MODEL_LOCAL}"
echo "[WRAPPER] Rectifier:     ${RECTIFIER_MODEL_LOCAL}"
echo "[WRAPPER] Boxes CSV:     ${BOXES_CSV}"
echo "[WRAPPER] Log file:      ${LOG_FILE}"

if [[ ! -f "${BOXES_CSV}" ]]; then
  echo "[WRAPPER] Precomputing rectifier boxes..."
  bash scripts/run_precompute_rectifier_boxes.sh
  echo "[WRAPPER] Precomputation done."
else
  echo "[WRAPPER] Using existing boxes CSV: ${BOXES_CSV}"
fi

# RECTIFY_ALL=1 makes _apply_precomputed_crop_boxes apply boxes to phone photos too.
RECTIFY_ALL=1 PYTHONDONTWRITEBYTECODE=1 "${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2 \
  --device cpu \
  --batch-size 4 \
  --epochs 30 \
  --learning-rate 1e-6 \
  --init-model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 4 \
  --val-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --edge-focus-strength 1.0 \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --run-name scalar_rectify_all_v2 \
  2>&1 | tee "${LOG_FILE}"
