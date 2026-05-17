#!/usr/bin/env bash
set -euo pipefail

# Hard-tail fine-tune for the blur-aware OBB + geometry reader.
#
# This no-init version skips Keras model deserialization altogether so we can
# get a clean training run even when warm-start checkpoints are slow to load.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
TRAIN_LOG_FILE="${LOG_DIR}/mobilenetv2_bluraware_obb_geometry_hardtail_v38.log"
RUN_NAME="${RUN_NAME:-mobilenetv2_bluraware_obb_geometry_hardtail_v38}"
HARD_MANIFEST="${HARD_MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid_with_new5.csv}"
VAL_MANIFEST="${VAL_MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid_with_new6.csv}"
TEST_MANIFEST="${TEST_MANIFEST:-${REPO_ROOT}/data/board_rectified_probe_20260422.csv}"
OBB_MODEL="${OBB_MODEL:-${REPO_ROOT}/artifacts/training/mobilenetv2_obb_geometry_v32/model.keras}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting blur-aware OBB-geometry hard-tail fine-tune (no init)." | tee "${TRAIN_LOG_FILE}"
echo "[WRAPPER] Log file:   ${TRAIN_LOG_FILE}" | tee -a "${TRAIN_LOG_FILE}"

RECTIFY_ALL=1 "${POETRY_BIN}" run python -u scripts/run_training.py \
  --run-name "${RUN_NAME}" \
  --model-family mobilenet_v2_bluraware_obb_geometry \
  --gauge-id littlegood_home_temp_gauge_c \
  --device gpu \
  --no-gpu-memory-growth \
  --batch-size 4 \
  --epochs 24 \
  --learning-rate 1e-5 \
  --seed 21 \
  --mobilenet-backbone-trainable \
  --mobilenet-warmup-epochs 4 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --geometry-value-loss-weight 1.5 \
  --keypoint-heatmap-loss-weight 0.75 \
  --keypoint-coord-loss-weight 1.25 \
  --hard-case-manifest "${HARD_MANIFEST}" \
  --hard-case-repeat 12 \
  --val-manifest "${VAL_MANIFEST}" \
  --test-manifest "${TEST_MANIFEST}" \
  --precomputed-crop-boxes data/rectified_crop_boxes_v5_all.csv \
  --edge-focus-strength 2.0 \
  2>&1 | tee -a "${TRAIN_LOG_FILE}"

TRAIN_MODEL="${REPO_ROOT}/artifacts/training/${RUN_NAME}/model.keras"
EVAL_OUT_DIR="${REPO_ROOT}/artifacts/cascade_eval/${RUN_NAME}"

for MANIFEST in \
  "${REPO_ROOT}/data/hard_cases.csv" \
  "${REPO_ROOT}/data/hard_cases_plus_board30.csv"; do
  STEM="$(basename "${MANIFEST}" .csv)"
  OUTPUT_DIR="${EVAL_OUT_DIR}/${STEM}"
  EVAL_LOG_FILE="${LOG_DIR}/${RUN_NAME}_${STEM}.log"

  echo "[WRAPPER] Evaluating cascade on ${STEM}." | tee "${EVAL_LOG_FILE}"
  "${POETRY_BIN}" run python -u scripts/eval_obb_scalar_on_manifest.py \
    --obb-model "${OBB_MODEL}" \
    --obb-model-kind keras \
    --scalar-model "${TRAIN_MODEL}" \
    --scalar-model-kind keras \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee -a "${EVAL_LOG_FILE}"
done
