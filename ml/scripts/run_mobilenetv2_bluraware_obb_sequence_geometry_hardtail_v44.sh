#!/usr/bin/env bash
set -euo pipefail

# Hard-tail fine-tune for the blur-aware OBB sequence-geometry reader.
# We keep the fixed OBB front-end and the sequence-style geometry head, then
# bias the schedule harder toward the cold / preview-heavy manifest rows so we
# can see whether the strong v43 pretrainer can be pushed further.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
TRAIN_LOG_FILE="${LOG_DIR}/mobilenetv2_bluraware_obb_sequence_geometry_hardtail_v44.log"
EVAL_LOG_FILE="${LOG_DIR}/mobilenetv2_bluraware_obb_sequence_geometry_hardtail_v44_hardcase.log"
RUN_NAME="${RUN_NAME:-mobilenetv2_bluraware_obb_sequence_geometry_hardtail_v44}"
INIT_MODEL_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_bluraware_obb_sequence_geometry_v43.model.keras"
HARD_MANIFEST="${HARD_MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid_with_new5.csv}"
OBB_MODEL="${OBB_MODEL:-${REPO_ROOT}/artifacts/deployment/prod_model_v0.3_obb_int8/model_int8.tflite}"
RECTIFIER_MODEL="${RECTIFIER_MODEL:-${REPO_ROOT}/artifacts/deployment/mobilenetv2_rectifier_hardcase_finetune_v3_int8/model_int8.tflite}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${INIT_MODEL_LOCAL}")"

if [[ ! -f "${REPO_ROOT}/artifacts/training/mobilenetv2_bluraware_obb_sequence_geometry_v43/model.keras" ]]; then
  echo "[WRAPPER] Expected warm-start checkpoint is missing." >&2
  exit 1
fi

cp -f "${REPO_ROOT}/artifacts/training/mobilenetv2_bluraware_obb_sequence_geometry_v43/model.keras" "${INIT_MODEL_LOCAL}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Starting hard-tail fine-tune for blur-aware OBB sequence geometry." | tee "${TRAIN_LOG_FILE}"
echo "[WRAPPER] Init model: ${INIT_MODEL_LOCAL}" | tee -a "${TRAIN_LOG_FILE}"
echo "[WRAPPER] Train log:   ${TRAIN_LOG_FILE}" | tee -a "${TRAIN_LOG_FILE}"

RECTIFY_ALL=1 "${POETRY_BIN}" run python -u scripts/run_training.py \
  --run-name "${RUN_NAME}" \
  --model-family mobilenet_v2_obb_sequence_geometry \
  --gauge-id littlegood_home_temp_gauge_c \
  --device gpu \
  --batch-size 4 \
  --epochs 12 \
  --learning-rate 5e-6 \
  --seed 21 \
  --mobilenet-backbone-trainable \
  --mobilenet-warmup-epochs 2 \
  --mobilenet-unfreeze-last-n 6 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 96 \
  --mobilenet-head-dropout 0.15 \
  --geometry-value-loss-weight 2.5 \
  --keypoint-heatmap-size 56 \
  --keypoint-heatmap-loss-weight 0.75 \
  --keypoint-coord-loss-weight 1.25 \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 12 \
  --precomputed-crop-boxes data/rectified_crop_boxes_v5_all.csv \
  --edge-focus-strength 2.0 \
  --init-model "${INIT_MODEL_LOCAL}" \
  2>&1 | tee -a "${TRAIN_LOG_FILE}"

TRAIN_MODEL="${REPO_ROOT}/artifacts/training/${RUN_NAME}/model.keras"
EVAL_OUTPUT_DIR="${REPO_ROOT}/artifacts/cascade_eval/${RUN_NAME}_hardcase"

echo "[WRAPPER] Starting hard-case board replay." | tee "${EVAL_LOG_FILE}"
echo "[WRAPPER] Scalar model: ${TRAIN_MODEL}" | tee -a "${EVAL_LOG_FILE}"
echo "[WRAPPER] Eval log:     ${EVAL_LOG_FILE}" | tee -a "${EVAL_LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/eval_board_pipeline_on_captures.py \
  --manifest "${HARD_MANIFEST}" \
  --obb-model "${OBB_MODEL}" \
  --rectifier-model "${RECTIFIER_MODEL}" \
  --scalar-model "${TRAIN_MODEL}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  2>&1 | tee -a "${EVAL_LOG_FILE}"
