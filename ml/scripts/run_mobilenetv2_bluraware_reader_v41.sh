#!/usr/bin/env bash
set -euo pipefail

# Train a reader-only blur-aware MobileNetV2 model and benchmark it against the
# hard-case manifest using the fixed OBB front-end. This keeps the front-end
# constant while we test whether a lighter frequency-aware reader improves the
# tail behavior.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
TRAIN_LOG_FILE="${LOG_DIR}/mobilenetv2_bluraware_reader_v41.log"
EVAL_LOG_FILE="${LOG_DIR}/mobilenetv2_bluraware_reader_v41_hardcase.log"
RUN_NAME="${RUN_NAME:-mobilenetv2_bluraware_reader_v41}"

BASE_MODEL_SRC="${REPO_ROOT}/artifacts/training/scalar_rectified_crop_finetune_v2_20260422/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/scalar_rectified_crop_finetune_v2_20260422.model.keras"

OBB_MODEL="${OBB_MODEL:-${REPO_ROOT}/artifacts/deployment/prod_model_v0.3_obb_int8/model_int8.tflite}"
RECTIFIER_MODEL="${RECTIFIER_MODEL:-${REPO_ROOT}/artifacts/deployment/mobilenetv2_rectifier_hardcase_finetune_v3_int8/model_int8.tflite}"
HARD_MANIFEST="${HARD_MANIFEST:-${REPO_ROOT}/data/hard_cases_plus_board30_valid_with_new5.csv}"

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

echo "[WRAPPER] Starting blur-aware reader training." | tee "${TRAIN_LOG_FILE}"
echo "[WRAPPER] Init model: ${BASE_MODEL_LOCAL}" | tee -a "${TRAIN_LOG_FILE}"
echo "[WRAPPER] Train log:   ${TRAIN_LOG_FILE}" | tee -a "${TRAIN_LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --run-name "${RUN_NAME}" \
  --model-family mobilenet_v2_bluraware_reader \
  --gauge-id littlegood_home_temp_gauge_c \
  --device gpu \
  --batch-size 4 \
  --epochs 18 \
  --learning-rate 5e-6 \
  --seed 21 \
  --mobilenet-backbone-trainable \
  --mobilenet-warmup-epochs 4 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --linear-output \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 4 \
  --precomputed-crop-boxes data/rectified_crop_boxes_v5_all.csv \
  --edge-focus-strength 1.0 \
  --init-model "${BASE_MODEL_LOCAL}" \
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
