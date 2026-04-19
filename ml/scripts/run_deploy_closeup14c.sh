#!/usr/bin/env bash
set -euo pipefail

# Quantize and package scalar_full_finetune_closeup14c for the N6 board.
# This model (MAE ~0.8C on 33C board captures) was trained but never deployed.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
MODEL_IN="${REPO_ROOT}/artifacts/training/scalar_full_finetune_closeup14c/model.keras"
DEPLOY_DIR="${REPO_ROOT}/artifacts/deployment/scalar_full_finetune_closeup14c_int8"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/deploy_closeup14c.log"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi
if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry not found." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${DEPLOY_DIR}"

cd "${REPO_ROOT}"

echo "[WRAPPER] Step 1: Quantize Keras model to INT8 TFLite."
CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python scripts/export_board_artifacts.py \
  --model "${MODEL_IN}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5_closeup14c.csv \
  --output-dir "${DEPLOY_DIR}" \
  --deployment-kind scalar \
  --representative-count 32 \
  2>&1 | tee "${LOG_FILE}"

echo "[WRAPPER] Step 2: Package TFLite into xSPI2 blob."
MODEL_IN="${DEPLOY_DIR}/model_int8.tflite" \
OUTPUT_DIR="${REPO_ROOT}/artifacts/runtime/scalar_full_finetune_closeup14c_int8_reloc" \
WORKSPACE_DIR="${REPO_ROOT}/../st_ai_output/packages/scalar_full_finetune_closeup14c/st_ai_ws" \
STAI_OUTPUT_DIR="${REPO_ROOT}/../st_ai_output/packages/scalar_full_finetune_closeup14c/st_ai_output" \
CUDA_VISIBLE_DEVICES="-1" "${POETRY_BIN}" run python -u scripts/package_scalar_model_for_n6.py \
  --model "${DEPLOY_DIR}/model_int8.tflite" \
  --output-dir "${REPO_ROOT}/artifacts/runtime/scalar_full_finetune_closeup14c_int8_reloc" \
  --workspace-dir "${REPO_ROOT}/../st_ai_output/packages/scalar_full_finetune_closeup14c/st_ai_ws" \
  --stai-output-dir "${REPO_ROOT}/../st_ai_output/packages/scalar_full_finetune_closeup14c/st_ai_output" \
  --name scalar_full_finetune_closeup14c \
  --canonical-raw-path "${REPO_ROOT}/../st_ai_output/atonbuf.xSPI2.raw" \
  --compression high \
  --optimization balanced \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Done. Flash with: FLASH_MODEL=1 flash_boot.bat"
