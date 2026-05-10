#!/usr/bin/env bash
set -euo pipefail

# Train the V5-style rectified scalar model on the broad real-data pool.
#
# The synthetic renders are used as a separate pretraining stage because they
# are useful for learning the gauge/value relationship, but they are visually
# more idealized than the real captures. The final fine-tune stays anchored on
# the full real manifest and its precomputed rectifier crop boxes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_all_data_v21.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

cd "${ROOT_DIR}"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

SYNTH_DIR="../tmp/synth_gauge_all_v21"
SYNTH_MANIFEST="${SYNTH_DIR}/manifest.csv"
SYNTH_PRETRAIN_OUTPUT="artifacts/training/mobilenetv2_synthetic_pretrain_v21"

REAL_MANIFEST="data/full_scalar_manifest_v1.csv"
REAL_BOXES_CSV="data/rectified_crop_boxes_full_scalar_v1.csv"
REAL_OUTPUT_DIR="artifacts/training/mobilenetv2_rectified_scalar_all_data_v21"
HARD_CASE_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"

echo "[WRAPPER] Starting V5 synthetic pretrain + all-data rectified fine-tune v21." | tee "${LOG_FILE}"
echo "[WRAPPER] Synthetic manifest: ${SYNTH_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Real manifest:      ${REAL_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Real boxes CSV:     ${REAL_BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Real output dir:    ${REAL_OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:           ${LOG_FILE}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Generating synthetic gauge pretraining data..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/generate_synthetic_gauge_dataset.py \
  --output-dir "${SYNTH_DIR}" \
  --manifest-path "${SYNTH_MANIFEST}" \
  --num-samples 1600 \
  --image-size 224 \
  --seed 21 \
  --profile standard \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Pretraining the V5 backbone on synthetic gauges..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${SYNTH_PRETRAIN_OUTPUT}" \
  --manifest-path "${SYNTH_MANIFEST}" \
  --linear-output \
  --batch-size 8 \
  --epochs 10 \
  --warmup-epochs 4 \
  --learning-rate 1e-4 \
  --fine-tune-lr 2e-5 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --dropout 0.15 \
  --seed 21 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Fine-tuning on the full real-data manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${REAL_OUTPUT_DIR}" \
  --manifest-path "${REAL_MANIFEST}" \
  --precomputed-crop-boxes "${REAL_BOXES_CSV}" \
  --init-model "${SYNTH_PRETRAIN_OUTPUT}/model.keras" \
  --linear-output \
  --augment-mode hard_preview \
  --batch-size 8 \
  --epochs 16 \
  --warmup-epochs 6 \
  --learning-rate 5e-5 \
  --fine-tune-lr 2e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --dropout 0.15 \
  --seed 21 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Evaluating the final model on the hard-case manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/eval_rectified_scalar_model_on_manifest.py \
  --model "${REAL_OUTPUT_DIR}/model.keras" \
  --manifest "${HARD_CASE_MANIFEST}" \
  --crop-boxes "${REAL_BOXES_CSV}" \
  2>&1 | tee -a "${LOG_FILE}"
