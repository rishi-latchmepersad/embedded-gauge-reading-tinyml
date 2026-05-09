#!/usr/bin/env bash
set -euo pipefail

# Pretrain a linear-head MobileNetV2 on synthetic gauge renders.
#
# This is meant to teach the backbone the angle/value relationship before we
# fine-tune on the strict rectified real images.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_synthetic_pretrain_v11.log"
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

SYNTH_DIR="../tmp/synth_gauge_pretrain_v1"
SYNTH_MANIFEST="${SYNTH_DIR}/manifest.csv"
PRETRAIN_OUTPUT="artifacts/training/mobilenetv2_synthetic_pretrain_v11"

echo "[WRAPPER] Generating synthetic gauge dataset..." | tee "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/generate_synthetic_gauge_dataset.py \
  --output-dir "${SYNTH_DIR}" \
  --manifest-path "${SYNTH_MANIFEST}" \
  --num-samples 1600 \
  --image-size 224 \
  --seed 21 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[WRAPPER] Pretraining on synthetic gauges..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "${PRETRAIN_OUTPUT}" \
  --manifest-path "${SYNTH_MANIFEST}" \
  --linear-output \
  --batch-size 8 \
  --epochs 12 \
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
