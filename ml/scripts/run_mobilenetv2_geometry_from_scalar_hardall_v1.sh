#!/usr/bin/env bash
set -euo pipefail

# Geometry-first hard-case run:
# - warm-start from the strongest scalar checkpoint
# - add keypoint supervision from CVAT center/tip labels
# - include the full hard-case manifest without tail-specific bias

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_geometry_from_scalar_hardall_v1.log"
INIT_MODEL="${REPO_ROOT}/artifacts/training/scalar_full_finetune_from_best_board30_clean_plus_new6/model.keras"

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

echo "[WRAPPER] Starting geometry-from-scalar hard-case run v1." | tee "${LOG_FILE}"
echo "[WRAPPER] Init model: ${INIT_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_geometry \
  --device gpu \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --mobilenet-warmup-epochs 1 \
  --mobilenet-unfreeze-last-n 16 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-backbone-trainable \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 96 \
  --mobilenet-head-dropout 0.15 \
  --init-model "${INIT_MODEL}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
  --hard-case-repeat 4 \
  --edge-focus-strength 0.0 \
  --keypoint-heatmap-size 56 \
  --keypoint-heatmap-loss-weight 1.0 \
  --keypoint-coord-loss-weight 1.0 \
  --geometry-value-loss-weight 3.0 \
  --run-name mobilenetv2_geometry_from_scalar_hardall_v1 \
  2>&1 | tee -a "${LOG_FILE}"
