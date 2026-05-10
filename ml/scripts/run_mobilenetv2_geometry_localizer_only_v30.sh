#!/usr/bin/env bash
set -euo pipefail

# Train a geometry-only MobileNetV2 localizer for the two-stage cascade.
#
# This stage is responsible only for learning sharper keypoint heatmaps and
# more stable coordinates. The scalar reader is trained and evaluated as a
# separate model, which keeps the gauge-value objective from dominating the
# localization objective.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_geometry_localizer_only_v30.log"

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

echo "[WRAPPER] Starting geometry-only MobileNetV2 localizer fine-tune."
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_geometry \
  --device gpu \
  --no-gpu-memory-growth \
  --no-mobilenet-pretrained \
  --init-model artifacts/training/mobilenetv2_geometry_literature_v29/model.keras \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --mobilenet-warmup-epochs 1 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-backbone-trainable \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 8 \
  --edge-focus-strength 1.0 \
  --keypoint-heatmap-size 56 \
  --keypoint-heatmap-loss-weight 4.0 \
  --keypoint-coord-loss-weight 6.0 \
  --geometry-value-loss-weight 0.0 \
  --run-name mobilenetv2_geometry_localizer_only_v30 \
  2>&1 | tee "${LOG_FILE}"
