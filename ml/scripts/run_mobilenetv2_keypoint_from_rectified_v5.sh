#!/usr/bin/env bash
set -euo pipefail

# Train a lighter keypoint-aware MobileNetV2 from the rectified v5 data mix.
#
# This keeps the research direction intact:
# learn explicit needle structure first, then let the scalar gauge head use
# those spatial features. It is intentionally lighter than the geometry model
# that was hanging during build on this WSL stack.

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_keypoint_from_rectified_v5.log"

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

# XLA can make the first epoch look frozen on this WSL/GPU stack, so disable
# automatic JIT for the first-pass geometry experiment.
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

echo "[WRAPPER] Starting keypoint-from-rectified-v5 fine-tune."
echo "[WRAPPER] Log file: ${LOG_FILE}"

"${POETRY_BIN}" run python -u scripts/run_training.py \
  --model-family mobilenet_v2_keypoint \
  --device gpu \
  --no-gpu-memory-growth \
  --batch-size 4 \
  --epochs 8 \
  --learning-rate 2e-5 \
  --mobilenet-warmup-epochs 1 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --mobilenet-backbone-trainable \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 64 \
  --mobilenet-head-dropout 0.15 \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
  --hard-case-repeat 6 \
  --edge-focus-strength 1.0 \
  --keypoint-heatmap-size 28 \
  --keypoint-heatmap-loss-weight 0.1 \
  --geometry-value-loss-weight 3.0 \
  --run-name mobilenetv2_keypoint_from_rectified_v5 \
  2>&1 | tee "${LOG_FILE}"
