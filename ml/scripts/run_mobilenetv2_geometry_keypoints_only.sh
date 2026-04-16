#!/usr/bin/env bash
set -euo pipefail

# Train a geometry-first MobileNetV2 that learns keypoints before any value target.
#
# The goal is to make the detector learn gauge structure first and let the
# deterministic sweep conversion handle the final temperature mapping.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_geometry_keypoints_only.log"

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

echo "[WRAPPER] Starting geometry-first keypoint-only MobileNetV2 training."
echo "[WRAPPER] Log file: ${LOG_FILE}"

TRAIN_ARGS=(
  --model-family mobilenet_v2_geometry
  --device gpu
  --epochs 12
  --learning-rate 1e-6
  --mobilenet-warmup-epochs 2
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv
  --hard-case-repeat 4
  --edge-focus-strength 1.0
  --keypoint-heatmap-size 56
  --keypoint-heatmap-loss-weight 2.0
  --keypoint-coord-loss-weight 4.0
  --geometry-value-loss-weight 0.0
  --run-name mobilenetv2_geometry_keypoints_only
)

"${POETRY_BIN}" run python -u scripts/run_training.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
