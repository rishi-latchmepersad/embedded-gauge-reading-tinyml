#!/usr/bin/env bash
set -euo pipefail

# Export the center detector to board-ready TFLite int8 artifacts.
REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
MODEL_IN="${MODEL_IN:-artifacts/training/center_detector_v1_20260530_201137/best_model.keras}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/deployment/center_detector_v1_int8}"
WORK_ROOT="${WORK_ROOT:-${HOME}/center_detector_board_export}"
BASE_MODEL_LOCAL="${WORK_ROOT}/model.keras"

mkdir -p "${LOG_DIR}" "${WORK_ROOT}" "${REPO_ROOT}"
rm -rf "${WORK_ROOT}"/*
cp "${REPO_ROOT}/${MODEL_IN}" "${BASE_MODEL_LOCAL}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="-1" poetry run python -u scripts/export_board_artifacts.py \
  --model "${BASE_MODEL_LOCAL}" \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --output-dir "${OUTPUT_DIR}" \
  --deployment-kind center_detector \
  --representative-count 64 2>&1 | tee "${LOG_DIR}/center_detector_board_export.log"
