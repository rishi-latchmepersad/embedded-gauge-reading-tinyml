#!/usr/bin/env bash
set -euo pipefail

# Sequence-style geometry fine-tune for the OBB-mask reader.
# This version keeps the OBB localizer and pointer mask, but adds sweep-endpoint
# landmarks so the network has to model the full gauge arc, not just center+tip.
cd "$(dirname "$0")/.."

mkdir -p artifacts/training_logs

LOG_FILE="artifacts/training_logs/mobilenetv2_obb_sequence_geometry_v39.log"

echo "[RUN] Starting MobileNetV2 OBB-sequence-geometry v39 training."
echo "[RUN] Log file: $LOG_FILE"

RECTIFY_ALL=1 poetry run python -u scripts/run_training.py \
  --model-family mobilenet_v2_obb_sequence_geometry \
  --image-height 224 \
  --image-width 224 \
  --batch-size 8 \
  --epochs 16 \
  --learning-rate 2e-5 \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 96 \
  --mobilenet-head-dropout 0.15 \
  --mobilenet-warmup-epochs 4 \
  --mobilenet-unfreeze-last-n 12 \
  --geometry-value-loss-weight 2.0 \
  --keypoint-heatmap-loss-weight 0.25 \
  --keypoint-coord-loss-weight 0.5 \
  --edge-focus-strength 1.5 \
  --precomputed-crop-boxes data/rectified_crop_boxes_v5_all.csv \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 8 \
  --init-model artifacts/training/20260509_213225/model.keras \
  2>&1 | tee "$LOG_FILE"
