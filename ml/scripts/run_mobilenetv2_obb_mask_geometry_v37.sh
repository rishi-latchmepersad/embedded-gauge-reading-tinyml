#!/usr/bin/env bash
set -euo pipefail

# Train the literature-backed OBB + mask + keypoint geometry reader.
cd "$(dirname "$0")/.."

mkdir -p artifacts/training_logs

LOG_FILE="artifacts/training_logs/mobilenetv2_obb_mask_geometry_v37.log"

echo "[RUN] Starting MobileNetV2 OBB-mask-geometry v37 training."
echo "[RUN] Log file: $LOG_FILE"

# Use the strict v5 reader as a backbone warm-start so the shared MobileNetV2
# trunk starts from the strongest scalar baseline we have, while the new heads
# learn pointer-mask and keypoint supervision on top.
RECTIFY_ALL=1 poetry run python -u scripts/run_training.py \
  --model-family mobilenet_v2_obb_mask_geometry \
  --image-height 224 \
  --image-width 224 \
  --batch-size 8 \
  --epochs 16 \
  --learning-rate 5e-5 \
  --mobilenet-alpha 0.35 \
  --mobilenet-head-units 96 \
  --mobilenet-head-dropout 0.15 \
  --mobilenet-warmup-epochs 4 \
  --mobilenet-unfreeze-last-n 12 \
  --edge-focus-strength 0.5 \
  --precomputed-crop-boxes data/rectified_crop_boxes_v5_all.csv \
  --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
  --hard-case-repeat 2 \
  --init-model artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras \
  2>&1 | tee "$LOG_FILE"
