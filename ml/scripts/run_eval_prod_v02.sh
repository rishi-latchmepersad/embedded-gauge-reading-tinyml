#!/usr/bin/env bash
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

echo "=== prod_model_v0.2 on 33C board captures ==="
CUDA_VISIBLE_DEVICES=-1 poetry run python scripts/eval_scalar_fixed_crop_yuv422.py \
  --model artifacts/deployment/prod_model_v0.2_raw_int8/model_int8.tflite \
  --captures-dir /mnt/d/Projects/embedded-gauge-reading-tinyml/captured_images \
  --pattern "capture_2026-04-18_17-*.yuv422" \
  --true-value 33.0
