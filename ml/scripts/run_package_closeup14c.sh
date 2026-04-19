#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
REPO="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
cd "$REPO"
CUDA_VISIBLE_DEVICES=-1 poetry run python -u scripts/package_scalar_model_for_n6.py \
  --model artifacts/deployment/scalar_full_finetune_closeup14c_int8/model_int8.tflite \
  --output-dir artifacts/runtime/scalar_full_finetune_closeup14c_int8_reloc \
  --workspace-dir ../st_ai_output/packages/scalar_full_finetune_closeup14c/st_ai_ws \
  --stai-output-dir ../st_ai_output/packages/scalar_full_finetune_closeup14c/st_ai_output \
  --name scalar_full_finetune_closeup14c \
  --canonical-raw-path ../st_ai_output/atonbuf.xSPI2.raw \
  --compression high \
  --optimization balanced
