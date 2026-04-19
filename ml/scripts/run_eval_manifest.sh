#!/usr/bin/env bash
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
CUDA_VISIBLE_DEVICES=-1 poetry run python scripts/eval_keras_on_manifest.py \
  --model artifacts/training/scalar_full_finetune_closeup14c/model.keras \
  --manifest data/hard_cases_plus_board30_valid_with_new5_closeup14c.csv
