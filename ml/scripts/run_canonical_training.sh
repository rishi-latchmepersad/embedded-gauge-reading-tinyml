#!/bin/bash
# Training script for canonical baseline

cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

# Run training using the existing run_training.py infrastructure
# with the canonical split files

.venv/bin/python scripts/run_training.py \
    --epochs 40 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --mobilenet-warmup-epochs 8 \
    --model-family mobilenet_v2 \
    --mobilenet-alpha 1.0 \
    --mobilenet-head-units 128 \
    --mobilenet-head-dropout 0.2 \
    --val-manifest data/splits/canonical_split_v1_val.csv \
    --test-manifest data/splits/canonical_split_v1_test.csv \
    --hard-case-manifest data/splits/canonical_split_v1_train.csv \
    --hard-case-repeat 1 \
    --seed 42 \
    --device cpu \
    2>&1 | tee artifacts/training_log.txt

echo "Training complete!"
