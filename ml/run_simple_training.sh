#!/bin/bash
# Run simple training script

cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

~/.cache/pypoetry/virtualenvs/embedded-gauge-reading-tinyml-fB-DUwEd-py3.12/bin/python scripts/train_simple.py \
    --train-manifest data/combined_training_manifest.csv \
    --val-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
    --epochs 50 \
    --batch-size 16
