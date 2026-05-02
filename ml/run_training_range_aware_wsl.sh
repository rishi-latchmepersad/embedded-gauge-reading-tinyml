#!/bin/bash
# Training script for range-aware sampling experiment
# Uses linear output head with post-training calibration

cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

VENV_PATH=~/.cache/pypoetry/virtualenvs/embedded-gauge-reading-tinyml-fB-DUwEd-py3.12/bin/python

# Run training with range-aware sampling and linear output
$VENV_PATH scripts/run_training_range_aware.py \
    --model-family mobilenet_v2 \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --range-aware-sampling \
    --cold-tail-fraction 0.15 \
    --hot-tail-fraction 0.15 \
    --oversampling-factor 3.0 \
    --linear-output \
    --val-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
    --artifacts-dir artifacts/training/range_aware_linear \
    --run-name range_aware_linear_20260502 \
    --image-height 224 \
    --image-width 224 \
    --seed 42 \
    --device gpu
