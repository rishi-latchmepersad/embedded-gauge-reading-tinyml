#!/bin/bash
# Training script for combined manifest

cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

VENV_PATH=~/.cache/pypoetry/virtualenvs/embedded-gauge-reading-tinyml-fB-DUwEd-py3.12/bin/python

# Use hard-case manifest for additional training data
# The base training data comes from data/labelled (CVAT exports)
$VENV_PATH scripts/run_training.py \
    --model-family mobilenet_v2 \
    --hard-case-manifest data/combined_training_manifest.csv \
    --hard-case-repeat 3 \
    --val-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --artifacts-dir artifacts/training \
    --run-name combined_147_samples \
    --image-height 224 \
    --image-width 224 \
    --seed 42 \
    --device gpu
