#!/bin/bash
# Train MobileNetV2-interval model with aggressive hard-case focus.
#
# This script trains the strongest known configuration for getting <5C MAE on
# hard cases while keeping the MobileNetV2 backbone for STM32N6 deployment.
#
# Key features:
# - MobileNetV2-interval head (coarse 5C bins + residual correction)
# - Full backbone trainable from scratch (ImageNet warm-start)
# - Range-aware sampling: 4x oversampling of cold (-30C) and hot (45-50C) tails
# - Hard-case repeat: 12x repetition of known difficult captures
# - Long training: 80 epochs with cosine decay
# - Aggressive augmentation + edge focus
# - Unified dataset: 409 images covering full -30C to 50C range
#
# Usage (from ml/ directory in WSL with Poetry environment):
#   bash scripts/train_hardcase_interval.sh

set -euo pipefail

cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

# Find the Poetry virtual environment Python
VENV_PATH=""
for path in ~/.cache/pypoetry/virtualenvs/*/bin/python; do
    if [ -x "$path" ]; then
        VENV_PATH="$path"
        break
    fi
done

if [ -z "$VENV_PATH" ]; then
    echo "ERROR: Could not find Poetry virtual environment Python." >&2
    echo "Make sure Poetry is installed and the project dependencies are set up." >&2
    echo "Run: poetry install --with dev" >&2
    exit 1
fi

echo "Using Python: $VENV_PATH"

# Run the training script with hard-case focused configuration
$VENV_PATH scripts/train_hardcase_interval.py \
    --model-family mobilenet_v2_interval \
    --epochs 80 \
    --batch-size 8 \
    --learning-rate 0.0001 \
    --mobilenet-alpha 1.0 \
    --mobilenet-head-units 128 \
    --mobilenet-head-dropout 0.2 \
    --interval-bin-width 5.0 \
    --edge-focus-strength 1.5 \
    --range-aware-sampling \
    --cold-tail-fraction 0.20 \
    --hot-tail-fraction 0.20 \
    --oversampling-factor 4.0 \
    --hard-case-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
    --hard-case-repeat 12 \
    --val-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
    --artifacts-dir artifacts/training \
    --run-name hardcase_interval_v1 \
    --image-height 224 \
    --image-width 224 \
    --seed 42 \
    --device gpu

echo "Training complete! Check artifacts/training/hardcase_interval_v1/ for results."
