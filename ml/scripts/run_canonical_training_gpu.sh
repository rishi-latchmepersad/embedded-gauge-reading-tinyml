#!/bin/bash
# Training script for canonical baseline with GPU support
# IMPORTANT: Run this AFTER restarting WSL

set -euo pipefail

PROJECT_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml"
ML_ROOT="$PROJECT_ROOT/ml"
DATA_DIR="$ML_ROOT/data"
ARTIFACTS_DIR="$ML_ROOT/artifacts/canonical_baseline"

# Create output directory
mkdir -p "$ARTIFACTS_DIR"

cd "$ML_ROOT"
source .venv/bin/activate

# Setup GPU environment - this is the key fix for GPU support
CUDA_PY_LIB_PATHS="$(python3 -c "import site; import glob; import os; paths=[]; [paths.extend(glob.glob(os.path.join(b, 'nvidia', '*', 'lib'))) for b in site.getsitepackages()]; print(':'.join(paths))")"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CUDA_PY_LIB_PATHS}:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

# Verify GPU is available
echo "Verifying GPU availability..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs found: {len(gpus)}'); [print(f'  - {g}') for g in gpus]"

# Training parameters
EPOCHS=40
WARMUP_EPOCHS=8
BATCH_SIZE=8
SEED=42
LEARNING_RATE=0.0001

echo "=========================================="
echo "Training Canonical Baseline CNN"
echo "=========================================="
echo "Epochs: $EPOCHS (warmup: $WARMUP_EPOCHS)"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Output: $ARTIFACTS_DIR"
echo "=========================================="

# Run training using the existing run_training.py infrastructure
# with the canonical split files
python3 scripts/run_training.py \
    --model-family mobilenet_v2 \
    --epochs $EPOCHS \
    --mobilenet-warmup-epochs $WARMUP_EPOCHS \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    --learning-rate $LEARNING_RATE \
    --val-manifest data/splits/canonical_split_v1_val.csv \
    --test-manifest data/splits/canonical_split_v1_test.csv \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --run-name "canonical_baseline" \
    --device gpu \
    2>&1 | tee "$ARTIFACTS_DIR/training.log"

echo "=========================================="
echo "Training complete!"
echo "Output: $ARTIFACTS_DIR"
echo "=========================================="
