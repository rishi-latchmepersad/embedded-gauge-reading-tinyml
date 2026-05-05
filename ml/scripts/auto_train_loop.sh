#!/bin/bash
# Auto-train loop: trains, evaluates, and restarts with adjusted params until MAE < 5C on hard cases
# Usage: bash scripts/auto_train_loop.sh
# This runs indefinitely until a good model is found or manually stopped

set -e

export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

# Configuration
MAX_RUNS=10
TARGET_MAE=5.0
EPOCHS=80
BATCH_SIZE=8
DEVICE="gpu"

# Hyperparameter sweep strategy
# Run 1: conservative (low LR, moderate hard-case repeat)
# Run 2: more aggressive if needed
# Run 3+: adjust based on failure mode

LEARNING_RATES=(5e-5 3e-5 1e-4 5e-5 3e-5)
HARD_CASE_REPEATS=(4 6 3 8 2)
EDGE_FOCUS_STRENGTHS=(1.5 2.0 1.5 2.5 1.0)

RUN_DIR_BASE="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/training"
HARD_CASE_MANIFEST="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/data/hard_cases_plus_board30_valid_with_new6.csv"

echo "=========================================="
echo "AUTO-TRAIN LOOP"
echo "Target: MAE < ${TARGET_MAE}C on hard cases"
echo "Max runs: ${MAX_RUNS}"
echo "=========================================="

for RUN_NUM in $(seq 1 $MAX_RUNS); do
    LR="${LEARNING_RATES[$((RUN_NUM-1))]}"
    REPEAT="${HARD_CASE_REPEATS[$((RUN_NUM-1))]}"
    EDGE="${EDGE_FOCUS_STRENGTHS[$((RUN_NUM-1))]}"
    
    echo ""
    echo "=========================================="
    echo "RUN ${RUN_NUM}/${MAX_RUNS}"
    echo "LR=${LR}, Hard-case repeat=${REPEAT}, Edge focus=${EDGE}"
    echo "=========================================="
    
    # Train
    poetry run python scripts/train_hardcase_interval.py \
        --device ${DEVICE} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --learning-rate ${LR} \
        --hard-case-repeat ${REPEAT} \
        --edge-focus-strength ${EDGE} \
        --no-gpu-memory-growth \
        2>&1 | tee "${RUN_DIR_BASE}/auto_train_run_${RUN_NUM}.log"
    
    # Find the latest run directory
    LATEST_RUN=$(ls -td ${RUN_DIR_BASE}/hardcase_interval_* | head -1)
    echo "Latest run: ${LATEST_RUN}"
    
    # Check if training completed successfully
    if [ ! -f "${LATEST_RUN}/history.json" ]; then
        echo "WARNING: Training may not have completed. Checking for model..."
        if [ ! -f "${LATEST_RUN}/best_model.keras" ]; then
            echo "ERROR: No model found. Restarting WSL and trying again..."
            wsl --shutdown
            sleep 5
            continue
        fi
    fi
    
    # Evaluate on hard cases
    echo ""
    echo "Evaluating on hard cases..."
    
    # Create a quick eval script
    cat > /tmp/eval_hard_cases.py << 'PYEOF'
import sys
sys.path.insert(0, "/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/src")

import json
import numpy as np
from pathlib import Path
import tensorflow as tf

from embedded_gauge_reading_tinyml.training import load_gauge_specs, load_dataset, _build_training_examples
from embedded_gauge_reading_tinyml.presets import LABELLED_DIR, RAW_DIR

run_dir = sys.argv[1]
manifest_path = sys.argv[2]

# Load model
model_path = Path(run_dir) / "best_model.keras"
if not model_path.exists():
    model_path = Path(run_dir) / "final_model.keras"

print(f"Loading model from {model_path}")
model = tf.keras.models.load_model(str(model_path))

# Load hard case paths
hard_paths = set()
with open(manifest_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("image_path"):
            hard_paths.add(line.split(",")[0])

print(f"Hard cases to evaluate: {len(hard_paths)}")

# Load specs and dataset
specs = load_gauge_specs()
spec = specs["littlegood_home_temp_gauge_c"]
samples = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
examples, _ = _build_training_examples(samples, spec, image_height=224, image_width=224)

# Filter to hard cases
hard_examples = [e for e in examples if e.image_path in hard_paths]
print(f"Found {len(hard_examples)} hard examples in dataset")

# Evaluate
errors = []
for ex in hard_examples:
    img = tf.io.read_file(ex.image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, 0)
    
    pred = model(img, training=False)
    if isinstance(pred, dict):
        pred_val = float(pred["gauge_value"][0])
    else:
        pred_val = float(pred[0])
    
    error = abs(pred_val - ex.gauge_value)
    errors.append(error)

errors = np.array(errors)
mae = np.mean(errors)
max_err = np.max(errors)
pct_under_5 = np.mean(errors < 5.0) * 100

print(f"\n{'='*50}")
print(f"HARD CASE EVALUATION RESULTS")
print(f"{'='*50}")
print(f"MAE: {mae:.2f}C")
print(f"Max error: {max_err:.2f}C")
print(f"% under 5C: {pct_under_5:.1f}%")
print(f"{'='*50}")

# Save results
results = {
    "run_dir": str(run_dir),
    "mae": float(mae),
    "max_error": float(max_err),
    "pct_under_5c": float(pct_under_5),
    "num_evaluated": len(errors),
    "errors": [float(e) for e in errors]
}

with open(Path(run_dir) / "hard_case_eval.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {Path(run_dir) / 'hard_case_eval.json'}")

# Return exit code based on target
if mae < 5.0 and pct_under_5 > 80:
    print("\n✓ SUCCESS: Target achieved!")
    sys.exit(0)
else:
    print("\n✗ Target not achieved. Will retry with different hyperparameters.")
    sys.exit(1)
PYEOF

    if poetry run python /tmp/eval_hard_cases.py "${LATEST_RUN}" "${HARD_CASE_MANIFEST}"; then
        echo ""
        echo "=========================================="
        echo "TARGET ACHIEVED ON RUN ${RUN_NUM}!"
        echo "Model: ${LATEST_RUN}"
        echo "=========================================="
        exit 0
    else
        echo ""
        echo "Run ${RUN_NUM} did not meet target."
        
        # Check if we should restart WSL
        if [ $RUN_NUM -lt $MAX_RUNS ]; then
            echo "Restarting WSL before next run..."
            wsl --shutdown
            sleep 10
        fi
    fi
done

echo ""
echo "=========================================="
echo "MAX RUNS REACHED (${MAX_RUNS})"
echo "Check ${RUN_DIR_BASE} for all results"
echo "=========================================="
exit 1
