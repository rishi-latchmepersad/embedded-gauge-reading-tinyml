#!/bin/bash
# Simple training monitor - checks progress and reports status
# Run this in a separate WSL terminal while training
# Usage: bash scripts/simple_monitor.sh [RUN_DIR]

set -e

export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

RUN_DIR="${1:-}"
TARGET_MAE=5.0

if [ -z "$RUN_DIR" ]; then
    # Auto-detect latest run
    RUN_DIR=$(ls -td artifacts/training/hardcase_interval_* 2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "No training run found. Start training first."
        exit 1
    fi
fi

HISTORY_FILE="${RUN_DIR}/history.json"
CONFIG_FILE="${RUN_DIR}/config.json"

echo "=========================================="
echo "MONITORING: $(basename ${RUN_DIR})"
echo "=========================================="

if [ -f "${CONFIG_FILE}" ]; then
    echo "Config:"
    cat "${CONFIG_FILE}" | python3 -m json.tool 2>/dev/null | grep -E "(learning_rate|hard_case_repeat|epochs|batch_size)" || true
    echo ""
fi

LAST_EPOCH=0
BEST_MAE=999
BEST_EPOCH=0

while true; do
    sleep 60
    
    if [ ! -f "${HISTORY_FILE}" ]; then
        echo "$(date '+%H:%M:%S'): Waiting for history.json..."
        continue
    fi
    
    # Parse history
    python3 << 'PYEOF'
import json
import sys

try:
    with open('${HISTORY_FILE}') as f:
        h = json.load(f)
    
    epochs = h.get('epoch', [])
    if not epochs:
        print("NO_DATA")
        sys.exit(0)
    
    latest_epoch = epochs[-1]
    
    train_mae = h.get('gauge_value_mae', [])
    val_mae = h.get('val_gauge_value_mae', [])
    train_loss = h.get('loss', [])
    val_loss = h.get('val_loss', [])
    
    if val_mae:
        best_idx = min(range(len(val_mae)), key=lambda i: val_mae[i])
        best_epoch = epochs[best_idx] if best_idx < len(epochs) else 0
        best_mae = val_mae[best_idx]
        
        print(f"EPOCH:{latest_epoch}")
        print(f"BEST_EPOCH:{best_epoch}")
        print(f"BEST_MAE:{best_mae:.4f}")
        print(f"LATEST_TRAIN_MAE:{train_mae[-1]:.4f}" if train_mae else "LATEST_TRAIN_MAE:N/A")
        print(f"LATEST_VAL_MAE:{val_mae[-1]:.4f}" if val_mae else "LATEST_VAL_MAE:N/A")
        print(f"LATEST_TRAIN_LOSS:{train_loss[-1]:.4f}" if train_loss else "LATEST_TRAIN_LOSS:N/A")
        print(f"LATEST_VAL_LOSS:{val_loss[-1]:.4f}" if val_loss else "LATEST_VAL_LOSS:N/A")
    else:
        print("NO_VAL_DATA")
except Exception as e:
    print(f"ERROR:{e}")
PYEOF
    
    # Check if training is complete
    if [ -f "${RUN_DIR}/best_model.keras" ]; then
        echo ""
        echo "=========================================="
        echo "TRAINING COMPLETE!"
        echo "Best model: ${RUN_DIR}/best_model.keras"
        echo "=========================================="
        
        if (( $(echo "$BEST_MAE < $TARGET_MAE" | bc -l) )); then
            echo "TARGET ACHIEVED! MAE: ${BEST_MAE}C < ${TARGET_MAE}C"
        else
            echo "Target not achieved. Best MAE: ${BEST_MAE}C (target: ${TARGET_MAE}C)"
        fi
        
        exit 0
    fi
done
