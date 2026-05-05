#!/bin/bash
# Monitor training progress and auto-restart if needed
# Usage: bash scripts/monitor_and_restart.sh <RUN_DIR>

set -e

export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

RUN_DIR="${1:-}"
TARGET_MAE=5.0
CHECK_INTERVAL=300  # 5 minutes
STALL_TIMEOUT=1800  # 30 minutes without progress = stalled

if [ -z "$RUN_DIR" ]; then
    echo "Usage: bash scripts/monitor_and_restart.sh <RUN_DIR>"
    echo "Example: bash scripts/monitor_and_restart.sh artifacts/training/hardcase_interval_20260503_204246"
    exit 1
fi

RUN_DIR="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/${RUN_DIR}"
HISTORY_FILE="${RUN_DIR}/history.json"

echo "Monitoring: ${RUN_DIR}"
echo "Target MAE: ${TARGET_MAE}C"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

LAST_EPOCH=0
LAST_CHECK=$(date +%s)

while true; do
    sleep ${CHECK_INTERVAL}
    
    CURRENT_TIME=$(date +%s)
    
    # Check if history file exists
    if [ ! -f "${HISTORY_FILE}" ]; then
        echo "$(date): History file not found yet..."
        
        # Check for stall
        ELAPSED=$((CURRENT_TIME - LAST_CHECK))
        if [ $ELAPSED -gt $STALL_TIMEOUT ]; then
            echo "$(date): STALL DETECTED - No history file for ${ELAPSED}s"
            echo "$(date): Training may be stuck. Check manually."
        fi
        continue
    fi
    
    # Parse history
    LATEST_EPOCH=$(python3 -c "
import json
with open('${HISTORY_FILE}') as f:
    h = json.load(f)
epochs = h.get('epoch', [])
if epochs:
    print(epochs[-1])
else:
    print(0)
" 2>/dev/null || echo "0")
    
    if [ "$LATEST_EPOCH" = "$LAST_EPOCH" ]; then
        ELAPSED=$((CURRENT_TIME - LAST_CHECK))
        if [ $ELAPSED -gt $STALL_TIMEOUT ]; then
            echo "$(date): STALL DETECTED - Stuck at epoch ${LATEST_EPOCH} for ${ELAPSED}s"
            echo "$(date): Consider restarting WSL: wsl --shutdown"
        else
            echo "$(date): Still at epoch ${LATEST_EPOCH} (${ELAPSED}s elapsed)"
        fi
    else
        echo "$(date): Progress - Epoch ${LATEST_EPOCH}"
        LAST_EPOCH=$LATEST_EPOCH
        LAST_CHECK=$CURRENT_TIME
        
        # Check if training is complete
        if [ -f "${RUN_DIR}/best_model.keras" ]; then
            echo "$(date): Training complete! Best model saved."
            
            # Get best val MAE
            BEST_MAE=$(python3 -c "
import json
with open('${HISTORY_FILE}') as f:
    h = json.load(f)
val_mae = h.get('val_gauge_value_mae', [])
if val_mae:
    print(f'{min(val_mae):.2f}')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
            
            echo "$(date): Best validation MAE: ${BEST_MAE}C"
            
            if [ "$BEST_MAE" != "N/A" ]; then
                # Compare as float
                IS_GOOD=$(python3 -c "print('YES' if float('${BEST_MAE}') < ${TARGET_MAE} else 'NO')")
                if [ "$IS_GOOD" = "YES" ]; then
                    echo "$(date): TARGET ACHIEVED! MAE ${BEST_MAE}C < ${TARGET_MAE}C"
                    echo "$(date): Model: ${RUN_DIR}"
                    exit 0
                else
                    echo "$(date): Target not achieved. Best MAE: ${BEST_MAE}C (target: ${TARGET_MAE}C)"
                    echo "$(date): Consider running with different hyperparameters."
                fi
            fi
            
            exit 0
        fi
    fi
done
