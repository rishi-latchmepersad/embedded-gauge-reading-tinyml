#!/bin/bash
# Run multiple transfer learning experiments in parallel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting transfer learning experiments..."
echo "=========================================="

# Create output directory
mkdir -p /tmp/gauge_transfer_learning

# Experiment 1: MobileNetV2 from scratch with heavy board weighting
echo ""
echo "[EXP1] MobileNetV2 from scratch, board weight=15x"
poetry run python train_transfer_learning.py \
    --backbone mobilenetv2 \
    --board-weight 15 \
    --epochs-phase1 40 \
    --epochs-phase2 60 \
    --experiment mobilenetv2_scratch_w15 \
    --quantize \
    > /tmp/gauge_transfer_learning/exp1_mobilenetv2_scratch.log 2>&1 &
PID1=$!

# Experiment 2: EfficientNet-B0 from scratch with heavy board weighting
echo "[EXP2] EfficientNet-B0 from scratch, board weight=15x"
poetry run python train_transfer_learning.py \
    --backbone efficientnetb0 \
    --board-weight 15 \
    --epochs-phase1 40 \
    --epochs-phase2 60 \
    --experiment efficientnetb0_scratch_w15 \
    --quantize \
    > /tmp/gauge_transfer_learning/exp2_efficientnetb0_scratch.log 2>&1 &
PID2=$!

# Experiment 3: MobileNetV2 ImageNet init with moderate board weighting
echo "[EXP3] MobileNetV2 ImageNet init, board weight=10x"
poetry run python train_transfer_learning.py \
    --backbone mobilenetv2 \
    --imagenet \
    --board-weight 10 \
    --epochs-phase1 30 \
    --epochs-phase2 50 \
    --experiment mobilenetv2_imagenet_w10 \
    --quantize \
    > /tmp/gauge_transfer_learning/exp3_mobilenetv2_imagenet.log 2>&1 &
PID3=$!

# Experiment 4: MobileNetV2 from scratch with very heavy board weighting
echo "[EXP4] MobileNetV2 from scratch, board weight=25x"
poetry run python train_transfer_learning.py \
    --backbone mobilenetv2 \
    --board-weight 25 \
    --epochs-phase1 40 \
    --epochs-phase2 60 \
    --experiment mobilenetv2_scratch_w25 \
    --quantize \
    > /tmp/gauge_transfer_learning/exp4_mobilenetv2_w25.log 2>&1 &
PID4=$!

# Experiment 5: EfficientNet-B0 ImageNet init
echo "[EXP5] EfficientNet-B0 ImageNet init, board weight=10x"
poetry run python train_transfer_learning.py \
    --backbone efficientnetb0 \
    --imagenet \
    --board-weight 10 \
    --epochs-phase1 30 \
    --epochs-phase2 50 \
    --experiment efficientnetb0_imagenet_w10 \
    --quantize \
    > /tmp/gauge_transfer_learning/exp5_efficientnetb0_imagenet.log 2>&1 &
PID5=$!

echo ""
echo "All experiments started (PIDs: $PID1, $PID2, $PID3, $PID4, $PID5)"
echo "Waiting for completion..."
echo ""

# Wait for all experiments
wait $PID1 && echo "[EXP1] Complete" || echo "[EXP1] Failed"
wait $PID2 && echo "[EXP2] Complete" || echo "[EXP2] Failed"
wait $PID3 && echo "[EXP3] Complete" || echo "[EXP3] Failed"
wait $PID4 && echo "[EXP4] Complete" || echo "[EXP4] Failed"
wait $PID5 && echo "[EXP5] Complete" || echo "[EXP5] Failed"

echo ""
echo "=========================================="
echo "All experiments finished!"
echo "Results in /tmp/gauge_transfer_learning/"
echo ""

# Show summary
echo "Summary:"
echo "--------"
for exp_dir in /tmp/gauge_transfer_learning/*/; do
    if [ -f "$exp_dir/board_eval.json" ]; then
        exp_name=$(basename "$exp_dir")
        mae=$(python3 -c "import json; print(json.load(open('$exp_dir/board_eval.json'))['mae'])")
        echo "$exp_name: MAE = ${mae}°C"
    fi
done
