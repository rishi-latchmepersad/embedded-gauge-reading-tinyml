#!/bin/bash
# Run geometric model experiments in parallel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting geometric model experiments..."
echo "=========================================="

mkdir -p /tmp/gauge_geometry

# EXP1: MobileNetV2 from scratch, heavy board weighting
echo "[EXP1] MobileNetV2 from scratch, board weight=20x"
poetry run python train_geometry_model.py \
    --backbone mobilenetv2 \
    --board-weight 20 \
    --epochs-phase1 50 \
    --epochs-phase2 80 \
    --experiment mobilenetv2_scratch_w20 \
    --quantize \
    > /tmp/gauge_geometry/exp1_mobilenetv2_scratch.log 2>&1 &
PID1=$!

# EXP2: EfficientNet-B0 from scratch, heavy board weighting
echo "[EXP2] EfficientNet-B0 from scratch, board weight=20x"
poetry run python train_geometry_model.py \
    --backbone efficientnetb0 \
    --board-weight 20 \
    --epochs-phase1 50 \
    --epochs-phase2 80 \
    --experiment efficientnetb0_scratch_w20 \
    --quantize \
    > /tmp/gauge_geometry/exp2_efficientnetb0_scratch.log 2>&1 &
PID2=$!

# EXP3: MobileNetV2 ImageNet init, moderate board weighting
echo "[EXP3] MobileNetV2 ImageNet init, board weight=15x"
poetry run python train_geometry_model.py \
    --backbone mobilenetv2 \
    --imagenet \
    --board-weight 15 \
    --epochs-phase1 40 \
    --epochs-phase2 60 \
    --experiment mobilenetv2_imagenet_w15 \
    --quantize \
    > /tmp/gauge_geometry/exp3_mobilenetv2_imagenet.log 2>&1 &
PID3=$!

# EXP4: MobileNetV2 from scratch, very heavy board weighting
echo "[EXP4] MobileNetV2 from scratch, board weight=30x"
poetry run python train_geometry_model.py \
    --backbone mobilenetv2 \
    --board-weight 30 \
    --epochs-phase1 50 \
    --epochs-phase2 80 \
    --experiment mobilenetv2_scratch_w30 \
    --quantize \
    > /tmp/gauge_geometry/exp4_mobilenetv2_w30.log 2>&1 &
PID4=$!

# EXP5: EfficientNet-B0 ImageNet init
echo "[EXP5] EfficientNet-B0 ImageNet init, board weight=15x"
poetry run python train_geometry_model.py \
    --backbone efficientnetb0 \
    --imagenet \
    --board-weight 15 \
    --epochs-phase1 40 \
    --epochs-phase2 60 \
    --experiment efficientnetb0_imagenet_w15 \
    --quantize \
    > /tmp/gauge_geometry/exp5_efficientnetb0_imagenet.log 2>&1 &
PID5=$!

# EXP6: MobileNetV2 from scratch, lower dropout
echo "[EXP6] MobileNetV2 from scratch, dropout=0.2, board weight=20x"
poetry run python train_geometry_model.py \
    --backbone mobilenetv2 \
    --board-weight 20 \
    --dropout 0.2 \
    --epochs-phase1 50 \
    --epochs-phase2 80 \
    --experiment mobilenetv2_scratch_drop02 \
    --quantize \
    > /tmp/gauge_geometry/exp6_mobilenetv2_drop02.log 2>&1 &
PID6=$!

echo ""
echo "All experiments started (PIDs: $PID1, $PID2, $PID3, $PID4, $PID5, $PID6)"
echo "Logs in /tmp/gauge_geometry/"
echo ""
echo "Monitor with: tail -f /tmp/gauge_geometry/exp*_*.log"
echo ""

# Wait for all
wait

echo ""
echo "=========================================="
echo "All experiments finished!"
echo ""

# Show summary
echo "Summary:"
echo "--------"
for exp_dir in /tmp/gauge_geometry/*/; do
    if [ -f "$exp_dir/board_eval.json" ]; then
        exp_name=$(basename "$exp_dir")
        mae=$(python3 -c "import json; print(json.load(open('$exp_dir/board_eval.json'))['temp_mae'])" 2>/dev/null || echo "N/A")
        angle_mae=$(python3 -c "import json; print(json.load(open('$exp_dir/board_eval.json'))['angle_mae'])" 2>/dev/null || echo "N/A")
        echo "$exp_name: Angle MAE=${angle_mae}°, Temp MAE=${mae}°C"
    fi
done
