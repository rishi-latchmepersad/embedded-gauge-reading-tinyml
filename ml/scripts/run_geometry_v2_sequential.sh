#!/bin/bash
# Sequential geometric model experiments - run overnight

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting geometric model experiments (sequential)..."
echo "===================================================="

mkdir -p /tmp/gauge_geometry_v2

run_exp() {
    local name=$1
    local backbone=$2
    local alpha=$3
    local imagenet=$4
    local board_weight=$5
    local epochs1=$6
    local epochs2=$7
    local dropout=$8
    
    echo ""
    echo "[$name] Starting at $(date)"
    echo "  Backbone: $backbone (alpha=$alpha), ImageNet: $imagenet, Board Weight: $board_weight, Dropout: $dropout"
    
    args="--backbone $backbone --alpha $alpha --board-weight $board_weight --epochs-phase1 $epochs1 --epochs-phase2 $epochs2 --dropout $dropout --experiment $name --quantize"
    if [ "$imagenet" = "true" ]; then
        args="$args --imagenet"
    fi
    
    poetry run python train_geometry_v2.py $args 2>&1 | tee "/tmp/gauge_geometry_v2/${name}.log"
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$name] Complete at $(date)"
        # Extract MAE from log
        mae=$(grep "Board Temp MAE" /tmp/gauge_geometry_v2/${name}.log | head -1 | awk '{print $NF}')
        echo "[$name] Result: $mae"
    else
        echo "[$name] FAILED at $(date) (exit=$status)"
    fi
}

# EXP1: MobileNetV2 alpha=0.35 from scratch, board weight=15x
run_exp "mobilenetv2_a035_w15" "mobilenetv2" 0.35 "" 15 50 80 0.3

if grep -q "TARGET ACHIEVED" /tmp/gauge_geometry_v2/mobilenetv2_a035_w15.log 2>/dev/null; then
    echo "Target achieved! Stopping."
    exit 0
fi

# EXP2: MobileNetV2 alpha=0.35 from scratch, board weight=25x
run_exp "mobilenetv2_a035_w25" "mobilenetv2" 0.35 "" 25 50 80 0.3

if grep -q "TARGET ACHIEVED" /tmp/gauge_geometry_v2/mobilenetv2_a035_w25.log 2>/dev/null; then
    echo "Target achieved! Stopping."
    exit 0
fi

# EXP3: MobileNetV2 alpha=0.35 ImageNet init, board weight=10x
run_exp "mobilenetv2_a035_imagenet_w10" "mobilenetv2" 0.35 "true" 10 50 80 0.3

if grep -q "TARGET ACHIEVED" /tmp/gauge_geometry_v2/mobilenetv2_a035_imagenet_w10.log 2>/dev/null; then
    echo "Target achieved! Stopping."
    exit 0
fi

# EXP4: MobileNetV2 alpha=0.5 from scratch, board weight=20x
run_exp "mobilenetv2_a05_w20" "mobilenetv2" 0.5 "" 20 50 80 0.3

if grep -q "TARGET ACHIEVED" /tmp/gauge_geometry_v2/mobilenetv2_a05_w20.log 2>/dev/null; then
    echo "Target achieved! Stopping."
    exit 0
fi

# EXP5: MobileNetV2 alpha=0.25 from scratch, board weight=20x (even smaller)
run_exp "mobilenetv2_a025_w20" "mobilenetv2" 0.25 "" 20 50 80 0.3

if grep -q "TARGET ACHIEVED" /tmp/gauge_geometry_v2/mobilenetv2_a025_w20.log 2>/dev/null; then
    echo "Target achieved! Stopping."
    exit 0
fi

# EXP6: MobileNetV2 alpha=0.35 from scratch, heavier dropout
run_exp "mobilenetv2_a035_w20_drop05" "mobilenetv2" 0.35 "" 20 50 80 0.5

if grep -q "TARGET ACHIEVED" /tmp/gauge_geometry_v2/mobilenetv2_a035_w20_drop05.log 2>/dev/null; then
    echo "Target achieved! Stopping."
    exit 0
fi

# EXP7: EfficientNet-B0 from scratch, board weight=20x
run_exp "efficientnetb0_w20" "efficientnetb0" 1.0 "" 20 50 80 0.3

echo ""
echo "===================================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "===================================================="

# Summary
echo ""
echo "Summary:"
echo "--------"
for exp_dir in /tmp/gauge_geometry_v2/*/; do
    if [ -f "$exp_dir/board_eval.json" ]; then
        exp_name=$(basename "$exp_dir")
        mae=$(python3 -c "import json; d=json.load(open('$exp_dir/board_eval.json')); print(f\"{d['temp_mae']:.2f}°C\")" 2>/dev/null || echo "N/A")
        median=$(python3 -c "import json; d=json.load(open('$exp_dir/board_eval.json')); print(f\"{d['temp_median']:.2f}°C\")" 2>/dev/null || echo "N/A")
        echo "$exp_name: MAE=$mae, Median=$median"
    fi
done
