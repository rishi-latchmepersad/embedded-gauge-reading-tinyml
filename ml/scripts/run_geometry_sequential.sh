#!/bin/bash
# Run geometric model experiments sequentially

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting geometric model experiments (sequential)..."
echo "===================================================="

mkdir -p /tmp/gauge_geometry

run_exp() {
    local name=$1
    local backbone=$2
    local imagenet=$3
    local board_weight=$4
    local epochs1=$5
    local epochs2=$6
    local dropout=$7
    
    echo ""
    echo "[$name] Starting..."
    echo "  Backbone: $backbone, ImageNet: $imagenet, Board Weight: $board_weight"
    
    poetry run python train_geometry_model.py \
        --backbone "$backbone" \
        ${imagenet:+--imagenet} \
        --board-weight "$board_weight" \
        --epochs-phase1 "$epochs1" \
        --epochs-phase2 "$epochs2" \
        --dropout "$dropout" \
        --experiment "$name" \
        --quantize \
        2>&1 | tee "/tmp/gauge_geometry/${name}.log"
    
    echo "[$name] Complete"
}

# EXP1: MobileNetV2 from scratch, heavy board weighting
run_exp "mobilenetv2_scratch_w20" "mobilenetv2" "" 20 50 80 0.3

# EXP2: EfficientNet-B0 from scratch
run_exp "efficientnetb0_scratch_w20" "efficientnetb0" "" 20 50 80 0.3

# EXP3: MobileNetV2 ImageNet init
run_exp "mobilenetv2_imagenet_w15" "mobilenetv2" "true" 15 40 60 0.3

# EXP4: MobileNetV2 from scratch, very heavy board weighting
run_exp "mobilenetv2_scratch_w30" "mobilenetv2" "" 30 50 80 0.3

# EXP5: EfficientNet-B0 ImageNet init
run_exp "efficientnetb0_imagenet_w15" "efficientnetb0" "true" 15 40 60 0.3

# EXP6: MobileNetV2 from scratch, lower dropout
run_exp "mobilenetv2_scratch_drop02" "mobilenetv2" "" 20 50 80 0.2

echo ""
echo "===================================================="
echo "All experiments complete!"
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
