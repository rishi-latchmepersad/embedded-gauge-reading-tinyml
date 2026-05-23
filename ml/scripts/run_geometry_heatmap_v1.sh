#!/bin/bash
# Geometry Heatmap v1 - Full Pipeline
# Trains, evaluates, and generates diagnostics for heatmap-based geometry model

set -e

cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml

echo "========================================"
echo "Geometry Heatmap v1 - Full Pipeline"
echo "========================================"
echo ""

# Step 1: Run tests
echo "Step 1: Running tests..."
echo "----------------------------------------"
~/.local/bin/poetry run pytest tests/test_gauge_geometry.py tests/test_geometry_crop_dataset.py tests/test_heatmap_utils.py -v
echo ""

# Step 2: Training
echo "Step 2: Training..."
echo "----------------------------------------"
~/.local/bin/poetry run python scripts/train_geometry_heatmap_v1.py
echo ""

# Step 3: Evaluation
echo "Step 3: Evaluation..."
echo "----------------------------------------"
~/.local/bin/poetry run python scripts/eval_geometry_heatmap_v1.py     --model-path artifacts/training/geometry_heatmap_v1/model.keras     --manifest-path data/geometry_reader_manifest_v2_clean.csv     --output-dir artifacts/training/geometry_heatmap_v1
echo ""

echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  - Model: artifacts/training/geometry_heatmap_v1/model.keras"
echo "  - History: artifacts/training/geometry_heatmap_v1/history.csv"
echo "  - Predictions: artifacts/training/geometry_heatmap_v1/test_predictions.csv"
echo "  - Metrics: artifacts/training/geometry_heatmap_v1/eval_metrics.json"
echo "  - Report: reports/geometry_heatmap_v1_eval.md"
