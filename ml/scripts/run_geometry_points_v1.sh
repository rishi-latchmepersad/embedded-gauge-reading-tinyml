#!/bin/bash
# Run script for geometry points v1 training pipeline.
# This script runs training, evaluation, and visualization generation.
#
# Usage:
#   bash ml/scripts/run_geometry_points_v1.sh
#
# Or from the ml directory:
#   bash scripts/run_geometry_points_v1.sh

set -e

echo "========================================"
echo "Geometry Points v1 - Full Pipeline"
echo "========================================"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo ""
echo "Step 1: Training..."
echo "----------------------------------------"
poetry run python scripts/train_geometry_points_v1.py

echo ""
echo "Step 2: Evaluation..."
echo "----------------------------------------"
poetry run python scripts/eval_geometry_points_v1.py

echo ""
echo "Step 3: Prediction Visualizations..."
echo "----------------------------------------"
poetry run python scripts/generate_geometry_predictions_viz.py

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  Model: ml/artifacts/training/geometry_points_v1/model.keras"
echo "  History: ml/artifacts/training/geometry_points_v1/history.csv"
echo "  Config: ml/artifacts/training/geometry_points_v1/config.json"
echo "  Test predictions: ml/artifacts/training/geometry_points_v1/test_predictions.csv"
echo "  Worst 30: ml/artifacts/training/geometry_points_v1/worst_30_predictions.csv"
echo "  Report: ml/reports/geometry_points_v1_eval.md"
echo "  Visualizations: ml/debug/geometry_points_v1_predictions/"
