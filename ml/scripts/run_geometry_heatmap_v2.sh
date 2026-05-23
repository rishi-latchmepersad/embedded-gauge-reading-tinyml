#!/bin/bash
# Geometry Heatmap v2 - Full Pipeline
# Runs tests, training, evaluation, overlays, and jitter robustness analysis.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Geometry Heatmap v2 - Full Pipeline"
echo "========================================"

echo ""
echo "Step 1: Tests"
echo "----------------------------------------"
poetry run pytest tests/test_gauge_geometry.py tests/test_geometry_crop_dataset.py tests/test_heatmap_utils.py tests/test_heatmap_losses.py

echo ""
echo "Step 2: Training"
echo "----------------------------------------"
poetry run python scripts/train_geometry_heatmap_v2.py

echo ""
echo "Step 3: Evaluation and overlays"
echo "----------------------------------------"
poetry run python scripts/eval_geometry_heatmap_v2.py

echo ""
echo "Step 4: Jitter robustness"
echo "----------------------------------------"
poetry run python scripts/eval_geometry_heatmap_v2_jitter_robustness.py

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  - Model: ml/artifacts/training/geometry_heatmap_v2/model.keras"
echo "  - History: ml/artifacts/training/geometry_heatmap_v2/history.csv"
echo "  - Config: ml/artifacts/training/geometry_heatmap_v2/config.json"
echo "  - Eval report: ml/reports/geometry_heatmap_v2_eval.md"
echo "  - Jitter report: ml/reports/geometry_heatmap_v2_jitter_robustness.md"
echo "  - Overlays: ml/debug/geometry_heatmap_v2_predictions/"
