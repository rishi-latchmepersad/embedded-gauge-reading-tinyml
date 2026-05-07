#!/bin/bash
# Evaluation script for canonical baseline

set -e

# Setup GPU environment
CUDA_PY_LIB_PATHS="$(python3 -c "import site; import glob; import os; paths=[]; [paths.extend(glob.glob(os.path.join(b, 'nvidia', '*', 'lib'))) for b in site.getsitepackages()]; print(':'.join(paths))")"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CUDA_PY_LIB_PATHS}:${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml"
ML_ROOT="$PROJECT_ROOT/ml"

cd "$ML_ROOT"
source .venv/bin/activate

echo "Running evaluation..."
python3 scripts/evaluate_canonical_baseline.py \
    --model-path artifacts/canonical_baseline/canonical_baseline/model.keras \
    --test-csv data/splits/canonical_split_v1_test.csv

echo "Evaluation complete!"
