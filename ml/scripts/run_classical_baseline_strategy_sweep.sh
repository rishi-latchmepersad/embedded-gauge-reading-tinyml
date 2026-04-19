#!/usr/bin/env bash
# Sweep classical CV geometry strategies across the hard-case manifests.
# Run from WSL Ubuntu-24.04:
#   wsl -d Ubuntu-24.04 -e bash /mnt/d/Projects/embedded-gauge-reading-tinyml/ml/scripts/run_classical_baseline_strategy_sweep.sh

set -e

export PATH=/home/rishi_latchmepersad/.local/bin:$PATH

REPO=/mnt/d/Projects/embedded-gauge-reading-tinyml
ML=$REPO/ml
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/ml/artifacts/baseline/classical_cv_sweep_$TIMESTAMP
mkdir -p "$OUT"

echo "=== Classical CV strategy sweep ==="
echo "Output: $OUT"
echo

cd "$ML"

poetry run python -u scripts/sweep_classical_baseline_strategies.py \
    --report-root "$OUT" \
    "$@" \
    2>&1 | tee "$OUT/sweep.log"

echo
echo "Done. Results in $OUT"
