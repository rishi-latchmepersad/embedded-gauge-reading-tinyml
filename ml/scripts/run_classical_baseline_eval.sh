#!/usr/bin/env bash
# Evaluate the classical CV baseline on the two canonical acceptance manifests.
# Run from WSL Ubuntu-24.04:
#   wsl -d Ubuntu-24.04 -e bash /mnt/d/Projects/embedded-gauge-reading-tinyml/ml/scripts/run_classical_baseline_eval.sh

set -e

export PATH=/home/rishi_latchmepersad/.local/bin:$PATH

REPO=/mnt/d/Projects/embedded-gauge-reading-tinyml
ML=$REPO/ml
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/ml/artifacts/baseline/classical_cv_$TIMESTAMP
mkdir -p "$OUT"

echo "=== Classical CV baseline ==="
echo "Output: $OUT"
echo

GAUGE_ID=littlegood_home_temp_gauge_c

cd "$ML"

for MANIFEST in hard_cases.csv hard_cases_plus_board30_valid_with_new5.csv; do
    NAME="${MANIFEST%.csv}"
    echo "--- $MANIFEST ---"
    poetry run python -u scripts/eval_classical_baseline_on_manifest.py \
        --manifest "data/$MANIFEST" \
        --gauge-id "$GAUGE_ID" \
        --predictions-csv "$OUT/${NAME}_predictions.csv" \
        --report-dir "$OUT/${NAME}_report" \
        2>&1 | tee "$OUT/${NAME}.log"
    echo
done

echo "Done. Results in $OUT"
