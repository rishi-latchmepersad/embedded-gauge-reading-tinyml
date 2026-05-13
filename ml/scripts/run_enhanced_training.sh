#!/usr/bin/env bash
# Train all enhanced model variants and compare against baseline.
# Run from WSL: bash ml/scripts/run_enhanced_training.sh
set -euo pipefail

REPO_ROOT="/mnt/d/Projects/embedded-gauge-reading-tinyml/ml"
LOG_DIR="${REPO_ROOT}/artifacts/training_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi
if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry not found." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "=========================================="
echo " Enhanced Model Training Suite"
echo " Timestamp: ${TIMESTAMP}"
echo "=========================================="

# ─── Variant 1: Baseline (linear output, no attention) ─────────────────────
echo ""
echo "=========================================="
echo " Variant 1/4: Baseline (linear output)"
echo "=========================================="
"${POETRY_BIN}" run python -u scripts/train_enhanced_model.py \
  --variant baseline \
  --epochs 60 \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --head-units 128 \
  --head-dropout 0.2 \
  --seed 42 \
  --run-name "enhanced_baseline_linear_e60_s42" \
  2>&1 | tee "${LOG_DIR}/enhanced_baseline_${TIMESTAMP}.log"

# ─── Variant 2: Coordinate Attention ────────────────────────────────────────
echo ""
echo "=========================================="
echo " Variant 2/4: Coordinate Attention"
echo "=========================================="
"${POETRY_BIN}" run python -u scripts/train_enhanced_model.py \
  --variant coord_attn \
  --epochs 60 \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --head-units 256 \
  --head-dropout 0.3 \
  --seed 42 \
  --run-name "enhanced_coord_attn_e60_s42" \
  2>&1 | tee "${LOG_DIR}/enhanced_coord_attn_${TIMESTAMP}.log"

# ─── Variant 3: Multi-Scale + CBAM ─────────────────────────────────────────
echo ""
echo "=========================================="
echo " Variant 3/4: Multi-Scale + CBAM"
echo "=========================================="
"${POETRY_BIN}" run python -u scripts/train_enhanced_model.py \
  --variant multi_scale \
  --epochs 60 \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --head-units 256 \
  --head-dropout 0.3 \
  --seed 42 \
  --run-name "enhanced_multi_scale_e60_s42" \
  2>&1 | tee "${LOG_DIR}/enhanced_multi_scale_${TIMESTAMP}.log"

# ─── Variant 4: Ensemble (3 heads) ──────────────────────────────────────────
echo ""
echo "=========================================="
echo " Variant 4/4: Ensemble (3 heads)"
echo "=========================================="
"${POETRY_BIN}" run python -u scripts/train_enhanced_model.py \
  --variant ensemble \
  --epochs 60 \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --head-units 256 \
  --head-dropout 0.3 \
  --num-heads 3 \
  --seed 42 \
  --run-name "enhanced_ensemble_3h_e60_s42" \
  2>&1 | tee "${LOG_DIR}/enhanced_ensemble_${TIMESTAMP}.log"

# ─── Compare Results ────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Results Summary"
echo "=========================================="

for variant in baseline coord_attn multi_scale ensemble; do
  RESULTS_DIR="${REPO_ROOT}/artifacts/training/enhanced_${variant}_e60_s42"
  RESULTS_FILE="${RESULTS_DIR}/results.json"
  if [[ -f "${RESULTS_FILE}" ]]; then
    echo ""
    echo "--- ${variant} ---"
    python3 -c "
import json
with open('${RESULTS_FILE}') as f:
    r = json.load(f)
hc = r.get('hard_case', {})
print(f'  Hard Case MAE: {hc.get(\"mae\", \"N/A\"):.2f}°C')
print(f'  Hard Case Max: {hc.get(\"max_error\", \"N/A\"):.2f}°C')
print(f'  Hard Case RMSE: {hc.get(\"rmse\", \"N/A\"):.2f}°C')
print(f'  Over 5°C: {hc.get(\"over_5c\", \"N/A\")}/{hc.get(\"count\", \"N/A\")}')
print(f'  Cold MAE: {hc.get(\"cold_mae\", \"N/A\"):.2f}°C')
print(f'  Low MAE: {hc.get(\"low_mae\", \"N/A\"):.2f}°C')
print(f'  Mid MAE: {hc.get(\"mid_mae\", \"N/A\"):.2f}°C')
print(f'  Hot MAE: {hc.get(\"hot_mae\", \"N/A\"):.2f}°C')
print(f'  Params: {r.get(\"total_params\", \"N/A\"):,}')
"
  else
    echo "--- ${variant} --- No results found"
  fi
done

echo ""
echo "=========================================="
echo " Training complete!"
echo " Logs: ${LOG_DIR}/"
echo "=========================================="
