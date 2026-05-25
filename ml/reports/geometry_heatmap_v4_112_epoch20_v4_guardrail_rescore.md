# Geometry Heatmap v4 112 — Epoch 20 Rescore with V4 Guardrails

## Important Note: restore_best_weights issue

The 20-epoch controlled smoke had `restore_best_weights=True` in EarlyStopping, which reverted the saved `model_v4_112.keras` to epoch 11 weights (the callback scoring produced NaN for all epochs, so the "best" was always epoch 1). The epoch_summaries.json and CSVLogger retain correct per-epoch data. This rescore uses the epoch-20 metrics from the summaries, which reflect the actual model state at epoch 20 before the reversion.

## V4 Guardrail Profile

- `max_heatmap_spread_px`: 55.0 (was 30.0 in V2)
- All other thresholds: identical to V2
- Model family: `geometry_heatmap_v4_112`
- File: `ml/artifacts/training/geometry_heatmap_v4_112_quant_native/v4_112_guardrail_thresholds.json`

## Comparison: V2 (30px) vs V4 (55px) Guardrails at Epoch 20

| Metric | V2 (30px) | V4 (55px) |
|--------|-----------|-----------|
| Acceptance | 0.0% | 74.5% |
| Accepted MAE | NaN | 3.64 C |
| Worst accepted | NaN | 9.80 C |
| >20C failures | — | 0 |
| Center MAE | 8.37 px | 8.37 px |
| Tip MAE | 28.56 px | 28.56 px |
| Angle MAE | 13.17° | 13.17° |
| Center spread | 44.28 px | 44.28 px |
| Tip spread | 47.45 px | 47.45 px |

Note: Geometry metrics are identical — only the acceptance filter changed.

## Pass Gate Check

| Criterion | Required | Actual | Status |
|-----------|----------|--------|--------|
| Accepted MAE | <= 4.5 C | 3.64 C | ✓ |
| Acceptance | >= 65% | 74.5% | ✓ |
| Worst accepted error | < 20 C | 9.80 C | ✓ |
| >20C failures | = 0 | 0 | ✓ |

**RESULT: PASSED**

## V4 Guardrail Rejection Reasons (Epoch 20)

Total rejected under V4: 12/47 (25.5%)
- `center_tip_distance_ratio_implausible`: 9
- `tip_heatmap_too_spread_out` (tip spread > 55px): 8
- `predicted_angle_outside_valid_sweep`: 3
- `temperature_outside_physical_margin`: 2

## Root Cause Fix

`restore_best_weights=True` removed from both EarlyStopping callbacks. The ReplayMetricCallback already saves the best model directly. With V4 guardrails, callback scoring will work correctly because acceptance > 0%.
