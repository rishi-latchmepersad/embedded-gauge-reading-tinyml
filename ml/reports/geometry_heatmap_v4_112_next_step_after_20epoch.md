# Geometry Heatmap v4 112 — Next Step After 20-Epoch Controlled Smoke

## Summary

Resume from 10-epoch checkpoint succeeded. Trained 10 additional epochs (20 total) with shadow spread-guard diagnostics.

## Final Metrics

| Metric | Final (Epoch 20) | Best | Trend |
|--------|-----------------|------|-------|
| Center MAE | 8.37 px | 8.37 px | Plateau |
| Tip MAE | 28.56 px | 28.56 px | Still dropping |
| Angle MAE | 13.17° | 13.17° | Plateau |
| Center spread | 44.3 px | — | Flat |
| Tip spread | 47.5 px | — | Flat |
| Normal acceptance | 0% | — | All spread-rejected |

## Shadow Spread-Guard Results (Epoch 20)

| Setting | Accept% | MAE | Worst | >20C |
|---------|--------|-----|-------|------|
| Normal (30px) | 0% | — | — | — |
| spread_45 (45px) | 38.3% | 2.96 C | 9.40 C | 0 |
| spread_55 (55px) | 74.5% | 3.64 C | 9.80 C | 0 |
| spread_65 (65px) | 80.9% | 4.03 C | 15.73 C | 0 |
| spread_disabled | 80.9% | 4.03 C | 15.73 C | 0 |

## Decision: **D. Recalibrate 112-specific spread guardrail**

The 30px spread threshold was tuned for 56×56 heatmaps and is structurally incompatible with 112×112 heatmaps. Geometry is good (tip MAE 28.6px, angle MAE 13.2°). Under spread=55px acceptance jumps to 74.5% with 3.64 C MAE and 0 >20C failures.

**Recommended new threshold:** 55px (scale 30px × 112/56 = 60px, rounded down for safety).

## Files Changed

- `ml/scripts/train_geometry_heatmap_v4_112_quant_native.py`:
  - Added `--resume-from` argument for checkpoint continuation
  - Added `GeometryDecodedPrediction` import
  - Added `from dataclasses import dataclass, replace`
  - Modified `_build_model_from_mode` to support resume
  - Modified `ReplayMetricCallback._evaluate_model` to return `(rows, decoded_predictions)`
  - Added `ReplayMetricCallback._build_shadow_rows` and `_shadow_summaries` methods
  - Modified `ReplayMetricCallback.on_epoch_end` to compute and log shadow metrics
- `tmp/generate_v4_112_20epoch_shadow_overlays.py`: New overlay script with shadow guardrail visualization

## Tests Run

`pytest tests/test_geometry_prediction_guardrails.py tests/test_geometry_heatmap_tflite_utils.py tests/test_geometry_heatmap_qat_utils.py tests/test_geometry_heatmap_v3_quant_native_utils.py tests/test_geometry_heatmap_v4_112_utils.py -q`

**Result: 29/29 passed**

## Artifacts

- `/tmp/geomq_v4_112_smoke_20epoch/` — full run outputs (model, history, summaries)
- `ml/debug/geometry_heatmap_v4_112_controlled_smoke_20epoch/` — 140 overlay images
- `ml/reports/geometry_heatmap_v4_112_controlled_smoke_20epoch.md`
- `ml/reports/geometry_heatmap_v4_112_spread_guard_diagnostics.md`
- `ml/reports/geometry_heatmap_v4_112_controlled_smoke_20epoch_decision.md`
