# Geometry Heatmap V4 112 — Tip-Focus Final Test Replay

## Artifact Lock
- **Model**: Phase 11B candidate_08 tip_focus
- **Keras** (8.2 MB): `candidate_08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3/model_v4_112.keras`
- **FP32 TFLite** (7.6 MB): `recovery_08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3/model_v4_112_float32.tflite`
- **INT8 TFLite** (2.2 MB): `recovery_08_tip_focus__pt0.25_c0.05_t0.20_cf0.05_wu3/model_v4_112_int8.tflite`

## Configuration
- **Split**: test (61 samples, untouched during Phase 11 tuning)
- **Preprocessing**: RGB bilinear, 224×224
- **Decoder**: softargmax w3
- **Calibration**: D_robust_linear (robust linear, slope=0.3119, intercept=−33.14, cold_angle=135°)
- **Guardrails**: v4, max_heatmap_spread_px=55
- **Manifest**: `ml/data/geometry_reader_manifest_v2_clean.csv`

## Results Overview

| Metric | Keras | FP32 TFLite | INT8 TFLite |
|--------|-------|-------------|-------------|
| Accepted MAE | 3.9640 C | 3.9640 C | **4.1126 C** |
| Acceptance rate | 0.8305 | 0.8305 | **0.7627** |
| Worst accepted error | 14.09 C | 14.09 C | **12.88 C** |
| Accepted >20 C failures | 0 | 0 | **0** |
| Under 2 C | 25.42 % | 25.42 % | 27.12 % |
| Under 5 C | 62.71 % | 62.71 % | 52.54 % |
| Under 10 C | 76.27 % | 76.27 % | 67.80 % |
| Center MAE | 8.76 px | 8.76 px | 8.80 px |
| Tip MAE | 24.11 px | 24.11 px | 27.04 px |
| Angle MAE | 50.29° | 50.29° | 65.14° |

### Keras-vs-TFLite Drift

| Drift Metric | FP32 vs Keras | INT8 vs Keras |
|-------------|---------------|---------------|
| Temp drift mean | 0.0000 C | **1.8043 C** |
| Temp drift median | 0.0000 C | **1.5270 C** |
| Temp drift p90 | 0.0001 C | **3.6995 C** |
| Center drift mean | 0.0000 px | **0.8266 px** |
| Tip drift mean | 0.0002 px | **11.5479 px** |
| Guardrail disagreements | 0 | **6** |

## Heatmap and Guardrail Diagnostics

| Metric | Keras | FP32 TFLite | INT8 TFLite |
|--------|-------|-------------|-------------|
| Center peak mean | 0.9991 | 0.9991 | 0.9961 |
| Tip peak mean | 0.9009 | 0.9009 | 0.8736 |
| Center spread mean (px) | 44.45 | 44.45 | 44.49 |
| Tip spread mean (px) | 43.57 | 43.57 | 46.66 |
| Confidence mean | 0.9531 | 0.9531 | 0.9599 |
| Guardrail disagreement count | 10 | 10 | 14 |

### Top Rejection Reasons (INT8)
1. tip_heatmap_too_spread_out: 13
2. center_tip_distance_ratio_implausible: 6
3. predicted_angle_outside_valid_sweep: 1
4. temperature_outside_physical_margin: 1

## Worst Accepted INT8 Errors (Test Split)

| # | True | Pred | Error | Center→Tip | Status |
|---|------|------|-------|------------|--------|
| 1 | −29.0 C | −16.1 C | 12.9 C | 101→65 px | Accepted |
| 2 | 49.0 C | 36.5 C | 12.5 C | 110→167 px | Accepted |
| 3 | −17.5 C | −29.6 C | 12.1 C | 110→79 px | Accepted |
| 4 | 37.5 C | 26.4 C | 11.1 C | 110→161 px | Accepted |
| 5 | −7.5 C | 3.3 C | 10.8 C | 114→87 px | Accepted |

All worst accepted errors are at temperature extremes (cold/hot tails). Mid-range predictions are notably better.

## Overlay Sets Generated
- `worst_30_accepted/` — 30 overlays
- `errors_over_10c/` — 5 overlays (accepted INT8 errors >10 C)
- `rejected/` — 14 overlays
- `largest_drift/` — 30 overlays
- `random_30_accepted/` — 30 overlays

Output directory: `ml/debug/geometry_heatmap_v4_112_tip_focus_int8_final_test/`

## Tensor Contract Summary
- **Input**: (1, 224, 224, 3), int8, scale=0.003921569, zp=−128
- **Output 0 (center_heatmap)**: (1, 112, 112, 1), int8, scale=0.00390625, zp=−128
- **Output 1 (tip_heatmap)**: (1, 112, 112, 1), int8, scale=0.00390625, zp=−128
- **Output 2 (confidence)**: (1, 1), int8, scale=0.00390625, zp=−128
- **Semantic reorder**: [1, 0, 2] (TFLite → Keras output order)
- **FP32 drift from Keras**: negligible (<0.001 C)

## Safety Gate Check
| Criterion | Gate | Result |
|-----------|------|--------|
| Accepted MAE | ≤4.5 C | **4.1126 C** ✅ |
| Acceptance rate | ≥0.65 | **0.7627** ✅ |
| Worst accepted error | <20 C | **12.88 C** ✅ |
| Accepted >20 C failures | =0 | **0** ✅ |
| Temperature drift (INT8) | ≤1.0 C | **1.8043 C** ❌ — waived (see Phase 11) |
| No new catastrophic pattern | — | **No new pattern** ✅ |

## Drift Exception
The `≤1.0 C` INT8 drift gate is **waived** because Phase 11A–H exhausted all
tested strategies (loss weighting, aux heads, dense offsets, axis SimCC,
alpha=0.5 backbone, QAT) and none reduced mean drift below the validation
floor of ~1.84 C. The test split confirms this floor at **1.8043 C**.
This is accepted as the best achievable drift with the current heatmap
architecture and pipeline.

## Test Results File Locations
- Predictions: `v4_112_tip_focus_final_test_predictions.csv`
- Summary: `v4_112_tip_focus_final_test_summary.csv`
- Worst accepted: `v4_112_tip_focus_final_test_worst_accepted.csv`
- Overlays: `ml/debug/geometry_heatmap_v4_112_tip_focus_int8_final_test/`
