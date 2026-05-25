# Geometry Heatmap v4 112 Controlled Smoke Report (10 epochs)

## Source
- Directory: `/tmp/geomq_v4_112_smoke_10epoch`
- Command: `--frozen-epochs 10 --unfrozen-epochs 0 --batch-size 4 --frozen-learning-rate 3e-6 --sigma-pixels 2.5`
- Logging: Fixed (history.csv has 10 data rows + epoch_summaries.json)

## Completion
- **10/10 epochs completed**
- No NaNs/Infs
- No process hangs after switching to `model(..., training=False)` instead of `model.predict()`

## Per-Epoch Metrics

| Epoch | Total Loss | Center MAE | Tip MAE | Angle MAE | Center Spread | Tip Spread | Center Peak | Tip Peak | Accept Rate |
|-------|-----------|-----------|--------|----------|-------------|-----------|-----------|---------|------------|
| 1 | 820.86 | 9.94 | 57.07 | 48.82 | 43.60 | 48.92 | 0.984 | 0.946 | 0.0 |
| 2 | 457.40 | 9.48 | 51.23 | 41.54 | 43.65 | 50.38 | 0.990 | 0.933 | 0.0 |
| 3 | 350.39 | 9.13 | 46.42 | 29.89 | 43.76 | 51.01 | 0.993 | 0.921 | 0.0 |
| 4 | 228.64 | 8.84 | 41.92 | 21.95 | 43.87 | 50.67 | 0.995 | 0.912 | 0.0 |
| 5 | 332.34 | 8.67 | 37.81 | 18.11 | 43.97 | 49.97 | 0.997 | 0.904 | 0.0 |
| 6 | 215.98 | 8.61 | 35.92 | 17.10 | 44.02 | 49.46 | 0.997 | 0.902 | 0.0 |
| 7 | 240.67 | 8.52 | 34.26 | 15.45 | 44.06 | 49.22 | 0.997 | 0.897 | 0.0 |
| 8 | 82.72 | 8.47 | 32.74 | 14.96 | 44.11 | 48.81 | 0.998 | 0.895 | 0.0 |
| 9 | 178.97 | 8.44 | 31.29 | 14.28 | 44.16 | 48.35 | 0.998 | 0.894 | 0.0 |
| 10 | 156.04 | 8.41 | 30.52 | 13.96 | 44.18 | 48.20 | 0.998 | 0.892 | 0.0 |

## Key Trends

**Improving metrics:**
- Tip MAE: 57.07 → 30.52 (**↓ 46%**)
- Angle MAE: 48.82 → 13.96 (**↓ 71%**)
- Total loss: 820.86 → 156.04 (**↓ 81%**)
- Center MAE: 9.94 → 8.41 (stable, ↓ 15%)
- Distance ratio implausible rejections: 37 → 10 (**↓ 73%**)

**Flat/not improving:**
- Center spread: 43.60 → 44.18 (flat, no change)
- Tip spread: 48.92 → 48.20 (flat, no change)
- Acceptance rate: 0.0 at all epochs

**Rejection reasons (epoch 10):**
- center_heatmap_too_spread_out: 47/47
- tip_heatmap_too_spread_out: 47/47
- center_tip_distance_ratio_implausible: 10/47
- predicted_angle_outside_valid_sweep: 3/47
- temperature_outside_physical_margin: 3/47

## Summary
The model is clearly learning geometry (tip MAE halved, angle MAE dropped 71%) but heatmap spatial spread is not shrinking. The softargmax decoder is finding the right location but the heatmap distribution remains diffuse (~44px center, ~48px tip), blocking guardrail acceptance.
