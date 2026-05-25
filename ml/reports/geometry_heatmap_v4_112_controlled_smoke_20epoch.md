# Geometry Heatmap v4 112 — Controlled Smoke 20 Epoch Results

Resumed from 10-epoch checkpoint `/tmp/geomq_v4_112_smoke_10epoch/model_v4_112_frozen_best.keras`.
Trained 10 additional epochs (epochs 11–20) with the same frozen-backbone config: batch-size 4, LR 3e-6, sigma_pixels 2.5.

## Epoch-by-Epoch Table

| Ep | Center MAE | Tip MAE | Angle° | C Spread | T Spread | Accept% |
|----|-----------|--------|--------|---------|---------|---------|
| 11 | 9.47 | 51.26 | 41.69 | 43.65 | 50.55 | 0% |
| 12 | 9.12 | 46.64 | 30.93 | 43.76 | 50.81 | 0% |
| 13 | 8.90 | 42.44 | 24.47 | 43.88 | 50.71 | 0% |
| 14 | 8.68 | 38.42 | 18.44 | 43.99 | 50.07 | 0% |
| 15 | 8.54 | 34.77 | 15.65 | 44.07 | 49.21 | 0% |
| 16 | 8.50 | 33.21 | 15.70 | 44.12 | 48.65 | 0% |
| 17 | 8.44 | 31.58 | 14.01 | 44.17 | 48.24 | 0% |
| 18 | 8.41 | 30.40 | 13.80 | 44.21 | 47.98 | 0% |
| 19 | 8.38 | 29.17 | 13.38 | 44.25 | 47.52 | 0% |
| 20 | 8.37 | 28.56 | 13.17 | 44.28 | 47.45 | 0% |

## Trends

### Center MAE (px)
- **Final:** 8.37
- **Best:** 8.37 (epoch 20) — plateauing
- Across epochs 11–20: 9.47 → 8.37 (−11.6%)
- From epoch 3 baseline (8.41): stable

### Tip MAE (px)
- **Final:** 28.56
- **Best:** 28.56 (epoch 20) — still trending down
- Across epochs 11–20: 51.26 → 28.56 (−44.3%)
- From epoch 3 baseline (57.07): −49.9%
- **Not converged** — still dropping ~2px/epoch

### Angle MAE (degrees)
- **Final:** 13.17
- **Best:** 13.17 (epoch 20) — plateauing
- Across epochs 11–20: 41.69 → 13.17 (−68.4%)
- From epoch 3 baseline (48.82): −73.0%

### Center Heatmap Spread (px)
- **Final:** 44.28
- **Trend:** 43.65 → 44.28 — essentially flat (slight +1.4%)
- **Not shrinking** — stable property of 112x112 decoder

### Tip Heatmap Spread (px)
- **Final:** 47.45
- **Trend:** 50.55 → 47.45 (−6.1%)
- **Minimal change** — still far above 30px guardrail

### Normal Acceptance Rate
- **All epochs:** 0%
- **Top rejection reasons (epoch 20):** center_heatmap_too_spread_out (47/47), tip_heatmap_too_spread_out (47/47), center_tip_distance_ratio_implausible (9/47), predicted_angle_outside_valid_sweep (3/47), temperature_outside_physical_margin (2/47)

## Shadow Spread Guard Acceptance Trends

| Ep | sp45 Acc | sp45 MAE | sp45 Wrst | sp55 Acc | sp55 MAE | sp55 Wrst | sp65 Acc | sp65 MAE | sp65 Wrst |
|----|---------|---------|----------|---------|---------|----------|---------|---------|----------|
| 11 |  9% | 2.99 |  4.61 | 43% | 6.02 | 12.96 | 43% | 6.02 | 12.96 |
| 12 | 11% | 3.64 |  7.00 | 51% | 5.41 | 14.56 | 53% | 6.03 | 20.93 |
| 13 | 13% | 2.86 |  6.08 | 53% | 4.83 | 13.63 | 55% | 5.43 | 20.38 |
| 14 | 21% | 2.38 |  5.47 | 57% | 4.18 | 11.88 | 60% | 4.74 | 19.66 |
| 15 | 26% | 2.89 |  9.08 | 62% | 3.84 | 10.28 | 66% | 4.46 | 19.18 |
| 16 | 34% | 3.30 |  9.18 | 66% | 4.25 | 12.79 | 72% | 4.73 | 18.31 |
| 17 | 34% | 3.18 |  9.37 | 70% | 3.84 | 11.13 | 77% | 4.30 | 17.63 |
| 18 | 38% | 2.99 |  9.37 | 72% | 3.69 | 10.63 | 79% | 4.13 | 17.08 |
| 19 | 38% | 2.99 |  9.35 | 74% | 3.71 | 10.11 | 81% | 4.13 | 16.28 |
| 20 | 38% | 2.96 |  9.40 | 74% | 3.64 |  9.80 | 81% | 4.03 | 15.73 |

Spread_65 and spread_disabled produce identical results — once spread ≥ 65px, it never rejects.

## Overlays
140 overlays saved to `ml/debug/geometry_heatmap_v4_112_controlled_smoke_20epoch/`:
- 30 lowest tip-error cases
- 30 highest tip-error cases
- 47 shadow-accepted-but-normal-rejected cases
- 24 cases with temperature error >10C under spread_55
- 9 cases with temperature error >20C under spread_55

## Output Files
- `/tmp/geomq_v4_112_smoke_20epoch/model_v4_112_frozen_best.keras` — best checkpoint
- `/tmp/geomq_v4_112_smoke_20epoch/history.csv` — per-epoch training log
- `/tmp/geomq_v4_112_smoke_20epoch/epoch_summaries.json` — full epoch summaries with shadow metrics
- `/tmp/geomq_v4_112_smoke_20epoch/val_predictions.csv` — per-sample predictions
