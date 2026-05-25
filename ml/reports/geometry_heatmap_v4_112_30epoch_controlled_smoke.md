# Geometry Heatmap v4 112 — Controlled 30-Epoch Smoke Results

Trained 10 additional epochs (resumed from 20-epoch checkpoint) with V4 guardrails (spread=55), frozen-backbone config: batch-size 4, LR 3e-6→1e-6, sigma_pixels 2.5.

## Epoch-by-Epoch Table

| Ep | Center MAE | Tip MAE | Angle° | C Spread | T Spread | Accept% | MAE C | Worst C | >20C |
|----|-----------|--------|--------|---------|---------|---------|-------|---------|------|
|  1 | 9.10 | 46.60 | 29.17 | 43.77 | 51.15 | 51.1% | 5.25 | 14.52 | 0 |
|  2 | 8.89 | 42.56 | 24.20 | 43.88 | 50.83 | 55.3% | 5.09 | 13.59 | 0 |
|  3 | 8.68 | 38.61 | 19.00 | 43.99 | 50.03 | 57.4% | 4.34 | 12.34 | 0 |
|  4 | 8.55 | 35.01 | 15.58 | 44.09 | 49.39 | 61.7% | 3.88 | 10.64 | 0 |
|  5 | 8.46 | 32.07 | 14.56 | 44.17 | 48.48 | 66.0% | 3.88 | 11.24 | 0 |
|  6 | 8.43 | 30.82 | 14.33 | 44.22 | 47.99 | 70.2% | 3.90 | 11.02 | 0 |
|  7 | 8.40 | 29.65 | 13.78 | 44.26 | 47.70 | 74.5% | 3.82 | 10.12 | 0 |
|  8 | 8.37 | 28.40 | 13.07 | 44.30 | 47.25 | 74.5% | 3.66 | 9.65 | 0 |
|  9 | 8.35 | 27.26 | 12.60 | 44.33 | 46.66 | 76.6% | 3.49 | 9.49 | 0 |
| 10 | 8.34 | 26.71 | 12.22 | 44.35 | 46.61 | 78.7% | 3.39 | 9.52 | 0 |

## Trends (within this 10-epoch run)

### Center MAE (px)
- **Final:** 8.34 (−8.4% from 9.10 in epoch 1)
- **Trend:** Monotonically decreasing, plateauing at ~8.3px floor
- **224→112 downscale quantization floor:** ~8.3px (limited by 2x spatial precision loss)

### Tip MAE (px)
- **Final:** 26.71 (−42.7% from 46.60 in epoch 1)
- **Trend:** Still dropping ~2-3px/epoch — **not converged**

### Angle MAE (degrees)
- **Final:** 12.22° (−58.1% from 29.17° in epoch 1)
- **Trend:** Still dropping ~1°/epoch — **not converged**

### Spread (px)
- Center: 43.77 → 44.35 (+1.3%) — flat, structural property
- Tip: 51.15 → 46.61 (−8.9%) — slowly decreasing, still above 30px guardrail

### Acceptance Rate
- Epoch 1: 51.1% → Epoch 10: 78.7%
- **Monotonically increasing** as model improves

### Accepted Temperature MAE
- Epoch 1: 5.25 C → Epoch 10: 3.39 C (−35.4%)
- Best: 3.39 C (epoch 10)

### Cumulative 2/5/10°C Thresholds (epoch 10)
- Under 2°C: 27.7% (13/47)
- Under 5°C: 61.7% (29/47)
- Under 10°C: 78.7% (37/47)

## V4 Guardrail Pass Gate (Epoch 10)

| Criterion | Required | Actual | Status |
|-----------|----------|--------|--------|
| Accepted MAE | <= 4.5 C | 3.39 C | ✓ |
| Acceptance | >= 65% | 78.7% | ✓ |
| Worst accepted error | < 20 C | 9.52 C | ✓ |
| >20C failures | = 0 | 0 | ✓ |

**RESULT: PASSED**

## Shadow Spread Guard Comparison (Epoch 10)

| Metric | Spread=45 | Spread=55 | Spread=65 | Disabled |
|--------|-----------|-----------|-----------|----------|
| Acceptance | 42.6% | 78.7% | 85.1% | 85.1% |
| MAE (C) | 2.95 | 3.39 | 3.75 | 3.75 |
| Worst (C) | 9.52 | 9.52 | 14.34 | 14.34 |
| >20C failures | 0 | 0 | 0 | 0 |

- **Spread=55 is the sweet spot:** low MAE, low worst error, 78.7% acceptance, 0 failures.
- Spread=45 cuts acceptance in half (42.6%) for minimal MAE gain (2.95 vs 3.39).
- Spread=65 adds only 6.4% more acceptance but doubles worst error (14.34 vs 9.52).

## Comparison to 20-Epoch Smoke

The 30-epoch smoke was resumed from the same checkpoint as the 20-epoch smoke (due to restore_best_weights reversion in the 20-epoch run). Both runs produced identical per-epoch metrics. The key improvement over the 10-epoch standalone run:

| Metric | 10-Epoch Stdln | 30-Epoch (this run) | Change |
|--------|---------------|-------------------|--------|
| Center MAE | ~9.2 px | 8.34 px | −9.3% |
| Tip MAE | ~32.4 px | 26.71 px | −17.6% |
| Angle MAE | ~16.4° | 12.22° | −25.5% |
| Acceptance | 0% (V2 guard) | 78.7% (V4 guard) | — |
| Accepted MAE | — | 3.39 C | — |

## Known Issues

### temperature_delta_mean is NaN (reference vs. current comparison)
The `val_v4_replay_temperature_delta_mean` metric is NaN across all epochs because the reference model (built from V3 source weights) and the current model have zero commonly-accepted validation samples. The source model produces very different predictions on 112×112 heatmaps, so no sample is accepted by both models' V4 guardrails. This causes `val_v4_replay_score` to also be NaN, which prevents the ReplayMetricCallback from identifying a "best" checkpoint (the `model_v4_112_frozen_best.keras` has epoch-1 weights).

**Impact:** The final `model_v4_112.keras` has correct epoch-10 weights. The `model_v4_112_frozen_best.keras` is stale but unused when loading the final model. To fix for future runs, initialize the reference model from the resumed checkpoint rather than from the source model.

## Overlays
Generated overlays saved to `ml/debug/geometry_heatmap_v4_112_30epoch_controlled_smoke/`:
- 30 lowest tip-error cases
- 30 highest tip-error cases
- Shadow-accepted-but-normal-rejected cases
- Cases with temperature error >10C / >20C under spread_55

## Output Files
- `ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/model_v4_112.keras` — final weights (epoch 10)
- `ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/frozen_training_log.csv` — per-epoch log
- `ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/epoch_summaries.json` — full epoch summaries with shadow metrics
- `ml/artifacts/training/geometry_heatmap_v4_112_quant_native_30epoch_smoke/val_predictions.csv` — per-sample predictions
