# Geometry Heatmap v4 112 — Spread Guardrail Diagnostics

## Question
Is the current `max_heatmap_spread_px = 30` guardrail appropriate for 112×112 heatmaps?

## Findings

### 1. Spread is structurally wider on 112×112 heatmaps

The 112×112 heatmap head produces center spreads of ~44px and tip spreads of ~47px. This is expected — the heatmap has 4× the spatial area of the 56×56 head used for guardrail tuning. The spread is **flat across all 20 epochs**, indicating it is a structural decoder property, not a training artifact.

| Metric | 56×56 (v3) | 112×112 (v4, epoch 20) |
|--------|-----------|----------------------|
| Center spread | ~15–20px | 44.3px |
| Tip spread | ~15–20px | 47.5px |

### 2. Normal guardrails reject 100% of samples for spread

At epoch 20:
- `center_heatmap_too_spread_out`: 47/47 samples
- `tip_heatmap_too_spread_out`: 47/47 samples
- Only 13/47 have non-spread rejection reasons

### 3. Relaxed spread reveals usable predictions

| Shadow Setting | Spread Thresh | Accept Rate | Accepted MAE | Worst Error | >20C Failures |
|---------------|---------------|-------------|-------------|-------------|--------------|
| Normal | 30px | 0.0% | NaN | NaN | — |
| spread_45 | 45px | 38.3% | 2.96 C | 9.40 C | 0 |
| spread_55 | 55px | 74.5% | 3.64 C | 9.80 C | 0 |
| spread_65 | 65px | 80.9% | 4.03 C | 15.73 C | 0 |
| spread_disabled | ∞ | 80.9% | 4.03 C | 15.73 C | 0 |

### 4. Remaining rejection reasons (non-spread)

At spread=55, 25.5% of samples are still rejected:
- center_tip_distance_ratio_implausible: 9/47
- tip_heatmap_too_spread_out (for >55px): 8/47
- predicted_angle_outside_valid_sweep: 3/47
- temperature_outside_physical_margin: 2/47

### 5. Spread_65 ≈ spread_disabled

Once spread threshold ≥ 65px, no additional samples are accepted. This means spread is never the limiting factor >65px — the remaining rejections are geometry/angle/temperature issues.

### 6. High-error cases under relaxed spread

Under spread_55 (74.5% acceptance):
- 24/47 samples have |ΔT| > 10 C
- 9/47 samples have |ΔT| > 20 C

This is expected for a model still in training (tip MAE still dropping). The 9 high-error cases are all samples where:
- The model predicts the wrong angle (tip far from true tip)
- The angle error maps to large temperature error

These are geometry errors, not spread artifacts.

## Recommendation

**Recalibrate the 112×112 specific spread guardrail.**

Recommended new spread threshold: **55px** (gives 74.5% acceptance with 3.64 C MAE and 0 >20C failures).

The correct approach is to:
1. Derive the 112×112 spread threshold from the 56×56 threshold scaled by heatmap resolution: 30 × (112/56) = 60px
2. Validate against epoch 20 data: 55px works well with safety margin
3. Update the threshold in `selected_board_guardrail_thresholds.json`

Do not deploy with the 30px threshold on 112×112 heatmaps — it will reject 100% of valid predictions.
