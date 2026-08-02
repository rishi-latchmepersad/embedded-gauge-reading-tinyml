# Balanced focal heatmap loss weights beat extreme tip-weighting — 2026-07-22

Date: 2026-07-22
Status: validated
Scope: center/tip keypoint heatmap models (gauge reading pipeline)
Evidence: `ml/artifacts/gauge_center_tip_littlegood_v{7,8,9}/report.json`
Decision: Always start with [4.0, 6.0] center:tip channel weights and y_true^1.5
         focal exponent. Never use [1.5, 10.0] or higher focal powers ≥2.0.

## The rule

For dual-heatmap models predicting center and tip keypoints:
- **Channel weights**: center=4.0, tip=6.0 (tip gets modestly more weight
  because it is sparser and harder, but not 10x like the old recipe)
- **Focal exponent**: 1.5 (mild amplification of peak regions; 2.0 over-sharpens)
- **Baseline weight**: 28 (provides stable background supervision; 48 is too
  aggressive and 24 from v7 is too weak for the deeper architecture)

## Formula

```python
weights = 1.0 + 28.0 * (y_true ** 1.5) * channel_weight
loss = tf.reduce_mean(weights * tf.square(y_pred - y_true))
```

## Evidence

| Channel weights | Focal exponent | Center ≤8px | Tip ≤8px | Verdict |
|----------------|---------------|------------|----------|---------|
| [1.0, 2.0] (v7) | 1.0 (linear) | 84.5% | 47.4% | Good center, terrible tip |
| [1.5, 10.0] (v8) | 2.0 | 80.4% | 62.9% | Better tip, worse center |
| **[4.0, 6.0] (v9)** | **1.5** | **86.6%** | **75.3%** | **Best on both metrics** |

Same architecture (deeper U-Net, 379 KB) was used for v8 and v9; only loss
weights changed.

## Why it works

- The center peak is a large, well-defined Gaussian around the needle pivot.
  Giving it 4x weight ensures the model doesn't ignore it in favor of tip.
- The tip peak is sparser but needs a modest boost (6x) to overcome class
  imbalance with the background.
- y_true^1.5 amplifies the Gaussian peak region more than the tail, focusing
  supervision on sub-pixel peak placement without making the loss overly
  peaky (y_true^2 collapses it to nearly the argmax pixel only).
- The 28× baseline weight provides enough background contrast without driving
  the model to predict all zeros.

## Related anti-patterns

- **Radius regression head**: Adding a scalar radius output to complement
  direction vectors consistently hurts accuracy (see wide_weighted, wide_hybrid,
  direction_radius, line_radius experiments). The compound errors from
  direction × radius are worse than direct endpoint heatmaps.
- **Extreme crop scales (>1.35×)**: The "wide" experiments used 1.60× runtime
  crop, making the gauge smaller in the input. All wide models underperform the
  1.35× crop used by v7 and v9.
