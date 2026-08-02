# Crop pipeline consistency is critical — 2026-07-23 (part 2)

Date: 2026-07-23
Status: validated
Scope: two-stage ellipse → center/tip pipeline
Evidence: eval results on test_1 (914 images), test_3 (11 images)

## The finding

The center/tip model is **extremely sensitive to crop consistency** between
training and evaluation. When the ellipse model used for crop generation
differs between training and evaluation, the center/tip accuracy collapses.

## Evidence

| Pipeline | test_3 center ≤8px | test_1 center ≤8px |
|----------|-------------------|-------------------|
| GT ellipse + v9 model | **100%** | 25.8% |
| Predicted ellipse + retrained model | 9.1% | 26.6% |

The v9 model was trained on crops from the v11 ellipse model (fixed radius
0.2539). When evaluated with the linear radius ellipse (varying 0.194-0.200),
the crops are shifted/scaled differently, and the model fails.

## Why this happens

The center/tip labels are normalized coordinates [0,1] within the crop.
When the crop changes (different ellipse center/radius), the normalized
positions of the center/tip change. The model learns the mapping from
crop+mask → normalized coordinates for one specific crop distribution,
and doesn't generalize to different crops.

## Implication for production

For the two-stage pipeline to work, the ellipse model used for crop
generation during training must be EXACTLY the same model used during
inference. Any change to the ellipse model requires retraining the
center/tip model.

## Recommended approach

1. **For the current pipeline**: Use v9 center/tip + the ellipse model it
   was trained with (v11 or equivalent). Accept that test_1 performance
   is limited by training data diversity.

2. **For better generalization**: Train a unified single-pass model that
   predicts ellipse + center/tip in one forward pass. This eliminates the
   two-stage error propagation entirely.

3. **If keeping two-stage**: Retrain the center/tip model with the ACTUAL
   production ellipse model (linear radius) and include ALL training images
   (generic + littlegood + test_1) with that ellipse model's predictions
   for crop generation.

## Current best models

| Model | File | Size | Best metric |
|-------|------|------|-------------|
| Ellipse (linear radius) | `artifacts/gauge_ellipse_qat_linear_v1/` | 1.2MB int8 | 100% center ≤8px test_3 |
| Center/tip (v9) | `artifacts/gauge_center_tip_littlegood_v9/` | 379KB int8 | 86.6% center, 75.3% tip (97-img test) |

## test_1 analysis

- 25.8% center ≤8px with GT ellipse + v9 model
- 4 images filtered out of precomputed test set (extreme elongated ellipses)
- Worst cases have very elongated ellipses (rx:ry ratio up to 8:1)
- Model works well on round gauges (≤8px for many images) but fails on
  elongated or unusual gauge shapes

## Recommendation for user

For gauges similar to the training distribution (round faces, similar
camera distance), the v9 model is production-ready. For diverse gauge
types, a unified single-pass model or significantly more training data
with the production ellipse model is needed.
