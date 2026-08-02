# Ellipse + center/tip pipeline lessons — 2026-07-23

Date: 2026-07-23
Status: current
Scope: gauge reading pipeline (ellipse → center/tip)

## What was built today

1. **QAT-safe ellipse detector** with linear radius head — `gauge_ellipse_qat_linear_v1`
   - Conv+BN+ReLU encoder, multi-head output (center sigmoid, radius LINEAR, confidence sigmoid)
   - 224×224 input, 1.2MB int8, 602KB peak activation
   - Radius varies correctly on test_3 (0.194-0.200, GT: 0.195)

2. **v9 center/tip UNet** — `gauge_center_tip_littlegood_v9`
   - 24/36/56/96 channel U-Net, 160×160 input, 80×80 heatmap output
   - 379KB int8, balanced focal loss [4.0 center, 6.0 tip], y_true^1.5
   - 86.6% center ≤8px, 75.3% tip ≤8px on 97-image test set
   - 100% on test_3 (board captures) with linear radius ellipse

## Critical finding: training/eval pipeline mismatch

The v9 model was trained on **pre-computed 160×160 crops** from
`data/gauge_center_tip_v1_160_gray/`. These crops were generated with
a specific two-stage pipeline (1.18× source crop from 640px → resize to
160px → mask with 1.35× scale).

When evaluating on raw 640px test images (test_1), the crop pipeline
generates **different crops** even with the same parameters, because:
- The pre-computed crops were clipped to 160px boundaries
- The evaluation crops are clipped to 640px boundaries
- The clipping changes the crop origin and normalization

Result: v9 gets 86.6% center on the pre-computed test set but only 3.3%
on test_1 (same gauge types, same GT ellipses).

## Fix needed

Train the center/tip model using the **exact same crop pipeline** as
evaluation — generating crops on-the-fly from 640px images with the
two-stage process. The `prepare_full_pipeline_data.py` script was created
for this but needs the correct ellipse source for training.

## Architecture lessons (confirmed)

| Lesson | Status |
|--------|--------|
| Conv+BN+ReLU prevents int8 collapse | ✅ Validated |
| Linear output for radius preserves variation | ✅ Validated |
| Balanced focal loss [4:6] beats [1.5:10] | ✅ Validated |
| Radius regression head hurts accuracy | ✅ Validated |
| Crop pipeline must match between training and eval | ✅ NEW |

## Key scripts created

- `scripts/train_gauge_ellipse_qat_linear.py` — linear radius head ellipse model
- `scripts/train_gauge_center_tip_v8_improved.py` — balanced focal loss U-Net
- `scripts/prepare_full_pipeline_data.py` — pipeline-matched training data
- `scripts/eval_pipeline_adaptive.py` — evaluation with adaptive crop scales
- `scripts/eval_pipeline_all_tests.py` — evaluation on test_1/2/3
