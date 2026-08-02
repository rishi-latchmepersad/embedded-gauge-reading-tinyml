# QAT Encoder Ellipse Model — 2026-07-23

Date: 2026-07-23
Status: experimental
Scope: ellipse detection for gauge reading pipeline

## Summary

Built a QAT-safe ellipse detector using Conv+BN+ReLU architecture that
produces meaningful int8 outputs (unlike the bias-only CNN variants that
collapse to constants).

## Architecture

- Encoder: 5 stages of Conv-BN-ReLU, stride-2 each (224→112→56→28→14→7)
- Width multiplier: 1.5 (48→72→96→144→192 channels)
- Head: GAP + Dense(128, relu) + Dense(5, sigmoid) for [cx, cy, rx, ry, conf]
- Input: 224×224×1 grayscale
- Peak activation: 112×112×48 = 602 KB int8 (under 1 MB SRAM)
- Model size: 1.2 MB int8 TFLite

## Results

| Metric | Value |
|--------|-------|
| Test MAE (FP32) | 0.0093 |
| Test MAE (int8) | Meaningful (varies by input) |
| Center error on test_3 | 3.5-9.7px (mean ~6.5px) |
| int8 output varies? | YES (not collapsed) |
| Radius prediction | Still somewhat fixed (0.2461 for all) |

## Key improvement over v9/v10/v11

The bias-only CNN (v9/v10/v11) collapses to a constant output after int8
quantization. The QAT encoder with Conv+BN+ReLU produces meaningful varying
outputs. This is a fundamental architectural requirement for int8 deployment.

## Artifacts

- `artifacts/gauge_ellipse_qat_encoder_v1/`
- `scripts/train_gauge_ellipse_qat_encoder.py`

## Next steps

- Improve radius prediction (currently fixed at 0.2461 for all test_3 images)
- Evaluate on full 97-image test set with new ellipse crops
- Consider joint model that predicts ellipse + center/tip in one pass
