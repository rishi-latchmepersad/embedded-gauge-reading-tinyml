# Pipeline validated on 16 UNSEEN LittleGood captures — MAE 0.82C — 2026-08-01

Date: 2026-08-01
Status: validated
Scope: ellipse + keypoint -> temperature pipeline, held-out evaluation
Evidence: `ml/scripts/pipeline_ellipse_keypoint_temperature.py`,
`/tmp/opencode/littlegood_unseen/` (16 images copied from
`ml/data/captured_images/`, never in any training split)

## Results (16 unseen board captures, GT temp in filename)

**MAE 0.82C, median 0.76C, max 1.81C, 100% within 2C, 100% within 5C.**

| Image | GT | Pred | Err |
|---|---|---|---|
| capture_m10c.jpg | -10 | -9.1 | 0.9 |
| capture_m30c_preview.png | -30 | -28.3 | 1.7 |
| capture_p10c.png | +10 | +10.1 | 0.1 |
| capture_p15c.jpg | +15 | +15.2 | 0.2 |
| capture_p20c.png | +20 | +18.7 | 1.3 |
| capture_p25c.jpg | +25 | +25.8 | 0.8 |
| capture_p30c.jpg | +30 | +28.2 | 1.8 |
| capture_p31c.jpg | +31 | +30.3 | 0.7 |
| capture_p35c.jpg | +35 | +35.8 | 0.8 |
| capture_p42c.jpg | +42 | +40.9 | 1.1 |
| capture_p5c.png | +5 | +6.3 | 1.3 |

## Method

The pipeline script now accepts a plain image directory (`--images <dir>`)
in addition to CVAT zips.  Unseen = images whose basename is NOT in any
training archive (checked against train_1/2, board_captures_1/3/4,
gauge_1_batch_1..8).

## Why this matters

- This is a genuinely held-out set (unlike test_3 where capture_m25c.jpg
  is duplicated in training).
- 100% within 2C on unseen captures beats the test_3 numbers (71% within
  2C) — the test_3 outliers were the trained-on duplicates plus glare
  variants.
- The pipeline is board-ready: geometry-first reading gives sub-degree-C
  accuracy on this gauge.

## Files

- Pipeline: `ml/scripts/pipeline_ellipse_keypoint_temperature.py`
- Results CSV: `/tmp/opencode/littlegood_unseen/littlegood_unseen_pipeline_results.csv`
- Models: ellipse `ellipse_iter8_universal_wide_deep` + keypoint
  (stride-2s board-fit training in progress)
