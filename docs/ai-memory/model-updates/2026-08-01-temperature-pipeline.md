# Temperature pipeline works end-to-end — MAE 1.52C on test_3 — 2026-08-01

Date: 2026-08-01
Status: current
Scope: ellipse + keypoint -> temperature pipeline (LittleGood board captures)
Evidence: `ml/scripts/pipeline_ellipse_keypoint_temperature.py`,
`ml/data/labelled/test_3_pipeline_results.csv`

## Pipeline

1. Ellipse detector (`ellipse_iter8_universal_wide_deep`, 384x384) -> gauge
   face (cx, cy, rx, ry)
2. 1.35x square crop around the ellipse -> 224x224
3. Keypoint model (224x224 -> 112x112 stride-2 heatmaps) -> center + tip
4. tip-center vector -> angle (atan2, image coords) -> temperature via
   LittleGood calibration: min_deg=135, sweep_deg=270, -30C..+50C

The LittleGood calibration was recovered from git history: the current
`gauge_calibration_parameters.toml` was OVERWRITTEN by a firmware-specific
gauge_1 spec (min_deg=-47.06, sweep_deg=114.99, needle_colour="black" which
also breaks the loader's dark/light validation). The correct LittleGood
spec is min_deg=135, sweep_deg=270, -30C..+50C, needle_colour="dark",
obb_pivot_y_offset_ratio=0.0625.

## Results (17 test_3 images with GT temperature in filename)

Using the CURRENT best keypoint model (wide_aug, 56x56 output):
**MAE 1.52C, median 1.55C, max 4.05C, 71% within 2C, 100% within 5C.**

- p10c 0.1C, p30c 0.2C, p35c 0.3C, p31c 0.7C, p42c 1.1C — excellent
- Worst: p50c preview 4.1C (needle near sweep end)
- The p31c/p42c "hard cases" that broke raw keypoint metrics are handled
  well through the ellipse crop + calibration

## Key insight

Raw per-split keypoint MAE (14px tip) does NOT predict temperature
accuracy: the 270-degree sweep spans 80C, so a 4px tip error at the
needle end is a fraction of a degree. The pipeline-level metric is what
matters for the board.

## Next

Rerun with the stride-2 keypoint model (`keypoint_unet_224g_stride2`) once
its full-data training finishes (expected MAE <= current).
