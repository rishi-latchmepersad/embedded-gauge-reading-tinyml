# Keypoint UNet v6 — expanded training data wins — 2026-07-30

Date: 2026-07-30
Status: current
Scope: 224×224 grayscale keypoint UNet for center/tip heatmaps
Evidence: `ml/artifacts/gauge_keypoint_unet_224g_v6/`, `ml/scripts/train_keypoint_unet_224.py`

## The finding

Adding 201 temperature gauge images (`initial_temp_gauge/board_captures_1.zip`)
to the training set improved cross-gauge generalization dramatically, with no
regression on the primary pressure gauge test set.

## Per-split results (int8, 995K params, 1.04 MB)

| Split | v4 (before) | v6 (after) | Change |
|-------|------------|------------|--------|
| test_1 center (913 imgs) | 4.07px | **3.63px** | -11% |
| test_1 tip | 14.34px | 14.87px | ~same |
| test_2 center (11 imgs) | 11.85px | **8.87px** | -25% |
| test_2 tip | 104.35px | **14.07px** | **-87%** |
| test_3 center (22 imgs) | 24.91px | **16.57px** | -33% |
| test_3 tip | 45.25px | **24.10px** | -47% |

## What changed

1. **Data prep fixed** (`prepare_gauge_keypoint_224_data.py`):
   - Now handles all 3 CVAT annotation formats: `<box>`, `<ellipse>`, `<points>`
   - Recognises all label name variants: `Center`/`temp_center`, `Tip`/`tip_tip`,
     `GaugeFace`/`temp_dial`
   - Previously test_3 (22 images) was silently excluded because its `temp_*`
     labels and `<points>` shapes were not recognised

2. **Training data expanded**: `board_captures_1.zip` (201 temperature gauge
   images) added to the train split.  These are 224×224 board captures with
   `temp_dial` ellipse + `temp_center`/`temp_tip` point annotations.

3. **Same architecture as v4**: 5-stage encoder (32/48/64/96/124ch), 4-stage
   decoder, 56×56×2 heatmap output.  No architecture change — the improvement
   is purely from data.

## Key insight

The v4 model was trained only on pressure gauge web images (train_1) and
performed well on test_1 (same domain) but failed catastrophically on
test_2 (augmented board captures, 104px tip error) and test_3 (temperature
gauge, 45px tip error).  Adding just 201 in-domain images was enough to
dramatically improve generalisation.

## Augmentation lesson

Rotating heatmaps with `scipy.ndimage.rotate` corrupts the Gaussian peak
shape and degrades accuracy.  The correct approach is to rotate the image
and re-generate the heatmap from the rotated keypoint coordinates.  Even
with this fix, augmentation did not help — the model trained without
augmentation on the expanded dataset performed better than the augmented
version.

## Architecture experiments that failed

- **v2 wider model** (2.4M params, 40/56/80/112/144ch): larger but did not
  beat v4 on per-split eval.  Worse on test_1, marginally better on test_3.
  The extra capacity is not needed for this task.
- **Augmentation (rotation + brightness)**: even with correct heatmap
  re-generation, augmented training regressed val metrics.

## Files

- `ml/artifacts/gauge_keypoint_unet_224g_v6/model_int8.tflite` — best model
- `ml/artifacts/gauge_keypoint_unet_224g_v6/model_fp32.keras` — FP32 source
- `ml/scripts/train_keypoint_unet_224.py` — training script (unchanged from v4)
- `ml/scripts/prepare_gauge_keypoint_224_data.py` — updated data prep
- `ml/scripts/eval_keypoint_per_split.py` — per-split evaluation
- `ml/src/embedded_gauge_reading_tinyml/keypoint_unet_224.py` — model arch
- `ml/src/embedded_gauge_reading_tinyml/keypoint_unet_224_v2.py` — wider variant (exploratory)
