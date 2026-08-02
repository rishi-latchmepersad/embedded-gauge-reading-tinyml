# Combined focal + decoded-keypoint L1 loss — 2026-07-30

Date: 2026-07-30
Status: experimental
Scope: keypoint UNet loss function
Evidence: `ml/artifacts/gauge_keypoint_unet_224g_v9/`

## The finding

Adding a differentiable decoded-keypoint L1 loss on top of the focal heatmap
loss improved val/test metrics dramatically (center 3.47→2.65px, tip 7.08→5.78px)
but did NOT improve per-split evaluation on the full test sets. The combined
loss model (v9) scored worse than v6 on test_1 and test_3 per-split.

## Per-split results (int8)

| Split | v6 (focal only) | v9 (focal + L1) |
|-------|----------------|-----------------|
| test_1 center | **3.63px** | 4.99px |
| test_1 tip | **14.87px** | 15.55px |
| test_2 center | 8.87px | **8.20px** |
| test_2 tip | **14.07px** | 15.36px |
| test_3 center | **16.57px** | 18.13px |
| test_3 tip | **24.10px** | 43.01px |

## Why it didn't help per-split

The decoded-keypoint L1 loss uses argmax-based GT keypoint extraction from
the heatmap, which is non-differentiable and noisy. The model learns to
produce heatmaps whose argmax peaks are close to the GT argmax peaks, but
this doesn't necessarily improve the softargmax-decoded coordinates that
the evaluation uses.

The val/test improvement was likely due to the model overfitting to the
first 200 images used for evaluation, not genuine improvement.

## Lesson

Adding a decoded-keypoint L1 loss on top of focal heatmap loss does not
help when the GT keypoint is extracted via argmax (non-differentiable).
The gradient signal from the L1 loss is noisy and doesn't align with
the softargmax decoding used at inference time.

If direct keypoint supervision is desired, the GT coordinates should be
computed from the original annotations (not from heatmap argmax), and
the loss should be applied to the softargmax-decoded predictions.
