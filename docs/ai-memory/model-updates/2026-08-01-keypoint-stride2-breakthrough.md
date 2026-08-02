# Keypoint BREAKTHROUGH: stride-2 112x112 output passes the tip gate — 2026-08-01

Date: 2026-08-01
Status: current
Scope: center/tip keypoint model (tip <10px target)
Evidence: `ml/artifacts/keypoint_screen_fast/leaderboard.json` (screen v6),
`/tmp/screen_keypoint.log`

## Screen v6 leaderboard (40 epochs, 3,000+3,000 aug samples, rotation-only aug)

| Rank | Arch | center | tip | score | params |
|---|---|---|---|---|---|
| **1** | **unet_stride2** (112x112 out) | 3.57px | **9.00px** | 12.57px | 2,300,786 |
| 2 | unet_stride2_wider (alpha=2.0) | 3.60px | 9.07px | 12.67px | 4,088,386 |
| 3 | unet_wider (56x56 out) | 3.35px | 10.19px | 13.54px | 3,977,538 |
| 4 | unet_wide (56x56 out) | 3.37px | 10.69px | 14.06px | 2,238,386 |

## The finding

**Every 56x56-output variant plateaued at 10.2-10.7px tip** across six
screens (GAP/polar/offset/offmap heads, alpha 1.0-2.0, deep, augmentation).
The stride-2 output head (112x112 heatmaps = 2px cells instead of 4px)
broke the wall: 9.00px tip — under the 10px gate.

This confirms the research doc's note (section 2.1): heatmap stride is the
quantization floor, and stride-2 halve it. The DARK Taylor decode failed
because the 56x56 int8 heatmaps were too coarse for second derivatives;
the right fix was finer heatmaps, not fancier decoding.

## Also ruled out this pass

- DARK decode (Zhang 2020): worse on int8 heatmaps (27.6 vs 18.4 c+t px)
- Test-time flip averaging: much worse (112px) — flipped dials are
  impossible views; the model's flipped predictions disagree
- CenterNet offmap head: fails in every form (884px tip even balanced)

## Next

1. Full-data run: `unet_stride2` (alpha=1.5, 112x112 output) + rotation
   augmentation on all 7,922 images, then QAT + int8 + per-split eval.
2. If test_2/test_3 tips stay >10px after the full run, consider RTMW-style
   FPN/HEM multi-scale features (the next literature lever) or DWPose
   distillation from a stride-2 teacher.
