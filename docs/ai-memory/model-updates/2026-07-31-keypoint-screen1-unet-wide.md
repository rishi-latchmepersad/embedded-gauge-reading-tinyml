# Keypoint screen #1 — unet_wide wins, normalization bug fixed — 2026-07-31

Date: 2026-07-31
Status: current
Scope: center/tip keypoint model architecture (56x56x2 heatmap contract)
Evidence: `ml/artifacts/keypoint_screen_fast/leaderboard.json`, `/tmp/screen_keypoint.log`

## Screen results (40 epochs, 3,000-sample subset, cosine LR, FP32, test_1 slice)

| Rank | Arch | center | tip | score | params |
|---|---|---|---|---|---|
| **1** | **unet_wide** (alpha=1.5) | 4.11px | 10.85px | **14.96px** | 2,238,386 |
| 2 | unet_deep | 4.47px | 11.75px | 16.22px | 1,438,754 |
| 3 | unet_v1 (deployed arch) | 4.24px | 13.46px | 17.70px | 995,746 |
| 4 | unet_eca | 6.05px | 17.63px | 23.68px | 1,000,822 |
| 5 | unet_noskip | 6.13px | 20.30px | 26.43px | 714,242 |

## Findings

1. **Width beats depth, attention, and skip removal** — same pattern as the
   ellipse screen. unet_wide (alpha=1.5) is the winner at 2.24M params,
   within the 2.5 MB SRAM budget.
2. **CRITICAL BUG FOUND**: the screen evaluated with raw uint8 images
   (0-255) while training on float32 [0,1]; the BN running stats saturated
   and every architecture scored 70-160px garbage. The deployed v6 sanity
   check scored 2.9/6.5px through the same eval only after normalizing.
   Fixed: `test_inputs = images.astype(np.float32) / 255.0`.
3. **Data regenerated with all board archives**: train 7,922 (was 7,525),
   val 925, test 946 — all with centre+tip labels in three CVAT formats
   (box/ellipse/points).

## Next

Promote unet_wide via `train_keypoint_unet_224.py --alpha 1.5` on the full
7,922-image set (60+20 epochs per the deployed recipe), then QAT + int8 +
per-split eval. If tip stays >10px, try wide+deep (alpha 1.5 + extra
bottleneck stage) or tip-weight tuning.
