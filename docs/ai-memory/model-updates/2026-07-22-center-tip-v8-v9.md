# Center/Tip U-Net v8–v9 Experiments — 2026-07-22

Date: 2026-07-22
Status: experimental
Scope: `gauge_center_tip` UNet family, int8 quantized, 160×160×2 input, 80×80×2 heatmap output

## Summary

We trained two deeper U-Net generations (v8, v9) to replace the poorly-performing
"wide" crop models (`gauge_center_tip_wide_weighted_littlegood_v1` and
`gauge_center_tip_wide_hybrid_littlegood_v1`) and the compact v7 UNet.

## Models compared (all on 97-sample LittleGood test set)

| Model | Center ≤8px | Tip ≤8px | Center err | Tip err | Size |
|-------|------------|----------|-----------|---------|------|
| v7 (compact UNet, 16/24/40/64) | 84.5% | 47.4% | 8.8 px | 16.0 px | 189 KB |
| v8 (deeper UNet, 24/36/56/96, focal 1.5:10, focal^2) | 80.4% | 62.9% | 11.9 px | 13.5 px | 379 KB |
| **v9 (deeper UNet, 24/36/56/96, focal 4:6, focal^1.5)** | **86.6%** | **75.3%** | **8.9 px** | **11.2 px** | 379 KB |
| wide weighted v1 | 79.4% | 62.9% | 9.6 px | 16.9 px | 604 KB |
| wide hybrid v1 | 75.3% | 65.0% | 10.6 px | 20.0 px | 604 KB |
| line radius v1 | 86.6% | 58.8% | 9.0 px | 25.6 px | 604 KB |

## Key findings

1. **Deeper U-Net with proper loss weights dominates**: The 379 KB model (v9)
   beats the 189 KB model (v7) on both center (+2.1%) and tip (+27.9%).

2. **Focal heatmap loss weights matter enormously**: [4:6 center:tip] with
   y_true^1.5 focal exponent gave 86.6%/75.3% vs [1.5:10] with y_true^2 which
   gave 80.4%/62.9%. Over-weighting the tip channel backfires.

3. **Radius head hurts accuracy**: All models with a scalar radius regression
   head (wide_weighted, wide_hybrid, direction_radius, line_radius) underperform
   the pure-heatmap approach. The direction+radius approach compounds errors.

4. **224×224 upscale from preprocessed 160² data does NOT help**: Upscaling
   160² inputs to 224² trained a 5-stage UNet that achieved 88.7% center but
   only 26.6 px tip error — worse on tip than any 160² model. Need to reprocess
   directly from 640² original images for true high-res benefit.

## Training recipe (v9, the best)

- Architecture: 160²×2 input → 3 encoder stages (24→36→56 channels) → 96-ch
  bottleneck → 2 decoder stages → 80²×2 sigmoid heatmaps
- Loss: `focal_heatmap_loss` with channel weights [4.0, 6.0], baseline=28,
  focal exponent=1.5
- Augmentation: random 90° rotation + photometric (brightness ±0.12, contrast
  ±0.12, gamma [0.82, 1.18])
- Training: 30 FP32 epochs (Adam 1e-3, cosine decay with 2-epoch warmup)
  + 10 QAT epochs (Adam 2e-4)
- Data: 7,309 generic + 451 LittleGood train, 979 + 72 val, 97 LittleGood test
- Exports: `artifacts/gauge_center_tip_littlegood_v9/gauge_center_tip_v8_int8.tflite` (379 KB)

## Artifacts

- v8: `artifacts/gauge_center_tip_littlegood_v8/`
- v9: `artifacts/gauge_center_tip_littlegood_v9/`
- 224-v1: `artifacts/gauge_center_tip_224_littlegood_v1/`
- 224-v2: `artifacts/gauge_center_tip_224_littlegood_v2/`

## Next steps

- [ ] KD from v9 FP32 teacher → int8 student (save teacher weights before QAT)
- [ ] Reprocess 640² images directly to 224² (not upscale 160²)
- [ ] Ellipse v9 training to tighten crop quality
- [ ] Target: ≥90% center ≤8px, ≥80% tip ≤8px
- [ ] Evaluate angular reading error (θ = atan2(dy, dx) from center to tip)
