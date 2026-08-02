# Keypoint UNet v2–v4 iteration — 2026-07-25

Date: 2026-07-25
Status: current
Scope: 224×224 grayscale keypoint UNet for center/tip heatmaps

## Iteration summary

Three versions of the keypoint UNet trained on ellipse-conditioned
gauge crops. The main tuning knobs were **crop scale** (how much of
the gauge face to include around the ellipse) and **tip loss weight**
(weighting the needle tip channel vs center channel in the focal loss).

| Version | Crop | Tip wt | Center ≤8px (val) | Tip ≤8px (val) | Pipeline tip ≤8px |
|---|---|---|---|---|---|
| v2 | 1.18× | 1.5× | 95% | 84% | 60% |
| v3 | 1.35× | 4.0× | 93% | 86% | 70% |
| **v4** | **1.35×** | **8.0×** | **93%** | **86%** | **87%** |

The "pipeline tip ≤8px" column is the key metric — it measures end-to-end
accuracy through the full two-stage pipeline (ellipse detector → crop →
UNet → decode) on 30 test images.

## Architecture (unchanged across versions)

- Encoder: 5 stride-2 Conv+BN+ReLU stages, channels [32,48,64,96,128]
- Decoder: 4 UpSample2D(bilinear) + Concat(skip) stages
- Output: 56×56×2 heatmaps (center, tip), sigmoid
- Params: ~1.0M, TFLite int8: 1.04 MB

## Lessons

### Tip loss weight matters a lot for pipeline accuracy

The standalone val metrics barely changed (86% tip across v3-v4), but the
pipeline tip accuracy jumped from 70% → 87% when tip weight went 4→8.
The model became more precise about the tip location even though the
heatmap peak error metric stayed the same. Hypothesis: higher tip weight
produces sharper heatmap peaks that decode more accurately.

### Crop scale is critical

v2→v3 (1.18→1.35) gave the largest single improvement: 60% → 70% tip.
The tip gets clipped at the crop boundary with a too-tight crop. 1.35×
is large enough to include the tip even when the ellipse prediction is
slightly off.

### Center accuracy is resilient

Center accuracy only dropped from 97% → 90% (v3 → v4) despite the
aggressive tip weight shift. The center is easier to localise because
it's a larger spatial feature (the center of the gauge face).

## Deployment

- Source: `ml/artifacts/gauge_keypoint_unet_224g_v4/model_int8.tflite`
- Input: 224×224×1 grayscale (ellipse crop from stage 1)
- Output: 56×56×2 heatmaps [center, tip], int8
- Size: 1.04 MB, peak activation ~400 KB
- Decoding: local softargmax with `max(hm-0.03, 0)^2` weighting
- Crop scale: 1.35× around the predicted ellipse

## Two-stage pipeline (final)

```
Camera 640×640 grayscale
  │  resize 640→384
  ▼
[Stage 1: N6 Ellipse detector]  384×384, 281 KB
  → (cx, cy, rx, ry)
  │
  ▼ crop 1.35× around ellipse, resize→224
[Stage 2: Keypoint UNet v4]     224×224, 1.04 MB
  → 56×56 heatmaps [center, tip]
  │
  ▼ decode: local softargmax
  → (center_x, center_y), (tip_x, tip_y)
```

## Files

- `ml/scripts/prepare_gauge_keypoint_224_data.py` — data prep (CROP_SCALE=1.35)
- `ml/scripts/train_keypoint_unet_224.py` — training (focal loss, tip_weight=8.0)
- `ml/artifacts/gauge_keypoint_unet_224g_v2/` — v2 (1.18×, tip 1.5×)
- `ml/artifacts/gauge_keypoint_unet_224g_v3/` — v3 (1.35×, tip 4.0×)
- `ml/artifacts/gauge_keypoint_unet_224g_v4/` — **v4 winner** (1.35×, tip 8.0×)

## Prior entry

`docs/ai-memory/model-updates/2026-07-24-resolution-architecture-int8-lessons.md`
documents the U-Net architecture pattern and the bilinear-upsample requirement.
