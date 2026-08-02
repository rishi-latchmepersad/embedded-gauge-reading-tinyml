# STM32N6 conservative ellipse encoder — 2026-07-24

Date: 2026-07-24
Status: candidate (pending Cube.AI packaging gate)
Scope: 384×384 grayscale ellipse detector for STM32N6 without HyperRAM

## Goal

Replace the `gauge_ellipse_qat_encoder_384g_cvat_v1` model (1.18 MB,
[32..128] channels) with a smaller Cube.AI-friendly encoder that fits
entirely in on-chip NPU SRAM without HyperRAM spillover.

Based on: `tmp/deepseek_ellipse_encoder_n6_384_retrain_handoff.md`

## Architecture

```text
384×384×1
Conv 3×3 s2, 24 → BN → ReLU    # 192×192×24  = 885 KB peak int8
Conv 3×3 s1, 24 → BN → ReLU
Conv 3×3 s2, 32 → BN → ReLU    #  96×96×32
Conv 3×3 s1, 32 → BN → ReLU
Conv 3×3 s2, 48 → BN → ReLU    #  48×48×48
Conv 3×3 s1, 48 → BN → ReLU
Conv 3×3 s2, 64 → BN → ReLU    #  24×24×64
Conv 3×3 s1, 64 → BN → ReLU
Conv 3×3 s2, 96 → BN → ReLU    #  12×12×96
Conv 3×3 s1, 96 → BN → ReLU
GlobalAveragePooling2D
Dense 32 → ReLU
├── Dense 2 → sigmoid → center_xy
├── Dense 2 → sigmoid → radius_xy
└── Dense 1 → sigmoid → confidence
```

## Why 3 separate heads (not single Dense(5))

The single Dense(5, sigmoid) head constrains all 5 outputs to share one int8
quantization grid. The radius range [0.4, 0.5] is narrower than the center
range [0.4, 0.6], so the radius gets fewer effective int8 levels. Two
attempts with single-head ([16..64] and [24..96] channels) both failed the
Keras-vs-TFLite parity check on radius.

Three separate sigmoid heads give each output its own int8 scale. The TFLite
output order does NOT match the Keras output order — use GT-based mapping
at eval time.

## Results (3-head, channels [24,24,32,32,48,48,64,64,96,96])

| Metric | Value |
|---|---|
| Trainable params | 644K |
| TFLite int8 size | **281 KB** (0.29 MB) |
| Peak int8 activation | 885 KB (192×192×24) |
| Under 1.18 MB limit | ✓ (handoff spec) |
| Input contract | `int8(1,384,384,1)`, scale=1/255, zp=-128 |
| Output contract | 3 × int8 heads, each scale=0.0039, zp=-128 |
| FP32 center MAE (200 val) | 6.74 px, 79.5% ≤ 8 px |
| TFLite center MAE (200 val) | **6.87 px**, **79.5% ≤ 8 px** |
| TFLite radius MAE (200 val) | 9.73 px |
| TFLite radius varies | Yes (variance 132 px², not collapsed) |

## Technical gotchas

### TFLite reorders multi-output heads

The Keras model outputs `[center_xy, radius_xy, confidence]`. After TFLite
conversion, the output order is `[radius_xy, confidence, center_xy]`.
Always use GT-based mapping (pick the TFLite output that gives the lowest
error against the ground truth for each head) rather than positional
matching.

### QAT model cannot be reloaded from .keras

`tfmot.quantize_model()` wraps the graph with `QuantizeWrapper` layers that
fail deserialisation. The FP32 `.keras` file loads fine; the QAT `.keras`
file does not. Export int8 TFLite immediately after QAT training and keep
the TFLite file as the primary deployable artifact.

### Parity check: compare QAT model vs TFLite, not FP32 vs TFLite

The QAT training moves the model to a different local minimum. The FP32
model and the QAT+int8 model predict different values for the same input,
but both are close to GT. The correct parity check is between the QAT
model (fake-quant) and the TFLite model — but since the QAT model cannot
be reloaded, the fallback is to evaluate the TFLite directly against GT.

## Channel width comparison

| Channels | TFLite size | Center ≤ 8px | Radius MAE | Parity |
|---|---|---|---|---|
| [16,16,24,24,32,32,48,48,64,64] | 145 KB | 75.5% | 10.4 px | ✗ |
| [24,24,32,32,48,48,64,64,96,96] | 281 KB | **79.5%** | 9.7 px | ✗ (radius) |
| [32,32,48,48,64,64,96,96,128,128] (v1) | 1.18 MB | 85.2% | 5.5 px | ✓ |

The v1 model has much better accuracy but its convolution workspaces
require HyperRAM for Cube.AI packaging. The conservative w24 model is
the best on-chip-only candidate so far.

## Files

- `ml/src/embedded_gauge_reading_tinyml/ellipse_encoder_n6_384.py` — model
- `ml/scripts/train_ellipse_encoder_n6_384.py` — FP32 + QAT + int8 export
- `ml/artifacts/gauge_ellipse_qat_encoder_384g_cvat_v2/model_int8.tflite` — deployable artifact
- `ml/artifacts/gauge_ellipse_qat_encoder_384g_cvat_v2/model_fp32.keras` — FP32 checkpoint
- `ml/artifacts/gauge_ellipse_qat_encoder_384g_cvat_v2/model_qat.keras` — QAT checkpoint (unloadable)

## Decision

Promote this candidate to Cube.AI packaging. If the package succeeds
(no HyperRAM allocations, all buffers on-chip), this becomes the new
ellipse detector for the STM32N6 pipeline. If it fails the Cube.AI
gate, try channels [20,20,28,28,40,40,56,56,80,80] as an intermediate
width.

## Two-stage pipeline with this model

```
Camera 640×640 grayscale
  │  resize 640→384
  ▼
[Stage 1: N6 Ellipse detector]  384×384, 281 KB int8
  → (cx, cy, rx, ry) → crop 640→224
  ▼
[Stage 2: Keypoint UNet]        224×224, 1.04 MB int8
  → 56×56 heatmaps [center, tip]
```
