# RepVGG ellipse detector for 640x640 grayscale — 2026-07-24

Date: 2026-07-24
Status: failed (TFLite int8 conversion collapses to constant output)
Scope: 640x640 grayscale ellipse detector for the gauge-face stage

## What was built

A RepVGG ellipse detector for 640x640 grayscale input, trained on
9,191 GaugeFace ellipse labels from `ml/data/labelled/{train,val,test}_*.zip`.

| Component | Value |
|---|---|
| Architecture | Multi-branch RepVGG (3x3 + 1x1 + identity BN) per block |
| Stem | 3x3 stride 4, 24 channels (640 → 160) |
| Backbone | [2, 4, 6, 2] RepVGG blocks at widths [48, 96, 192, 256] |
| Head | 5-vector: cx/cy sigmoid + rx/ry LINEAR + confidence sigmoid |
| Input | 640x640x1 grayscale, [0, 1] float |
| Output | center_xy (2), radius_xy (2), confidence (1) |
| Multi-branch params | 3.58 M (~3.6 MB int8) |
| Fused params | 3.22 M (~3.2 MB int8) |
| Trained | 60 epochs FP32 (AdamW, cosine LR, ReduceLROnPlateau) |

## What works

The **Keras fp32 fused model** performs well:

| Metric | Val | Test |
|---|---|---|
| Center MAE | 8.06 px | 10.42 px |
| Center median | 5.34 px | 5.18 px |
| % within 8 px | 71.3% | 70.8% |
| % within 4 px | 37.2% | 37.0% |
| Radius variance (pred) | 604 px² | 671 px² |
| Radius variance (GT) | 917 px² | 1402 px² |

The radius output is **not collapsed** (variance > 600), so the
linear-radius-head design from `2026-07-23-linear-radius-head.md` works.

Activation budget (int8 deployment):
- Input: 640x640x1 = 410 KB
- Stem: 160x160x24 = 614 KB
- Largest intermediate under 1.5 MB

## What fails (TFLite int8)

The **TFLite int8 model** (both QAT and PTQ paths) produces **constant
output for all inputs**. Verified with 3 random int8 inputs and 3 real
images -- all produced identical dequantized outputs.

| TFLite approach | Result |
|---|---|
| Fused + QAT (tfmot) | Constant output (130 px center error) |
| Fused + PTQ | Constant output |
| Multi-branch + PTQ | Constant output (with BN present) |
| Multi-branch + QAT | Failed to apply (BN layers unsupported by tfmot) |

The exact failure mode is the "bias-only conv" pattern documented in
`2026-07-23-qat-safe-architecture.md`. The fused RepVGG block has
Conv2D(use_bias=True) + ReLU per block, no BatchNorm. The int8
calibration picks (min, max) for the activations once, but without
BN to normalize the inputs, the int8 grid is too coarse and the
output collapses to a constant.

Even the multi-branch model (with BN present) collapses. The TFLite
converter strips BN during conversion and the resulting graph behaves
the same as the fused version.

## Why this matters

The AI memory's `2026-07-23-qat-safe-architecture.md` lesson states
clearly: "do not spend time on post-training quantization, float16
export, or conversion-rescue experiments once a family has shown
TFLite mismatch." This model has shown TFLite mismatch.

The lesson also lists RepVGG as QAT-compatible in principle ("Multi-branch
with Add -> QAT-compatible (after fusion)"), but in this environment
the RepVGG topology at 640x640 fails in practice. The TFLite converter
appears to mishandle the RepVGG multi-branch graph for int8.

## Files (for future reference)

- `ml/src/embedded_gauge_reading_tinyml/ellipse_repvgg.py` --
  multi-branch + fused RepVGG builders, reparameterize
- `ml/src/embedded_gauge_reading_tinyml/ellipse_repvgg_qat_safe.py` --
  attempted BN-inserted variant (unused, see below)
- `ml/scripts/prepare_repvgg_ellipse_data.py` -- CVAT zips -> staged data
- `ml/scripts/train_repvgg_ellipse.py` -- FP32 -> reparam -> QAT -> int8
- `ml/scripts/eval_repvgg_ellipse_tflite.py` -- parity + per-split eval
- `ml/scripts/qat_multibranch_export.py` -- QAT on multi-branch (failed)
- `ml/scripts/ptq_multibranch_export.py` -- PTQ on multi-branch (constant output)
- `ml/artifacts/gauge_ellipse_repvgg_640g_v1/` -- trained models

## Recommended next steps (alternatives)

Pick one:

1. **Switch to the QAT encoder architecture** documented in
   `2026-07-23-qat-encoder-ellipse.md`. That model is plain
   Conv+BN+ReLU at 224x224 and produced meaningful int8 outputs.
   Scale it up to 640x640 grayscale (more channels, deeper stages)
   and re-export.

2. **Re-use the Keras fp32 fused model** for an offline path that
   doesn't need int8 deployment. The 3.2 MB fp32 fused model is
   small enough to run on a Raspberry Pi or similar if the NPU
   deployment is not strictly required.

3. **Try a different TFLite export path** (e.g. edge-TPU specific
   converter, or ONNX -> TFLite via onnx2tf). This goes against
   the AI memory guidance but might unblock the project.

## Decision

Do not promote `gauge_ellipse_repvgg_640g_v1` to a deployment
candidate. The Keras fp32 fused model is saved and can be used
for offline evaluation, but the int8 TFLite export is broken.

Future RepVGG work on this hardware should start with the QAT
encoder architecture (alternative 1) and validate the int8
export on a small sample before scaling to full training.
