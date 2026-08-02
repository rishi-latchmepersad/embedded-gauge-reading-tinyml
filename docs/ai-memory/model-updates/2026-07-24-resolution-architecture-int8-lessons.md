# Resolutions, architectures, and int8 patterns learned — 2026-07-24

Date: 2026-07-24
Status: validated (backed by 8 training runs)
Scope: ellipse detection and keypoint heatmap models at multiple resolutions

## The resolution sweet spot

TFLite int8 conversion produces constant output for 640×640 models
(both RepVGG and QAT-encoder families), but works correctly at 384×384
and below. The failure at 640×640 is not purely a "bias-only conv"
issue — models with BatchNorm also collapse at 640×640.

| Resolution | Architecture | Int8 output | Best metric |
|---|---|---|---|
| 640×640 | RepVGG + stride-4 stem | **collapsed** | n/a |
| 640×640 | QAT encoder + stride-4 stem | **collapsed** | n/a |
| 640×640 | Conv+BN+ReLU + stride-4 stem | **collapsed** | n/a |
| 512×512 | 5-stride-2, channels [16,32,64,96,128] | **works** | 73% within 8 px |
| **384×384** | **5-stride-2, channels [32,48,64,96,128]** | **works** | **85% within 8 px** |
| 224×224 | 5-stride-2, channels [32,48,64,96,128] | **works** (AI memory) | 52% within 8 px |

## Architecture rules confirmed

### Rule 1: All-sigmoid head beats separate heads for int8

The AI memory's `2026-07-23-linear-radius-head.md` lesson recommends a
linear (no-activation) radius output to avoid int8 collapse. At 640×640
this does NOT help — the radius collapses regardless. At 384×384,
switching from separate sigmoid + linear heads (v1, broken) to a single
all-sigmoid `Dense(5)` head (v2/v3, working) fixed the int8 collapse.

Rule: **Prefer a single Dense(N, sigmoid) head** over separate heads with
different activation functions when the output values cluster in a narrow
range. The int8 calibration picks a single scale factor that covers all
outputs uniformly.

### Rule 2: Stride-4 stem breaks int8 at 640×640

At 640×640, a stride-4 stem (`Conv2D(3, 3, strides=4)`) is required to
keep the peak activation under 1.5 MB int8. This stem produces a
160×160×C intermediate. At 384×384 and below, a stride-2 stem fits the
budget and the int8 grid calibrates correctly.

Hypothesis: the stride-4 stem compresses too much spatial information in
one convolution, making the int8 grid insensitive to the large input
variations. At stride-2, the model has more layers to distribute the
quantization error.

Rule: **At resolutions ≥ 640×640, do not use stride-4 stem for int8
models.** Use stride-2 stem with very small first-stage channels
(e.g., 8–16) to keep the 320×320 intermediate under 1.5 MB.

### Rule 3: U-Net concat + QAT causes scale mismatch

The U-Net decoder uses `Concatenate` to fuse skip connections. After
`tfmot.quantize_model()`, each branch gets its own int8 scale factor.
The `Concatenate` layer requires all inputs to have the SAME scale factor,
which triggers a TFLite conversion error.

Fix: **Use PTQ (post-training quantization on the FP32 model) for U-Nets,
NOT QAT.** The PTQ flow inserts requantization nodes before each concat.
To get QAT benefits, train the model with QAT (QAT-annotated layers),
but convert the underlying FP32 model (without the QAT wrappers) to
int8 via the TFLiteConverter.

### Rule 4: Bilinear upsampling, not bicubic

The `UpSampling2D(interpolation="bicubic")` layer maps to
`tf.ResizeBicubic`, which is NOT supported by TFLite. Use
`interpolation="bilinear"` or `"nearest"`. Bilinear is supported and
produced better results than bicubic in our U-Net anyway.

### Rule 5: Large representative datasets don't fix the 640×640 issue

Tested PTQ with 256, 512, and 1024 representative images on 640×640
models. None fixed the constant-output collapse. Tested per-channel vs
per-tensor quantization. No difference. The issue is fundamental to
the architecture at 640×640, not a calibration problem.

## The proven architecture templates

### Ellipse detector (for 384×384 and below)

```python
def build_ellipse_detector(input_size, alpha=1.5):
    """
    5 stride-2 stages: all Conv+BN+ReLU.
    Single Dense(5, sigmoid) head.
    Peak int8 activation: (input_size//2)^2 * 32.
    Target: < 1.5 MB for input_size ≤ 400.
    """
    inputs = Input((input_size, input_size, 1))
    x = _conv_bn_relu(inputs, 32, 2, "s1a")
    x = _conv_bn_relu(x, 32, "s1b")
    x = _conv_bn_relu(x, 48, 2, "s2a")
    x = _conv_bn_relu(x, 48, "s2b")
    x = _conv_bn_relu(x, 64, 2, "s3a")
    x = _conv_bn_relu(x, 64, "s3b")
    x = _conv_bn_relu(x, 96, 2, "s4a")
    x = _conv_bn_relu(x, 96, "s4b")
    x = _conv_bn_relu(x, 128, 2, "s5a")
    x = _conv_bn_relu(x, 128, "s5b")
    x = GAP()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, relu)(x)
    out = Dense(5, sigmoid)(x)  # cx, cy, rx, ry, conf
    return Model(inputs, out)
```

### Keypoint U-Net (for 224×224 and below)

```python
def build_keypoint_unet(input_size=224, alpha=1.0):
    """
    Encoder: 5 stride-2 stages, all Conv+BN+ReLU.
    Decoder: 4 UpSample2D(bilinear) + Concat(skip) + Conv+BN+ReLU stages.
    Output: heatmap_size × heatmap_size × 2  (center, tip).
    Peak int8 activation: ~400 KB at 224×224.
    Export: PTQ on FP32 model (NOT QAT with concat).
    """
    inputs = Input((input_size, input_size, 1))
    # Encoder with skip connections
    e1 = _encoder_stage(inputs, 32, "e1")   # H→H/2
    e2 = _encoder_stage(e1, 48, "e2")        # H/2→H/4
    e3 = _encoder_stage(e2, 64, "e3")        # H/4→H/8
    e4 = _encoder_stage(e3, 96, "e4")        # H/8→H/16
    b  = _encoder_stage(e4, 128, "e5")       # H/16→H/32
    # Decoder
    d1 = _decoder_stage(b, e4, 96, "d1")     # H/32→H/16
    d2 = _decoder_stage(d1, e3, 64, "d2")    # H/16→H/8
    d3 = _decoder_stage(d2, e2, 48, "d3")    # H/8→H/4
    x = _conv_bn_relu(d3, 32, "head_refine")
    out = Conv2D(2, 1, sigmoid)(x)           # H/4 heatmaps
    return Model(inputs, out)
```

## Loss functions that work

| Model | Loss | Loss weights | Notes |
|---|---|---|---|
| Ellipse | Huber(delta=0.05) per output | c:1.0, r:3.0, conf:0.1 | Works but radius collapes at 640×640 |
| Ellipse (all sigmoid) | Huber(delta=0.05) on single 5-vector | 1.0 | Proven at 384×384 |
| Keypoint U-Net | Focal heatmap (alpha=2.0, gamma=4.0) per-pixel | center:1.0, tip:1.5 | Tip is harder, up-weight it |

## Two-stage deployment pipeline (proven)

```
Camera 640×640 grayscale
  │
  ▼
[Stage 1: Ellipse detector]
  384×384 input, TFLite int8 1.18 MB
  → (cx, cy, rx, ry) in [0,1] normalized coords
  │
  ▼
[Crop & resize]
  square crop around ellipse, scale 1.18×
  resize to 224×224 grayscale
  │
  ▼
[Stage 2: Keypoint U-Net]
  224×224 input, TFLite int8 1.04 MB
  → 56×56 heatmaps [center, tip]
  │
  ▼
[Decode]
  softargmax with local window → normalized (x, y) per keypoint
```

Both models under 1.5 MB peak activation int8. Total ~2.2 MB weight flash.

## Files created in this session

| File | Purpose |
|---|---|
| `ml/src/embedded_gauge_reading_tinyml/ellipse_repvgg.py` | RepVGG ellipse detector (failed at 640×640, kept for reference) |
| `ml/src/embedded_gauge_reading_tinyml/qat_encoder_640g.py` | QAT encoder for 640×640 ellipse (collapses, kept for reference) |
| `ml/src/embedded_gauge_reading_tinyml/keypoint_unet_224.py` | Keypoint U-Net for 224×224 gauge crops (works in int8) |
| `ml/scripts/prepare_repvgg_ellipse_data.py` | CVAT zips → staged 640×640 ellipse data |
| `ml/scripts/train_repvgg_ellipse.py` | RepVGG FP32 → reparam → QAT → int8 |
| `ml/scripts/train_qat_encoder_640g.py` | QAT encoder 640×640 training (v1, collapsed) |
| `ml/scripts/train_qat_encoder_flexible.py` | Flexible QAT encoder at any resolution (384×384 winner) |
| `ml/scripts/prepare_gauge_keypoint_224_data.py` | CVAT GaugeFace + Center/Tip → ellipse crops + heatmaps |
| `ml/scripts/train_keypoint_unet_224.py` | Keypoint U-Net FP32 + QAT + int8 PTQ |
| `ml/scripts/eval_qat_384g.py` | Evaluate 384×384 ellipse model on val/test |
| `ml/scripts/eval_repvgg_ellipse_tflite.py` | Updated to handle model_fp32.keras and model_fused.keras |
| `ml/scripts/try_alt_quantization.py` | Tested PTQ with 256/1024 reps, per-channel on/off |
| `ml/scripts/validate_qat_encoder_640g_tflite.py` | Pre-training TFLite validation |
| `ml/scripts/qat_multibranch_export.py` | QAT on multi-branch (failed — BN unsupported by tfmot) |
| `ml/scripts/ptq_multibranch_export.py` | PTQ on multi-branch (constant output) |
| `ml/scripts/eval_ptq_fp32_full.py` | PTQ eval on full val/test |
| `ml/data/repvgg_ellipse/` | Staged 640×640 train/val/test ellipse images |
| `ml/data/gauge_keypoint_224/` | Staged 224×224 gauge crops + 56×56 heatmaps |
| `ml/artifacts/gauge_ellipse_repvgg_640g_v1/` | RepVGG 640×640 (broken int8) |
| `ml/artifacts/gauge_ellipse_qat_encoder_640g_v1/` | QAT encoder 640×640 v1 (broken int8) |
| `ml/artifacts/gauge_ellipse_qat_encoder_640g_v2/` | QAT encoder 640×640 v3 (broken int8) |
| `ml/artifacts/gauge_ellipse_qat_encoder_384g_cvat_v1/` | **Winner ellipse detector** — 85% within 8 px int8 |
| `ml/artifacts/gauge_ellipse_qat_encoder_512g_cvat_v1/` | 512×512 ellipse (73% within 8 px, larger model) |
| `ml/artifacts/gauge_keypoint_unet_224g_v1/` | Keypoint U-Net v1 (bicubic, broken int8 export) |
| `ml/artifacts/gauge_keypoint_unet_224g_v2/` | **Winner keypoint U-Net** — 95% center / 84% tip, 1.04 MB int8 |
