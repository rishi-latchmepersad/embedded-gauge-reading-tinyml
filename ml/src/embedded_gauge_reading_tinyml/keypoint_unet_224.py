"""QAT-friendly UNet for gauge center/tip keypoint heatmaps.

Architecture:
  Input:   224x224x1  grayscale (gauge crop from ellipse)
  Encoder: 5 stages of Conv+BN+ReLU, stride 2, with skip connections.
  Bottleneck: 7x7 Conv+BN+ReLU.
  Decoder: 4 upsampling stages, each UpSample2D(bicubic) + Conv+BN+ReLU + skip concat.
  Output:  56x56x2   heatmaps [center, tip], sigmoid activation.

Activation budget at int8:
  s1 (112x112x32):  401 KB
  s2 (56x56x48):    154 KB
  s3 (28x28x64):     50 KB
  s4 (14x14x96):     19 KB
  s5 (7x7x128):       6 KB
  d1 (14x14x128):    25 KB
  d2 (28x28x96):     77 KB
  d3 (56x56x64):    205 KB
  d4 (112x112x48):  602 KB
  output (56x56x2):    6 KB
  Peak: ~602 KB int8 — well under 1.5 MB.

All layers are Conv2D + BatchNormalization + ReLU (no SE, no Multiply,
no Lambda) so tfmot.quantize_model() can wrap the graph cleanly.
"""

from __future__ import annotations

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model


def _conv_bn_relu(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    name: str = "",
) -> tf.Tensor:
    """3x3 Conv2D(no bias) + BatchNorm + ReLU — the QAT-safe primitive."""
    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same",
        use_bias=False, name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.9, name=f"{name}_bn",
    )(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _encoder_stage(x, filters, name, downsample=True):
    """Two Conv+BN+ReLU blocks; first downsamples if requested. Returns (output, skip)."""
    x = _conv_bn_relu(x, filters, stride=2 if downsample else 1, name=f"{name}a")
    x = _conv_bn_relu(x, filters, name=f"{name}b")
    return x


def _decoder_stage(x, skip, filters, name):
    """UpSample2D + concat skip + 2 Conv+BN+ReLU blocks."""
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = _conv_bn_relu(x, filters, name=f"{name}a")
    x = _conv_bn_relu(x, filters, name=f"{name}b")
    return x


def build_keypoint_unet_224(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    alpha: float = 1.0,
    stride2: bool = False,
) -> Model:
    """Build a QAT-safe UNet for gauge center/tip keypoint heatmaps.

    Args:
        input_shape: (H, W, C) tuple. C must be 1 (grayscale).
        alpha: Width multiplier. 1.0 gives ~850K params (~0.85 MB int8).
        stride2: Output 112x112 heatmaps (2px cells) instead of 56x56
            (4px cells).  Screen v6 showed stride-2 is the tip gate winner
            (9.0px vs 10.7px) because the 56x56 int8 heatmaps were too
            coarse for the needle tip.

    Activation budget at int8 (stride2=False):
      s1 (112x112x32):  401 KB   d3 (56x56x64):    205 KB
      d4 (112x112x32):  401 KB   output (56x56x2):   6 KB
      Peak: ~602 KB int8 — well under 1.5 MB.

    All layers are Conv2D + BatchNormalization + ReLU (no SE, no Multiply,
    no Lambda) so tfmot.quantize_model() can wrap the graph cleanly.
    """
    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input(shape=input_shape, name="image")

    # ── Encoder ────────────────────────────────────────────────────────
    e1 = _encoder_stage(inputs, w(32), "e1", downsample=True)   # 224→112, 32ch
    e2 = _encoder_stage(e1, w(48), "e2")                        # 112→ 56, 48ch
    e3 = _encoder_stage(e2, w(64), "e3")                        #  56→ 28, 64ch
    e4 = _encoder_stage(e3, w(96), "e4")                        #  28→ 14, 96ch
    b = _encoder_stage(e4, w(128), "e5")                         #  14→  7, 128ch (bottleneck)

    # ── Decoder ────────────────────────────────────────────────────────
    d1 = _decoder_stage(b, e4, w(96), "d1")   # 7→14, concat e4
    d2 = _decoder_stage(d1, e3, w(64), "d2")  # 14→28, concat e3
    d3 = _decoder_stage(d2, e2, w(48), "d3")  # 28→56, concat e2
    if stride2:
        # why: one more upsample to 112x112 with the e1 skip halves the
        # keypoint quantization floor (4px -> 2px cells), which screen v6
        # showed is the tip accuracy gate.
        d4 = _decoder_stage(d3, e1, w(32), "d4")  # 56→112, concat e1
        x = _conv_bn_relu(d4, w(32), name="head_refine")
    else:
        x = _conv_bn_relu(d3, w(32), name="head_refine")

    # ── Output head ────────────────────────────────────────────────────
    outputs = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)

    name = f"keypoint_unet_224_a{alpha}" + ("_s2" if stride2 else "")
    return Model(inputs=inputs, outputs=outputs, name=name)


__all__ = ["build_keypoint_unet_224"]
