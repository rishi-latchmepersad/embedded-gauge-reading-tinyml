"""QAT-safe UNet for needle center/tip keypoint heatmaps from gauge face crops.

Architecture:
  Input:   224x224x1  grayscale (gauge face crop from ellipse detector)
  Encoder: 5 stages Conv+BN+ReLU stride 2 with skip connections
  Bottleneck: 7x7 Conv+BN+ReLU
  Decoder: 4 upsample stages with skip concat
  Output:  56x56x2   heatmaps [center, tip], sigmoid activation

Peak int8 activation budget:
  s1 (112x112x32): 401 KB  ← peak
  s2 (56x56x48):   154 KB
  s3 (28x28x64):    50 KB
  s4 (14x14x96):    19 KB
  s5 (7x7x128):      6 KB
  d1 (14x14x96):    19 KB
  d2 (28x28x64):    50 KB
  d3 (56x56x48):   154 KB
  output (56x56x2):   6 KB
  Peak: ~401 KB int8 — well under 2.5 MB budget.

Design rationale:
  - Conv+BN+ReLU only — QAT-safe (no bias-only convs, no Lambda, no Multiply)
  - Bilinear upsample (not bicubic) — required for TFLite compatibility
  - Single Conv2D(2, sigmoid) output head — keeps quantization grid shared
  - Skip connections preserve spatial precision for sub-pixel localization
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
    """3x3 Conv2D(no bias) + BatchNorm + ReLU — QAT-safe primitive."""
    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same",
        use_bias=False, name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.9, name=f"{name}_bn",
    )(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _encoder_stage(x: tf.Tensor, filters: int, name: str, downsample: bool = True) -> tf.Tensor:
    """Two Conv+BN+ReLU blocks; first downsamples if requested. Returns output."""
    x = _conv_bn_relu(x, filters, stride=2 if downsample else 1, name=f"{name}a")
    x = _conv_bn_relu(x, filters, name=f"{name}b")
    return x


def _decoder_stage(x: tf.Tensor, skip: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """UpSample2D(bilinear) + concat skip + 2 Conv+BN+ReLU blocks."""
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = _conv_bn_relu(x, filters, name=f"{name}a")
    x = _conv_bn_relu(x, filters, name=f"{name}b")
    return x


def build_needle_unet_224(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    alpha: float = 1.0,
) -> Model:
    """Build a QAT-safe UNet for needle center/tip keypoint heatmaps.

    Args:
        input_shape: (H, W, C) tuple. C must be 1 (grayscale).
        alpha: Width multiplier. 1.0 gives channels [32,48,64,96,128].
    """
    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input(shape=input_shape, name="image")

    # ── Encoder ────────────────────────────────────────────────────────
    e1 = _encoder_stage(inputs, w(32), "e1", downsample=True)   # 224->112, 32ch
    e2 = _encoder_stage(e1, w(48), "e2")                        # 112-> 56, 48ch
    e3 = _encoder_stage(e2, w(64), "e3")                        #  56-> 28, 64ch
    e4 = _encoder_stage(e3, w(96), "e4")                        #  28-> 14, 96ch
    b = _encoder_stage(e4, w(128), "e5")                         #  14->  7, 128ch

    # ── Decoder ────────────────────────────────────────────────────────
    d1 = _decoder_stage(b, e4, w(96), "d1")   # 7->14, concat e4
    d2 = _decoder_stage(d1, e3, w(64), "d2")  # 14->28, concat e3
    d3 = _decoder_stage(d2, e2, w(48), "d3")  # 28->56, concat e2

    # ── Output head ────────────────────────────────────────────────────
    x = _conv_bn_relu(d3, w(32), name="head_refine")
    outputs = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)

    return Model(inputs=inputs, outputs=outputs, name=f"needle_unet_224_a{alpha}")


__all__ = ["build_needle_unet_224"]
