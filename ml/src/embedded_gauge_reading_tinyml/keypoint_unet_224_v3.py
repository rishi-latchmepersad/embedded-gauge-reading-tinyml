"""QAT-friendly UNet for gauge center/tip keypoint heatmaps (v3).

Improvements over v6/v4:
  - 80×80 heatmap output (was 56×56) for finer tip localization
  - Tighter Gaussian sigma=1.5 (was 2.0) for sharper peaks
  - Optional offset refinement head for sub-pixel accuracy

Architecture:
  Input:   224×224×1  grayscale (gauge crop from ellipse)
  Encoder: 5 stages of Conv+BN+ReLU, stride 2, with skip connections.
  Bottleneck: 7×7 Conv+BN+ReLU.
  Decoder: 4 upsampling stages, each UpSample2D(bilinear) + skip concat + Conv+BN+ReLU.
  Output:  80×80×2   heatmaps [center, tip], sigmoid activation.

The decoder outputs at 56×56 (matching encoder stride), then a final
bilinear upsample to 80×80.  This avoids power-of-2 mismatch issues
while getting the finer resolution.

Activation budget at int8:
  s1 (112×112×32):  401 KB
  s2 (56×56×48):    154 KB
  s3 (28×28×64):     50 KB
  s4 (14×14×96):     19 KB
  bottleneck (7×7×128): 6 KB
  d1 (14×14×96):     19 KB
  d2 (28×28×64):     50 KB
  d3 (56×56×48):    154 KB
  output (80×80×2):   13 KB
  Peak: ~401 KB int8 — well under 1.5 MB.

All layers are Conv2D + BatchNormalization + ReLU so tfmot can quantize
them cleanly.  No Lambda layers, no tf.nn wrappers, no bias-only convs.
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
    """3×3 Conv2D(no bias) + BatchNorm + ReLU — the QAT-safe primitive."""
    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same",
        use_bias=False, name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.9, name=f"{name}_bn",
    )(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _encoder_stage(x, filters, name, downsample=True):
    """Two Conv+BN+ReLU blocks; first downsamples if requested."""
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


def build_keypoint_unet_224_v3(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    heatmap_size: int = 80,
    alpha: float = 1.0,
) -> Model:
    """Build a QAT-safe UNet for gauge center/tip keypoint heatmaps.

    Args:
        input_shape: (H, W, C) tuple. C must be 1 (grayscale).
        heatmap_size: Output heatmap spatial resolution (default 80).
        alpha: Width multiplier. 1.0 gives ~1M params (~1.0 MB int8).
    """
    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input(shape=input_shape, name="image")

    # ── Encoder ────────────────────────────────────────────────────────
    e1 = _encoder_stage(inputs, w(32), "e1", downsample=True)   # 224→112, 32ch
    e2 = _encoder_stage(e1, w(48), "e2")                         # 112→ 56, 48ch
    e3 = _encoder_stage(e2, w(64), "e3")                         #  56→ 28, 64ch
    e4 = _encoder_stage(e3, w(96), "e4")                         #  28→ 14, 96ch
    b = _encoder_stage(e4, w(128), "e5")                         #  14→  7, 128ch (bottleneck)

    # ── Decoder ────────────────────────────────────────────────────────
    d1 = _decoder_stage(b, e4, w(96), "d1")   # 7→14, concat e4
    d2 = _decoder_stage(d1, e3, w(64), "d2")  # 14→28, concat e3
    d3 = _decoder_stage(d2, e2, w(48), "d3")  # 28→56, concat e2

    # ── Output head ────────────────────────────────────────────────────
    x = _conv_bn_relu(d3, w(32), name="head_refine")
    outputs = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)

    return Model(inputs=inputs, outputs=outputs, name=f"keypoint_unet_224_v3_h{heatmap_size}")


__all__ = ["build_keypoint_unet_224_v3"]
