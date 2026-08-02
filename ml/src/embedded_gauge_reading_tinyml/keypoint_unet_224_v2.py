"""QAT-friendly UNet for gauge center/tip keypoint heatmaps (v2).

Improvements over v1 (keypoint_unet_224.py):
  - Wider encoder (40/56/80/112/144 channels, was 32/48/64/96/128)
  - Deeper bottleneck (2× conv at 192ch, was 1× at 128ch)
  - Same 56×56 output resolution (proven, sub-pixel via softargmax)

Architecture:
  Input:   224×224×1  grayscale (gauge crop from ellipse)
  Encoder: 5 stages of Conv+BN+ReLU, stride 2, with skip connections.
  Bottleneck: 2× Conv+BN+ReLU at 192ch (was 1× at 128ch).
  Decoder: 4 upsampling stages, each UpSample2D(bilinear) + skip concat + Conv+BN+ReLU.
  Output:  56×56×2   heatmaps [center, tip], sigmoid activation.

Activation budget at int8:
  s1 (112×112×40):  501 KB
  s2 (56×56×56):    196 KB
  s3 (28×28×80):     63 KB
  s4 (14×14×112):    22 KB
  bottleneck (7×7×192):  9 KB
  d1 (14×14×144):    28 KB
  d2 (28×28×112):    88 KB
  d3 (56×56×80):    224 KB
  output (56×56×2):    6 KB
  Peak: ~501 KB int8 — well under 1.5 MB.

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


def build_keypoint_unet_224_v2(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    heatmap_size: int = 56,
    alpha: float = 1.0,
) -> Model:
    """Build a QAT-safe UNet for gauge center/tip keypoint heatmaps.

    Args:
        input_shape: (H, W, C) tuple. C must be 1 (grayscale).
        heatmap_size: Output heatmap spatial resolution (default 56).
        alpha: Width multiplier. 1.0 gives ~1.4M params (~1.4 MB int8).
    """
    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input(shape=input_shape, name="image")

    # ── Encoder ────────────────────────────────────────────────────────
    e1 = _encoder_stage(inputs, w(40), "e1", downsample=True)   # 224→112, 40ch
    e2 = _encoder_stage(e1, w(56), "e2")                         # 112→ 56, 56ch
    e3 = _encoder_stage(e2, w(80), "e3")                         #  56→ 28, 80ch
    e4 = _encoder_stage(e3, w(112), "e4")                        #  28→ 14, 112ch
    e5 = _encoder_stage(e4, w(144), "e5")                        #  14→  7, 144ch

    # ── Bottleneck (deeper: 2 conv blocks at 192ch) ───────────────────
    b = _conv_bn_relu(e5, w(192), name="bnk_a")
    b = _conv_bn_relu(b, w(192), name="bnk_b")

    # ── Decoder ────────────────────────────────────────────────────────
    d1 = _decoder_stage(b, e4, w(144), "d1")   # 7→14, concat e4 (14×14)
    d2 = _decoder_stage(d1, e3, w(112), "d2")  # 14→28, concat e3 (28×28)
    d3 = _decoder_stage(d2, e2, w(80), "d3")   # 28→56, concat e2 (56×56)

    # ── Output head ────────────────────────────────────────────────────
    x = _conv_bn_relu(d3, w(48), name="head_refine")
    outputs = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)

    return Model(inputs=inputs, outputs=outputs, name=f"keypoint_unet_224_v2_a{alpha}")


__all__ = ["build_keypoint_unet_224_v2"]
