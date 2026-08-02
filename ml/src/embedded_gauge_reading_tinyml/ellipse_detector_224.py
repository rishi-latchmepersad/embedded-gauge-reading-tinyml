"""Compact QAT-safe ellipse detector for 224x224 grayscale input.

Architecture: 5-stage Conv+BN+ReLU encoder with stride-2 downsampling,
GlobalAveragePooling, and a single Dense(4, sigmoid) head for
[cx, cy, rx, ry] in normalized [0, 1] coordinates.

Design rationale:
  - Conv+BN+ReLU is the only QAT-safe pattern (bias-only convs collapse
    after int8 quantization — see lessons-learned/2026-07-23-qat-safe-architecture.md)
  - Single sigmoid head avoids the multi-head int8 reorder issue
  - 224x224 input keeps peak activation small (~401 KB int8)
  - No Lambda, no Multiply, no SE/CA — fully tfmot compatible

Peak int8 activation:
  s1 (112x112x32): 401 KB
  s2 (56x56x48):   154 KB
  s3 (28x28x64):    50 KB
  s4 (14x14x96):    19 KB
  s5 (7x7x128):      6 KB
  Total peak: ~401 KB — well under 2.5 MB budget.
"""

from __future__ import annotations

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model


def _conv_bn_relu(x: tf.Tensor, filters: int, stride: int = 1, name: str = "") -> tf.Tensor:
    """3x3 Conv2D(no bias) + BatchNorm + ReLU — QAT-safe building block."""
    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same",
        use_bias=False, name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.9, name=f"{name}_bn",
    )(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_ellipse_detector_224(
    channels: list[int] | None = None,
) -> Model:
    """Build a compact ellipse detector for 224x224 grayscale input.

    Args:
        channels: Per-layer channel counts (10 entries for 5 stages x 2 convs).
                  Default: [32, 32, 48, 48, 64, 64, 96, 96, 128, 128].
    """
    if channels is None:
        channels = [32, 32, 48, 48, 64, 64, 96, 96, 128, 128]

    if len(channels) != 10:
        raise ValueError(f"channels must have 10 entries, got {len(channels)}")

    inputs = keras.Input(shape=(224, 224, 1), name="image")
    x = inputs

    # 5 stride-2 stages, each with 2 convs: 224->112->56->28->14->7
    for stage in range(5):
        c1, c2 = channels[stage * 2], channels[stage * 2 + 1]
        x = _conv_bn_relu(x, c1, stride=2, name=f"s{stage}a")
        x = _conv_bn_relu(x, c2, stride=1, name=f"s{stage}b")

    # Regression head: GAP + Dense(4, sigmoid) for [cx, cy, rx, ry]
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.15, name="dropout")(x)
    x = layers.Dense(32, activation="relu", name="head_dense")(x)
    outputs = layers.Dense(4, activation="sigmoid", name="ellipse")(x)

    return Model(inputs=inputs, outputs=outputs, name="ellipse_detector_224")


__all__ = ["build_ellipse_detector_224"]
