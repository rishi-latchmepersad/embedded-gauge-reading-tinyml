"""Conservative QAT-encoder for STM32N6 — no HyperRAM.

This is the Cube.AI-friendly replacement for the current
gauge_ellipse_qat_encoder_384g_cvat_v1. The channels are kept
small so Neural Art can fit every temporary/workspace buffer in
on-chip NPU SRAM without HyperRAM spillover.

Architecture (per deepseek_ellipse_encoder_n6_384_retrain_handoff):
  384x384x1
  Conv 3x3 s2, 16 -> BN -> ReLU    # 192x192x16  = 590 KB peak int8
  Conv 3x3 s1, 16 -> BN -> ReLU
  Conv 3x3 s2, 24 -> BN -> ReLU    #  96x96x24   = 221 KB
  Conv 3x3 s1, 24 -> BN -> ReLU
  Conv 3x3 s2, 32 -> BN -> ReLU    #  48x48x32   =  74 KB
  Conv 3x3 s1, 32 -> BN -> ReLU
  Conv 3x3 s2, 48 -> BN -> ReLU    #  24x24x48   =  28 KB
  Conv 3x3 s1, 48 -> BN -> ReLU
  Conv 3x3 s2, 64 -> BN -> ReLU    #  12x12x64   =   9 KB
  Conv 3x3 s1, 64 -> BN -> ReLU
  GlobalAveragePooling2D
  Dense 32 -> ReLU
  Dense 5 -> sigmoid   (single output [cx,cy,rx,ry,conf], int8)

Output is a single Dense(5, sigmoid) — no separate heads, no Lambda,
no Flatten, no Concatenate. TFLite-compatible QAT only.
"""

from __future__ import annotations

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model


def _conv_bn_relu(x, filters, stride, name):
    x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_ellipse_encoder_n6_384(channels=None):
    """Conservative encoder for STM32N6 without HyperRAM.

    Args:
        channels: List of per-layer channel counts, 10 entries for the
                 10 Conv+BN+ReLU blocks. Default is the conservative
                 schedule from the handoff note.
    """
    if channels is None:
        channels = [16, 16, 24, 24, 32, 32, 48, 48, 64, 64]

    if len(channels) != 10:
        raise ValueError(f"channels must have 10 entries, got {len(channels)}")

    inputs = keras.Input(shape=(384, 384, 1), name="image")
    x = inputs

    # 5 stride-2 stages, each with 2 convs.
    for stage in range(5):
        c1, c2 = channels[stage * 2], channels[stage * 2 + 1]
        stride = 2 if stage == 0 else 2  # first in stage always stride-2
        x = _conv_bn_relu(x, c1, stride=2, name=f"s{stage}a")
        x = _conv_bn_relu(x, c2, stride=1, name=f"s{stage}b")

    # Head: 3 separate sigmoid heads — each gets its own int8 grid.
    # The single-Dense(5) head failed parity on the radius because the
    # narrow radius range [0.4, 0.5] shares an int8 grid with the wider
    # center range. Separate heads give each output its own quantization
    # scale, which matches the proven pattern from the v1 model (85%).
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    shared = layers.Dense(32, activation="relu", name="shared")(x)
    center_xy = layers.Dense(2, activation="sigmoid", name="center_xy")(shared)
    radius_xy = layers.Dense(2, activation="sigmoid", name="radius_xy")(shared)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(shared)

    return Model(inputs=inputs, outputs=[center_xy, radius_xy, confidence],
                name="ellipse_encoder_n6_384")


__all__ = ["build_ellipse_encoder_n6_384"]
