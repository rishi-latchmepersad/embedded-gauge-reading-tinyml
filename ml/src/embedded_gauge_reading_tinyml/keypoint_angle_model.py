"""QAT-friendly UNet for gauge center heatmap + needle angle regression.

Instead of predicting center + tip heatmaps (which suffers from 180° flips),
this model predicts:
  1. Center heatmap: 56×56×1 (same as before — center is always perfect)
  2. Needle angle: scalar in [0, 2π) — the angle from center to tip
  3. Needle radius: scalar — the normalized distance from center to tip

At inference:
  tip_xy = center_xy + radius * [cos(angle), sin(angle)]

The angle has NO 180° ambiguity — it uniquely identifies the needle direction.

Loss functions:
  - Center heatmap: focal loss (same as before)
  - Angle: cyclic loss 1 - cos(pred - gt) (handles 0°/360° wraparound)
  - Radius: L1 loss

Architecture:
  Same UNet encoder-decoder as v6, but with:
    - Output head 1: Conv2D(1, sigmoid) for center heatmap
    - GlobalAveragePooling + Dense(1) for angle
    - GlobalAveragePooling + Dense(1) for radius
"""

from __future__ import annotations

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model


def _conv_bn_relu(x, filters, stride=1, name=""):
    x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def _encoder_stage(x, filters, name, downsample=True):
    x = _conv_bn_relu(x, filters, stride=2 if downsample else 1, name=f"{name}a")
    x = _conv_bn_relu(x, filters, name=f"{name}b")
    return x


def _decoder_stage(x, skip, filters, name):
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = _conv_bn_relu(x, filters, name=f"{name}a")
    x = _conv_bn_relu(x, filters, name=f"{name}b")
    return x


def build_angle_model():
    """Build UNet with center heatmap + angle/radius regression."""
    inputs = keras.Input(shape=(224, 224, 1), name="image")

    # Encoder (same as v6)
    e1 = _encoder_stage(inputs, 32, "e1", downsample=True)
    e2 = _encoder_stage(e1, 48, "e2")
    e3 = _encoder_stage(e2, 64, "e3")
    e4 = _encoder_stage(e3, 96, "e4")
    b = _encoder_stage(e4, 128, "e5")

    # Decoder (same as v6)
    d1 = _decoder_stage(b, e4, 96, "d1")
    d2 = _decoder_stage(d1, e3, 64, "d2")
    d3 = _decoder_stage(d2, e2, 48, "d3")

    # Shared features
    shared = _conv_bn_relu(d3, 32, name="head")

    # Head 1: Center heatmap (56×56×1, sigmoid)
    center_heatmap = layers.Conv2D(1, 1, padding="same", activation="sigmoid",
                                   name="center_hm")(shared)

    # Head 2: Needle angle as a circular heatmap (360 bins)
    angle_pool = layers.GlobalAveragePooling2D(name="angle_gap")(shared)
    angle_bins = layers.Dense(360, name="angle_dense")(angle_pool)
    angle_bins = layers.Activation("softmax", name="angle_act")(angle_bins)
    angle_bins = layers.RepeatVector(56 * 56, name="angle_repeat")(angle_bins)
    angle_bins = layers.Reshape((56, 56, 360), name="angle_hm")(angle_bins)

    # Head 3: Needle radius (broadcast to spatial dims)
    radius_pool = layers.GlobalAveragePooling2D(name="radius_gap")(shared)
    radius = layers.Dense(1, activation="sigmoid", name="radius_dense")(radius_pool)
    radius = layers.RepeatVector(56 * 56, name="radius_repeat")(radius)
    radius = layers.Reshape((56, 56, 1), name="radius_hm")(radius)

    # Concatenate all outputs: center(1) + angle(360) + radius(1) = 362 channels
    out = layers.Concatenate(name="all_out")([center_heatmap, angle_bins, radius])

    return Model(inputs=inputs, outputs=out, name="keypoint_angle_model")


__all__ = ["build_angle_model"]
