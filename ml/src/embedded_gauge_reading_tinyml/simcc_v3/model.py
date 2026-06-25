"""SimCC v3 model for gauge keypoint detection — with spatial trunk.

Fix from v2: v2 used GAP which destroyed all spatial information.
SimCC needs spatial awareness to know WHERE the keypoint is.

v3 architecture:
  - Backbone: MobileNetV2-Small (α=0.35) WITHOUT pooling
  - Spatial trunk: 14x14xC -> 28x28xC (lightweight upsampling)
  - 4 SimCC heads: pool along one axis, then Dense(112)
  - Confidence head: GAP -> Dense(1, sigmoid)

This is the standard SimCC design (Li et al. 2022) adapted for embedded.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import keras
import tensorflow as tf

INPUT_SIZE = 224
NUM_BINS = 112
SUB_BIN_WIDTH = INPUT_SIZE / NUM_BINS  # 2.0 pixels per bin.


@dataclass
class SimCCConfig:
    input_size: int = INPUT_SIZE
    alpha: float = 0.35
    num_bins: int = NUM_BINS
    spatial_channels: int = 64
    pretrained: bool = True


def coord_to_simcc_target(coord_pixels):
    coord = tf.cast(coord_pixels, tf.float32)
    bin_float = coord / SUB_BIN_WIDTH
    left = tf.floor(bin_float)
    right = left + 1.0
    weight_right = bin_float - left
    weight_left = 1.0 - weight_right
    left = tf.cast(tf.clip_by_value(left, 0, NUM_BINS - 1), tf.int32)
    right = tf.cast(tf.clip_by_value(right, 0, NUM_BINS - 1), tf.int32)
    target = tf.zeros([NUM_BINS], dtype=tf.float32)
    target = tf.tensor_scatter_nd_add(
        target, tf.stack([left, right])[:, tf.newaxis],
        tf.stack([weight_left, weight_right]),
    )
    return target / tf.reduce_sum(target)


def simcc_to_coord(logits):
    probs = tf.nn.softmax(logits, axis=-1)
    bin_centers = (tf.range(NUM_BINS, dtype=tf.float32) + 0.5) * SUB_BIN_WIDTH
    return tf.reduce_sum(probs * bin_centers, axis=-1)


def simcc_dfl_loss(pred_logits, target_dist):
    pred_probs = tf.nn.softmax(pred_logits, axis=-1)
    pred_probs = tf.clip_by_value(pred_probs, 1e-8, 1.0 - 1e-8)
    ce = -target_dist * tf.math.log(pred_probs)
    return tf.reduce_mean(tf.reduce_sum(ce, axis=-1))


def _axis_pool(x, axis, name):
    """Pool trunk along the other axis to get 1D feature."""
    if axis == "x":
        # Pool over H -> (B, W, C).
        x = keras.layers.AveragePooling2D(
            pool_size=(x.shape[1], 1), name=f"{name}_pool_x",
        )(x)
        x = keras.layers.Reshape((x.shape[2], x.shape[3]), name=f"{name}_reshape_x")(x)
    else:
        # Pool over W -> (B, H, C).
        x = keras.layers.AveragePooling2D(
            pool_size=(1, x.shape[2]), name=f"{name}_pool_y",
        )(x)
        x = keras.layers.Reshape((x.shape[1], x.shape[3]), name=f"{name}_reshape_y")(x)
    return x


def _axis_head(x1d, num_bins, name):
    """Flatten 1D feature and project to num_bins."""
    x1d = keras.layers.Flatten(name=f"{name}_flat")(x1d)
    logits = keras.layers.Dense(num_bins, name=f"{name}_logits")(x1d)
    return logits


def build_simcc_v3_model(config: SimCCConfig | None = None) -> keras.Model:
    """Build the SimCC v3 model with spatial trunk.

    Architecture:
      Input (224, 224, 3)
        → MobileNetV2 (α=0.35, pretrained, no pooling) → 7x7x1280
        → UpSample 2x → 14x14x1280
        → 1x1 Conv (spatial_channels) → 14x14x64
        → 3x3 Conv → 14x14x64 (spatial trunk)
        → 4 axis pools: pool_x for x-axes, pool_y for y-axes
        → 4 SimCC heads: center_x, center_y, tip_x, tip_y
        → Confidence: GAP → Dense(1, sigmoid)
    """
    if config is None:
        config = SimCCConfig()

    inputs = keras.Input(
        shape=(config.input_size, config.input_size, 3), name="input_image",
    )

    # ── Backbone (no pooling, keep spatial) ─────────────────────────
    backbone = keras.applications.MobileNetV2(
        input_shape=(config.input_size, config.input_size, 3),
        alpha=config.alpha, include_top=False, weights="imagenet",
        pooling=None,
    )
    backbone.trainable = True
    backbone_out = backbone(inputs)  # 7x7xC (C = 1280*alpha)

    # ── Spatial trunk: upsample to 14x14 ─────────────────────────────
    spatial = keras.layers.UpSampling2D(
        size=2, interpolation="bilinear", name="spatial_upsample",
    )(backbone_out)
    spatial = keras.layers.Conv2D(
        config.spatial_channels, 1, padding="same",
        activation="relu", kernel_initializer="he_normal",
        name="spatial_proj",
    )(spatial)
    spatial = keras.layers.Conv2D(
        config.spatial_channels, 3, padding="same",
        activation="relu", kernel_initializer="he_normal",
        name="spatial_conv1",
    )(spatial)
    spatial = keras.layers.Conv2D(
        config.spatial_channels, 3, padding="same",
        activation="relu", kernel_initializer="he_normal",
        name="spatial_conv2",
    )(spatial)

    # ── Axis pools for SimCC heads ───────────────────────────────────
    trunk_x = _axis_pool(spatial, "x", "trunk_x")
    trunk_y = _axis_pool(spatial, "y", "trunk_y")

    # ── SimCC heads ───────────────────────────────────────────────────
    center_x_logits = _axis_head(trunk_x, config.num_bins, "center_x")
    center_y_logits = _axis_head(trunk_y, config.num_bins, "center_y")
    tip_x_logits = _axis_head(trunk_x, config.num_bins, "tip_x")
    tip_y_logits = _axis_head(trunk_y, config.num_bins, "tip_y")

    # ── Confidence head (GAP) ────────────────────────────────────────
    gap = keras.layers.GlobalAveragePooling2D(name="conf_gap")(spatial)
    confidence = keras.layers.Dense(
        1, activation="sigmoid", name="confidence",
    )(gap)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "center_x": center_x_logits,
            "center_y": center_y_logits,
            "tip_x": tip_x_logits,
            "tip_y": tip_y_logits,
            "confidence": confidence,
        },
        name=f"simcc_v3_mnv2_a{int(config.alpha*100):03d}_bins{config.num_bins}",
    )
    return model


def simcc_loss(y_true, y_pred, *, coord_weight=1.0, conf_weight=0.5):
    cx_loss = simcc_dfl_loss(y_pred["center_x"], y_true["center_x"])
    cy_loss = simcc_dfl_loss(y_pred["center_y"], y_true["center_y"])
    tx_loss = simcc_dfl_loss(y_pred["tip_x"], y_true["tip_x"])
    ty_loss = simcc_dfl_loss(y_pred["tip_y"], y_true["tip_y"])
    conf_loss = tf.reduce_mean(
        tf.keras.backend.binary_crossentropy(
            y_true["confidence"], y_pred["confidence"],
        )
    )
    coord_loss = cx_loss + cy_loss + tx_loss + ty_loss
    total = coord_weight * coord_loss + conf_weight * conf_loss
    return {
        "loss": total,
        "coord_loss": coord_loss,
        "conf_loss": conf_loss,
    }
