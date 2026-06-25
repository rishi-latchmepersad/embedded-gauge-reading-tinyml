"""SimCC v2 model for gauge keypoint detection — for the OBB pipeline.

Takes the OBB-cropped gauge region (224x224) and finds:
  - center: where the needle pivots
  - tip: where the needle points

Architecture (efficient, NPU-friendly):

  - Backbone: MobileNetV2-Small (α=0.35) with GAP.
  - Shared bottleneck: Dense(96) → Dense(48).
  - 4 SimCC heads: center_x, center_y, tip_x, tip_y.
  - Confidence head: Dense(1, sigmoid).

SimCC (Simple Coordinate Classification):
  - Each coordinate is predicted as a 1D softmax over NUM_BINS bins.
  - Soft-argmax gives sub-bin accuracy.
  - Uses DFL (Distribution Focal Loss) for training.

Target: <1.5 MB INT8 TFLite, accurate to ~0.5 pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import keras
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

INPUT_SIZE = 224
NUM_BINS = 112  # Spatial bins per axis (each bin = 2 pixels on 224x224).
SUB_BIN_WIDTH = INPUT_SIZE / NUM_BINS  # 2.0 pixels per bin.
TOTAL_BINS = NUM_BINS  # 112 bins per coordinate (no sub-bins for simplicity).


@dataclass
class SimCCConfig:
    input_size: int = INPUT_SIZE
    alpha: float = 0.35  # MobileNetV2 width multiplier.
    num_bins: int = NUM_BINS
    bottleneck_units: int = 96
    pretrained: bool = True


# ──────────────────────────────────────────────────────────────────────
# SimCC utilities
# ──────────────────────────────────────────────────────────────────────

def coord_to_simcc_target(coord_pixels: tf.Tensor) -> tf.Tensor:
    """Convert a coordinate in pixels [0, 224) to a SimCC soft target.

    The target is a 112-bin distribution with the peak at the
    corresponding bin and a small spread for sub-bin accuracy.
    """
    coord = tf.cast(coord_pixels, tf.float32)
    # Map pixel to bin index: bin = coord / 2.0 (each bin is 2 pixels).
    bin_float = coord / SUB_BIN_WIDTH  # in [0, 112)
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
    total = tf.reduce_sum(target)
    return target / total


def simcc_to_coord(logits: tf.Tensor) -> tf.Tensor:
    """Convert SimCC logits (112 bins) back to coordinate in pixels."""
    probs = tf.nn.softmax(logits, axis=-1)
    # Bin centers in pixels: bin i covers [(i-0.5)*2, (i+0.5)*2].
    bin_centers = (tf.range(NUM_BINS, dtype=tf.float32) + 0.5) * SUB_BIN_WIDTH
    return tf.reduce_sum(probs * bin_centers, axis=-1)


# ──────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────

def simcc_dfl_loss(
    pred_logits: tf.Tensor, target_dist: tf.Tensor,
) -> tf.Tensor:
    """Distribution Focal Loss for SimCC coordinate prediction."""
    pred_probs = tf.nn.softmax(pred_logits, axis=-1)
    pred_probs = tf.clip_by_value(pred_probs, 1e-8, 1.0 - 1e-8)
    ce = -target_dist * tf.math.log(pred_probs)
    return tf.reduce_mean(tf.reduce_sum(ce, axis=-1))


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────

def build_simcc_model(config: SimCCConfig | None = None) -> keras.Model:
    """Build the SimCC v2 model for gauge keypoint detection.

    Architecture:
      Input (224, 224, 3)
        → MobileNetV2 (α=0.35, pretrained, GAP)
        → Shared bottleneck Dense(96) → Dense(48)
        → 4 SimCC heads (center_x, center_y, tip_x, tip_y) — each Dense(112)
        → Confidence head (Dense(1, sigmoid))
    """
    if config is None:
        config = SimCCConfig()

    inputs = keras.Input(
        shape=(config.input_size, config.input_size, 3), name="input_image",
    )

    # ── Backbone ───────────────────────────────────────────────────────
    backbone = keras.applications.MobileNetV2(
        input_shape=(config.input_size, config.input_size, 3),
        alpha=config.alpha, include_top=False, weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = True
    features = backbone(inputs)  # (B, 1280*alpha)

    # ── Shared bottleneck ─────────────────────────────────────────────
    shared = keras.layers.Dense(
        config.bottleneck_units,
        activation="relu", kernel_initializer="he_normal",
        name="simcc_bottleneck_1",
    )(features)
    shared = keras.layers.Dropout(0.1, name="simcc_dropout_1")(shared)
    shared = keras.layers.Dense(
        config.bottleneck_units // 2,
        activation="relu", kernel_initializer="he_normal",
        name="simcc_bottleneck_2",
    )(shared)

    # ── SimCC heads ────────────────────────────────────────────────────
    def _simcc_head(x, name):
        return keras.layers.Dense(
            config.num_bins, name=f"{name}_logits",
        )(x)

    center_x_logits = _simcc_head(shared, "center_x")
    center_y_logits = _simcc_head(shared, "center_y")
    tip_x_logits = _simcc_head(shared, "tip_x")
    tip_y_logits = _simcc_head(shared, "tip_y")

    # ── Confidence ────────────────────────────────────────────────────
    confidence = keras.layers.Dense(
        1, activation="sigmoid", name="confidence",
    )(shared)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "center_x": center_x_logits,
            "center_y": center_y_logits,
            "tip_x": tip_x_logits,
            "tip_y": tip_y_logits,
            "confidence": confidence,
        },
        name=f"simcc_v2_mnv2_a{int(config.alpha*100):03d}_bins{config.num_bins}",
    )
    return model


def simcc_loss(
    y_true: dict, y_pred: dict,
    *,
    coord_weight: float = 1.0,
    conf_weight: float = 0.5,
) -> dict:
    """Combined SimCC loss: 4 DFL losses + binary cross-entropy for confidence."""
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
