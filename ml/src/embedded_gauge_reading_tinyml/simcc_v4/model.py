"""SimCC v4 model for gauge needle keypoint detection.

This is the keypoint model that runs after the OBB detector.
It takes a 224x224 crop of the gauge face and predicts:
  - Center: (cx, cy) normalized [0, 1]
  - Tip: (tx, ty) normalized [0, 1]

Architecture follows the working deployed model
(build_mobilenetv2_center_simcc_model) but adapted for our budget.

Key design choices (from the working deployed model):
  1. Joint center detector + SimCC heads sharing a spatial trunk.
     The center detector (GAP → Dense) provides a strong geometric prior,
     then the SimCC heads only need to find the needle within the gauge face.

  2. Gaussian soft SimCC targets (sigma_bins=1.75) instead of 2-bin split.
     This spreads the target across ~5 bins, providing smoother gradients
     and much better learnability.

  3. Spatial trunk: MobileNetV2 features → 1x1 proj → 2x upsample → 3x3 convs.
     Preserves 14x14 spatial information needed for SimCC.

Architecture:
  Input (224, 224, 3)
    → MobileNetV2-Small (α=0.35, pretrained, no pooling) → 7x7xC
    → Spatial trunk: 1x1 conv → upsample 2x → 3x3 conv → 3x3 conv → 14x14x64
    → Two branches:
        a) Center head: GAP → Dense(96) → Dense(2, sigmoid) for (cx, cy)
        b) SimCC heads: axis-pool → Flatten → Dense(112) for each of
           center_x, center_y, tip_x, tip_y
    → Output: {center_xy (2), center_x_simcc (112), center_y_simcc (112),
              tip_x_simcc (112), tip_y_simcc (112)}

Target: ~1.2-1.5 MB INT8 (under the 1.6 MB remaining budget).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import keras
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

INPUT_SIZE = 224
NUM_BINS = 112
SIGMA_BINS = 1.75  # Gaussian sigma in bins — from the working deployed model.
SUB_BIN_WIDTH = INPUT_SIZE / NUM_BINS  # 2.0 pixels per bin.


# ──────────────────────────────────────────────────────────────────────
# SimCC target generation (Gaussian soft target)
# ──────────────────────────────────────────────────────────────────────

def coord_to_simcc_target(coord_pixels):
    """Convert a coordinate in pixels to a 112-bin Gaussian SimCC target.

    Uses a Gaussian distribution with sigma=1.75 bins centered on the
    target bin, providing smooth gradients for the model to learn from.
    """
    coord = tf.cast(coord_pixels, tf.float32)
    bins = tf.range(NUM_BINS, dtype=tf.float32)
    # Map pixel to bin center.
    center_bin = coord / SUB_BIN_WIDTH
    # Gaussian distribution.
    target = tf.exp(-((bins - center_bin) ** 2) / (2.0 * SIGMA_BINS ** 2))
    # Normalise to sum to 1.
    total = tf.reduce_sum(target)
    return target / total


def simcc_to_coord(logits):
    """Convert SimCC logits (112 bins) back to coordinate in pixels."""
    probs = tf.nn.softmax(logits, axis=-1)
    bin_centers = (tf.range(NUM_BINS, dtype=tf.float32) + 0.5) * SUB_BIN_WIDTH
    return tf.reduce_sum(probs * bin_centers, axis=-1)


def simcc_dfl_loss(pred_logits, target_dist):
    """Cross-entropy loss for SimCC distribution prediction."""
    pred_probs = tf.nn.softmax(pred_logits, axis=-1)
    pred_probs = tf.clip_by_value(pred_probs, 1e-8, 1.0 - 1e-8)
    return tf.reduce_mean(tf.reduce_sum(-target_dist * tf.math.log(pred_probs), axis=-1))


# ──────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SimCCv4Config:
    input_size: int = INPUT_SIZE
    alpha: float = 0.35
    num_bins: int = NUM_BINS
    spatial_channels: int = 64
    head_units: int = 96
    head_dropout: float = 0.15
    pretrained: bool = True


def _build_mobilenetv2_backbone(inputs, config: SimCCv4Config):
    """Build MobileNetV2 backbone without pooling (keeps spatial features)."""
    backbone = keras.applications.MobileNetV2(
        input_shape=(config.input_size, config.input_size, 3),
        alpha=config.alpha, include_top=False, weights="imagenet",
        pooling=None,
    )
    backbone.trainable = True
    return backbone(inputs)  # 7x7xC


def _build_spatial_trunk(features, config: SimCCv4Config):
    """Build the shared spatial trunk (upsample + convs)."""
    x = keras.layers.Conv2D(
        config.spatial_channels, 1, padding="same",
        activation="relu", kernel_initializer="he_normal",
        name="shared_spatial_proj",
    )(features)
    x = keras.layers.UpSampling2D(
        size=2, interpolation="bilinear", name="shared_spatial_up",
    )(x)
    x = keras.layers.Conv2D(
        config.spatial_channels, 3, padding="same",
        activation="relu", kernel_initializer="he_normal",
        name="shared_spatial_conv_1",
    )(x)
    x = keras.layers.Conv2D(
        config.spatial_channels, 3, padding="same",
        activation="relu", kernel_initializer="he_normal",
        name="shared_spatial_conv_2",
    )(x)
    return x


def _build_center_head(spatial_features, config: SimCCv4Config):
    """Center detector head: GAP → Dense → 2 outputs (cx, cy sigmoid)."""
    x = keras.layers.GlobalAveragePooling2D(name="center_gap")(spatial_features)
    x = keras.layers.Dense(
        config.head_units, activation="relu",
        kernel_initializer="he_normal", name="center_dense_1",
    )(x)
    x = keras.layers.Dropout(config.head_dropout, name="center_dropout")(x)
    center_xy = keras.layers.Dense(
        2, activation="sigmoid", name="center_xy",
    )(x)
    return center_xy


def _build_simcc_head(spatial_features, axis: str, num_bins: int, name: str):
    """One SimCC head for a single axis.

    Pools the spatial trunk along the other axis to get a 1D feature,
    then projects to num_bins.
    """
    if axis == "x":
        # Pool over H to get (B, W, C).
        x = keras.layers.AveragePooling2D(
            pool_size=(spatial_features.shape[1], 1),
            name=f"{name}_pool_x",
        )(spatial_features)
        x = keras.layers.Reshape(
            (x.shape[2], x.shape[3]), name=f"{name}_reshape_x",
        )(x)
    else:
        # Pool over W to get (B, H, C).
        x = keras.layers.AveragePooling2D(
            pool_size=(1, spatial_features.shape[2]),
            name=f"{name}_pool_y",
        )(spatial_features)
        x = keras.layers.Reshape(
            (x.shape[1], x.shape[3]), name=f"{name}_reshape_y",
        )(x)
    x = keras.layers.Flatten(name=f"{name}_flat")(x)
    logits = keras.layers.Dense(num_bins, name=f"{name}_logits")(x)
    return logits


def build_simcc_v4_model(config: SimCCv4Config | None = None) -> keras.Model:
    """Build the SimCC v4 model: center detector + SimCC heads.

    Architecture:
      Input (224, 224, 3)
        → MobileNetV2 (α=0.35, pretrained, no pooling) → 7x7xC
        → Spatial trunk → 14x14x64
        → Center head: GAP → Dense(96) → Dropout → Dense(2, sigmoid) = (cx, cy)
        → SimCC heads: axis-pool + Dense(112) for center_x/y, tip_x/y
    """
    if config is None:
        config = SimCCv4Config()

    inputs = keras.Input(
        shape=(config.input_size, config.input_size, 3), name="input_image",
    )

    # Backbone.
    backbone_features = _build_mobilenetv2_backbone(inputs, config)
    # Spatial trunk.
    spatial = _build_spatial_trunk(backbone_features, config)
    # Center detector head.
    center_xy = _build_center_head(spatial, config)
    # SimCC heads.
    center_x_logits = _build_simcc_head(spatial, "x", config.num_bins, "center_x")
    center_y_logits = _build_simcc_head(spatial, "y", config.num_bins, "center_y")
    tip_x_logits = _build_simcc_head(spatial, "x", config.num_bins, "tip_x")
    tip_y_logits = _build_simcc_head(spatial, "y", config.num_bins, "tip_y")

    model = keras.Model(
        inputs=inputs,
        outputs={
            "center_xy": center_xy,
            "center_x": center_x_logits,
            "center_y": center_y_logits,
            "tip_x": tip_x_logits,
            "tip_y": tip_y_logits,
        },
        name=f"simcc_v4_mnv2_a{int(config.alpha*100):03d}_bins{config.num_bins}",
    )
    return model


# ──────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────

def simcc_v4_loss(
    y_true, y_pred, *,
    center_xy_weight=1.0,
    simcc_weight=1.0,
):
    """Combined loss: center_xy MSE + 4 SimCC DFL losses."""
    # Center detector: MSE between predicted (cx,cy) and GT.
    center_loss = tf.reduce_mean(
        tf.square(y_pred["center_xy"] - y_true["center_xy"])
    )
    # SimCC heads: DFL loss for each of 4 coordinate heads.
    cx_loss = simcc_dfl_loss(y_pred["center_x"], y_true["center_x"])
    cy_loss = simcc_dfl_loss(y_pred["center_y"], y_true["center_y"])
    tx_loss = simcc_dfl_loss(y_pred["tip_x"], y_true["tip_x"])
    ty_loss = simcc_dfl_loss(y_pred["tip_y"], y_true["tip_y"])
    simcc_loss_val = cx_loss + cy_loss + tx_loss + ty_loss

    total = center_xy_weight * center_loss + simcc_weight * simcc_loss_val
    return {
        "loss": total,
        "center_loss": center_loss,
        "simcc_loss": simcc_loss_val,
    }
