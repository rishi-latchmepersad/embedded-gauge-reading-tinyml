"""SimCC v5 model: SimCC with strong augmentation + L2 regularization.

This is essentially v4 (axis pooling SimCC) but with:
  1. Stronger geometric augmentation (shift, scale, aspect, rotation).
  2. Color jitter (brightness, contrast, noise, blur).
  3. L2 regularization on SimCC heads.
  4. Dropout on SimCC heads.
  5. Slightly larger spatial channels (96 vs 64) for capacity.

The v4 model overfit the 335 training examples (val loss oscillated around 15).
The deployed v2 model (2M params, 128 spatial) trains on the same data and gets
29.86°C MAE on the val set — the val set is genuinely hard. The fix is more
augmentation + regularization, not architecture changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
import tensorflow as tf

INPUT_SIZE = 224
NUM_BINS = 112
SIGMA_BINS = 1.75
SUB_BIN_WIDTH = INPUT_SIZE / NUM_BINS  # 2.0 pixels per bin.


def coord_to_simcc_target(coord_pixels):
    """Convert a coordinate in pixels to a 112-bin Gaussian SimCC target."""
    coord = tf.cast(coord_pixels, tf.float32)
    bins = tf.range(NUM_BINS, dtype=tf.float32)
    center_bin = coord / SUB_BIN_WIDTH
    target = tf.exp(-((bins - center_bin) ** 2) / (2.0 * SIGMA_BINS ** 2))
    target = target / tf.reduce_sum(target)
    return target


def simcc_dfl_loss(pred_logits, target_dist):
    """Cross-entropy loss for SimCC distribution prediction."""
    pred_probs = tf.nn.softmax(pred_logits, axis=-1)
    pred_probs = tf.clip_by_value(pred_probs, 1e-8, 1.0 - 1e-8)
    return tf.reduce_mean(tf.reduce_sum(-target_dist * tf.math.log(pred_probs), axis=-1))


@dataclass
class SimCCv5Config:
    input_size: int = INPUT_SIZE
    alpha: float = 0.35
    spatial_channels: int = 96  # Up from 64 in v4 to add capacity.
    head_units: int = 96
    head_dropout: float = 0.2
    l2_reg: float = 1e-4
    pretrained: bool = True


def _build_mobilenetv2_backbone(inputs, config: SimCCv5Config):
    """Build MobileNetV2 backbone without pooling (keeps spatial features)."""
    backbone = keras.applications.MobileNetV2(
        input_shape=(config.input_size, config.input_size, 3),
        alpha=config.alpha, include_top=False, weights="imagenet",
        pooling=None,
    )
    backbone.trainable = True
    return backbone(inputs)


def _build_spatial_trunk(features, config: SimCCv5Config):
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


def _build_center_head(spatial_features, config: SimCCv5Config):
    """Center detector head: GAP → Dense → 2 outputs (cx, cy sigmoid)."""
    x = keras.layers.GlobalAveragePooling2D(name="center_gap")(spatial_features)
    x = keras.layers.Dense(
        config.head_units, activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(config.l2_reg),
        name="center_dense_1",
    )(x)
    x = keras.layers.Dropout(config.head_dropout, name="center_dropout")(x)
    center_xy = keras.layers.Dense(
        2, activation="sigmoid", name="center_xy",
    )(x)
    return center_xy


def _build_simcc_head(spatial_features, axis: str, num_bins: int, name: str, config: SimCCv5Config):
    """One SimCC head for a single axis with dropout and L2 reg."""
    if axis == "x":
        x = keras.layers.AveragePooling2D(
            pool_size=(spatial_features.shape[1], 1),
            name=f"{name}_pool_x",
        )(spatial_features)
        x = keras.layers.Reshape(
            (x.shape[2], x.shape[3]), name=f"{name}_reshape_x",
        )(x)
    else:
        x = keras.layers.AveragePooling2D(
            pool_size=(1, spatial_features.shape[2]),
            name=f"{name}_pool_y",
        )(spatial_features)
        x = keras.layers.Reshape(
            (x.shape[1], x.shape[3]), name=f"{name}_reshape_y",
        )(x)
    x = keras.layers.Flatten(name=f"{name}_flat")(x)
    x = keras.layers.Dropout(config.head_dropout, name=f"{name}_dropout")(x)
    logits = keras.layers.Dense(
        num_bins,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(config.l2_reg),
        name=f"{name}_logits",
    )(x)
    return logits


def build_simcc_v5_model(config: SimCCv5Config | None = None) -> keras.Model:
    """Build the SimCC v5 model: axis-pool SimCC + center detector + L2 reg."""
    if config is None:
        config = SimCCv5Config()

    inputs = keras.Input(
        shape=(config.input_size, config.input_size, 3), name="input_image",
    )
    backbone_features = _build_mobilenetv2_backbone(inputs, config)
    spatial = _build_spatial_trunk(backbone_features, config)
    center_xy = _build_center_head(spatial, config)
    center_x_logits = _build_simcc_head(spatial, "x", NUM_BINS, "center_x", config)
    center_y_logits = _build_simcc_head(spatial, "y", NUM_BINS, "center_y", config)
    tip_x_logits = _build_simcc_head(spatial, "x", NUM_BINS, "tip_x", config)
    tip_y_logits = _build_simcc_head(spatial, "y", NUM_BINS, "tip_y", config)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "center_xy": center_xy,
            "center_x": center_x_logits,
            "center_y": center_y_logits,
            "tip_x": tip_x_logits,
            "tip_y": tip_y_logits,
        },
        name=f"simcc_v5_mnv2_a{int(config.alpha*100):03d}_sc{config.spatial_channels}",
    )
    return model


def simcc_v5_loss(y_true, y_pred, *, center_xy_weight=1.0, simcc_weight=1.0):
    """Combined loss: center_xy MSE + 4 SimCC DFL losses."""
    center_loss = tf.reduce_mean(
        tf.square(y_pred["center_xy"] - y_true["center_xy"])
    )
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
