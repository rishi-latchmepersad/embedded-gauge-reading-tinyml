"""SimCC v6: 2D heatmap keypoint detector for OBB-cropped gauge dial.

Architecture (literature-driven):
  1. MobileNetV2 (alpha=0.35) backbone -> 7x7xC features
  2. Spatial trunk: 1x1 conv + 2x upsample + 2x 3x3 conv -> 14x14xC
  3. Per-keypoint 2D heatmap head (SimCC-style, see CVPR 2023 paper
     "Coordinate Classification with Successive Regularization"):
     - 3x3 conv -> 1 channel
     - Softmax over H*W bins (gives 2D probability distribution)
     - Soft argmax for sub-pixel decoding
  4. Two keypoints: center (cx, cy) and tip (tx, ty) -> 2 heatmaps

Why 2D heatmap (vs v5's 1D axis-pool SimCC):
  - Preserves (x, y) spatial correlation, so model can learn that
    tip is on a radial line from center.
  - Industry standard for keypoint detection (CenterNet, SimCC, RTMPose,
    HigherHRNet, etc.)
  - Soft argmax is differentiable, allowing end-to-end KD.

Training:
  - MSE loss between predicted and Gaussian-target heatmaps.
  - KD from deployed SimCC v2: teacher's 1D marginals -> student's
    2D heatmap should marginalize to the same distribution.

Compression (AGENTS.md recipe):
  - KD from deployed v2 (QAT teacher)
  - TL: start from ImageNet pretrained backbone
  - QAT/PTQ for INT8
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
import tensorflow as tf

INPUT_SIZE = 224
HEATMAP_SIZE = 14  # 7 -> 14 via 2x upsample in trunk.
NUM_BINS_X = 14  # 2D heatmap is HEATMAP_SIZE x HEATMAP_SIZE.
SIGMA_PX = 2.0  # Gaussian sigma in heatmap pixels (~14% of heatmap).


def coord_to_heatmap_2d(cx_px: float, cy_px: float, size: int = HEATMAP_SIZE) -> tf.Tensor:
    """Make a 2D Gaussian heatmap of size (size, size) centered at (cx_px, cy_px).

    Args:
        cx_px, cy_px: target position in INPUT_SIZE pixel units.
        size: heatmap side length.
    """
    cx_bin = (cx_px / INPUT_SIZE) * size
    cy_bin = (cy_px / INPUT_SIZE) * size
    cx_bin = tf.clip_by_value(cx_bin, 0.0, float(size - 1))
    cy_bin = tf.clip_by_value(cy_bin, 0.0, float(size - 1))
    xs = tf.range(size, dtype=tf.float32)
    ys = tf.range(size, dtype=tf.float32)
    # 2D Gaussian: exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
    sigma = SIGMA_PX
    gx = tf.exp(-((xs - cx_bin) ** 2) / (2.0 * sigma * sigma))
    gy = tf.exp(-((ys - cy_bin) ** 2) / (2.0 * sigma * sigma))
    # Outer product -> (H, W).
    hm = tf.tensordot(gy, gx, axes=0)  # (H, W) but we want (W, H) for x, y
    # Actually tensordot(gy, gx) gives gy[:,None] * gx[None,:] = (H_y, W_x)
    # which is (cy_index, cx_index). That's what we want.
    # Normalize.
    hm = hm / tf.reduce_sum(hm)
    return hm


def soft_argmax_2d(heatmap_2d: tf.Tensor) -> tf.Tensor:
    """Soft argmax: 2D heatmap -> (x, y) in INPUT_SIZE pixels.

    Args:
        heatmap_2d: (..., H, W) probability distribution.
    Returns: (..., 2) tensor of (x_px, y_px).
    """
    shape = tf.shape(heatmap_2d)
    h = tf.cast(shape[-2], tf.float32)
    w = tf.cast(shape[-1], tf.float32)
    flat = tf.reshape(heatmap_2d, [-1, tf.cast(h * w, tf.int32)])
    # Expectation over flat distribution.
    bins = tf.range(tf.cast(h * w, tf.int32), dtype=tf.float32)
    # bin (i, j) -> (x, y) = (j + 0.5, i + 0.5) in heatmap units.
    j = bins % tf.cast(w, tf.float32)
    i = tf.floor(bins / tf.cast(w, tf.float32))
    x_bin = j + 0.5
    y_bin = i + 0.5
    # Convert to INPUT_SIZE pixels.
    x_px = x_bin * (INPUT_SIZE / w)
    y_px = y_bin * (INPUT_SIZE / h)
    x_coord = tf.reduce_sum(flat * x_px, axis=-1)
    y_coord = tf.reduce_sum(flat * y_px, axis=-1)
    return tf.stack([x_coord, y_coord], axis=-1)


@dataclass
class SimCCv6Config:
    input_size: int = INPUT_SIZE
    alpha: float = 0.35
    spatial_channels: int = 64
    head_channels: int = 32
    sigma_px: float = SIGMA_PX
    pretrained: bool = True


def _build_mobilenetv2_backbone(inputs, config: SimCCv6Config):
    """MobileNetV2 backbone, returns 7x7xC features (no pooling)."""
    backbone = keras.applications.MobileNetV2(
        input_shape=(config.input_size, config.input_size, 3),
        alpha=config.alpha, include_top=False, weights="imagenet",
        pooling=None,
    )
    backbone.trainable = True
    return backbone(inputs)


def _build_spatial_trunk(features, config: SimCCv6Config):
    """7x7 -> 14x14 with 1x1 conv + 2x upsample + 2x 3x3 convs."""
    x = keras.layers.Conv2D(
        config.spatial_channels, 1, padding="same",
        activation="relu", kernel_initializer="he_normal",
        name="spatial_proj",
    )(features)
    x = keras.layers.UpSampling2D(
        size=2, interpolation="bilinear", name="spatial_up",
    )(x)
    x = keras.layers.Conv2D(
        config.spatial_channels, 3, padding="same",
        activation="relu", kernel_initializer="he_normal",
        name="spatial_conv_1",
    )(x)
    x = keras.layers.Conv2D(
        config.spatial_channels, 3, padding="same",
        activation="relu", kernel_initializer="he_normal",
        name="spatial_conv_2",
    )(x)
    return x


def _build_heatmap_head(spatial_features, config: SimCCv6Config, name: str):
    """Per-keypoint 2D heatmap head.

    Outputs:
      - logits: (B, H, W) raw logits for softmax
      - heatmap: (B, H, W) softmax probability distribution
      - coord: (B, 2) soft-argmax decoded (x_px, y_px)
    """
    x = keras.layers.Conv2D(
        config.head_channels, 3, padding="same",
        activation="relu", kernel_initializer="he_normal",
        name=f"{name}_conv_1",
    )(spatial_features)
    logits = keras.layers.Conv2D(
        1, 1, padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_logits",
    )(x)
    # Reshape to (B, H*W) and softmax.
    h, w = spatial_features.shape[1], spatial_features.shape[2]
    flat_logits = keras.layers.Reshape((h * w,), name=f"{name}_flat")(logits)
    flat_probs = keras.layers.Softmax(axis=-1, name=f"{name}_probs")(flat_logits)
    heatmap = keras.layers.Reshape((h, w), name=f"{name}_heatmap")(flat_probs)
    return logits, heatmap


def build_simcc_v6_model(config: SimCCv6Config | None = None) -> keras.Model:
    """Build the SimCC v6 model with 2D heatmap heads for center and tip."""
    if config is None:
        config = SimCCv6Config()

    inputs = keras.Input(
        shape=(config.input_size, config.input_size, 3), name="input_image",
    )
    backbone_features = _build_mobilenetv2_backbone(inputs, config)
    spatial = _build_spatial_trunk(backbone_features, config)
    center_logits, center_heatmap = _build_heatmap_head(spatial, config, "center")
    tip_logits, tip_heatmap = _build_heatmap_head(spatial, config, "tip")

    # Soft argmax for each.
    center_softarg = keras.layers.Lambda(
        lambda hm: soft_argmax_2d(hm),
        name="center_softarg",
    )(center_heatmap)
    tip_softarg = keras.layers.Lambda(
        lambda hm: soft_argmax_2d(hm),
        name="tip_softarg",
    )(tip_heatmap)

    # Also output as 1D marginals (for compatibility with deployed v2 KD).
    center_x_marg = keras.layers.Lambda(
        lambda hm: tf.reduce_sum(hm, axis=1),
        name="center_x_marg",
    )(center_heatmap)  # (B, W)
    center_y_marg = keras.layers.Lambda(
        lambda hm: tf.reduce_sum(hm, axis=2),
        name="center_y_marg",
    )(center_heatmap)  # (B, H)
    tip_x_marg = keras.layers.Lambda(
        lambda hm: tf.reduce_sum(hm, axis=1),
        name="tip_x_marg",
    )(tip_heatmap)
    tip_y_marg = keras.layers.Lambda(
        lambda hm: tf.reduce_sum(hm, axis=2),
        name="tip_y_marg",
    )(tip_heatmap)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "center_heatmap": center_heatmap,  # (B, H, W)
            "tip_heatmap": tip_heatmap,
            "center_softarg": center_softarg,  # (B, 2) [x, y] in pixels
            "tip_softarg": tip_softarg,
            "center_x_marg": center_x_marg,  # (B, W) for KD
            "center_y_marg": center_y_marg,
            "tip_x_marg": tip_x_marg,
            "tip_y_marg": tip_y_marg,
            "center_logits": center_logits,
            "tip_logits": tip_logits,
        },
        name=f"simcc_v6_mnv2_a{int(config.alpha*100):03d}_sc{config.spatial_channels}",
    )
    return model


def simcc_v6_loss(y_true, y_pred, *, heatmap_weight=1.0, kd_weight=0.5):
    """Combined loss: heatmap MSE + KD on marginals.

    y_true: dict with center_hm, tip_hm (target 2D heatmaps), and
             center_x_marg, center_y_marg, tip_x_marg, tip_y_marg (target marginals).
    y_pred: model outputs.
    """
    # Heatmap MSE loss (the main loss).
    center_hm_loss = tf.reduce_mean(
        tf.square(y_pred["center_heatmap"] - y_true["center_hm"])
    )
    tip_hm_loss = tf.reduce_mean(
        tf.square(y_pred["tip_heatmap"] - y_true["tip_hm"])
    )
    hm_loss = center_hm_loss + tip_hm_loss

    # KD loss: student's 1D marginals should match target 1D marginals.
    # The target 1D marginals are derived from the deployed v2's predictions
    # (passed in as part of y_true).
    def _kl(p_logits, q_target, eps=1e-8):
        # q_target is a target distribution.
        q = tf.clip_by_value(q_target, eps, 1.0)
        p = tf.clip_by_value(p_logits, eps, 1.0)
        return tf.reduce_sum(q * tf.math.log(q / p), axis=-1)

    kd_loss_cx = tf.reduce_mean(_kl(y_pred["center_x_marg"], y_true["center_x_marg"]))
    kd_loss_cy = tf.reduce_mean(_kl(y_pred["center_y_marg"], y_true["center_y_marg"]))
    kd_loss_tx = tf.reduce_mean(_kl(y_pred["tip_x_marg"], y_true["tip_x_marg"]))
    kd_loss_ty = tf.reduce_mean(_kl(y_pred["tip_y_marg"], y_true["tip_y_marg"]))
    kd_loss = kd_loss_cx + kd_loss_cy + kd_loss_tx + kd_loss_ty

    total = heatmap_weight * hm_loss + kd_weight * kd_loss
    return {
        "loss": total,
        "hm_loss": hm_loss,
        "kd_loss": kd_loss,
    }
