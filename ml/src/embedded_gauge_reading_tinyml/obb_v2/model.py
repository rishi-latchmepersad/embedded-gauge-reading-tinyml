"""OBB v2 model for embedded gauge detection — simplified for circular gauges.

Key design decision: A circular gauge is rotationally symmetric, so predicting
its rotation angle is ill-posed. The model outputs:
  - box: [cx, cy, w, h] in [0, 1] — axis-aligned bounding box
  - confidence: scalar (sigmoid) — gauge is present

No angle prediction. This dramatically simplifies the model and training.

Architecture follows 2024-2025 SOTA for embedded object detection:

  - Backbone: MobileNetV3-Small (α=0.75) — efficient, NPU-friendly.
  - Neck: Lite-FPN (P3 + P4 fusion) — multi-scale for different gauge sizes.
  - Head: Decoupled box + confidence branches.
  - Input: 224×224×3 (matches existing OBB deployment).
  - Output: 4 box params + 1 confidence = 5 params total.

Why no angle?
  A round gauge looks identical at any rotation. The only "angle" signal
  comes from the numbers/text on the dial, which is weak and noisy.
  Trying to predict it just adds noise to the training signal.
  If the camera is mounted at an angle, the gauge appears as an ellipse
  in the frame, but its bounding box is still axis-aligned in the image.

Target: <1.25 MB peak activation, <500 KB INT8 TFLite.
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
import tensorflow as tf

INPUT_SIZE = 224


@dataclass
class OBBConfig:
    """Configuration for the OBB model."""
    input_size: int = INPUT_SIZE
    alpha: float = 0.75  # MobileNetV3-Small width multiplier.
    fpn_channels: int = 64
    pretrained: bool = True
    backbone_trainable: bool = True


def _conv_bn_act(x, filters, kernel=3, stride=1, name=""):
    """Conv2D + BatchNorm + ReLU6 (standard mobile block)."""
    x = keras.layers.Conv2D(
        filters, kernel, strides=stride, padding="same",
        use_bias=False, name=f"{name}_conv",
    )(x)
    x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = keras.layers.ReLU(6.0, name=f"{name}_relu")(x)
    return x


def _depthwise_separable_conv(x, filters, stride=1, name=""):
    """Depthwise-separable conv: cheaper than standard conv."""
    x = keras.layers.DepthwiseConv2D(
        3, strides=stride, padding="same", use_bias=False,
        name=f"{name}_dw",
    )(x)
    x = keras.layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = keras.layers.ReLU(6.0, name=f"{name}_dw_relu")(x)
    x = keras.layers.Conv2D(
        filters, 1, padding="same", use_bias=False, name=f"{name}_pw",
    )(x)
    x = keras.layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = keras.layers.ReLU(6.0, name=f"{name}_pw_relu")(x)
    return x


def _lite_fpn_neck(c3, c4, out_channels=64, name="fpn"):
    """Lite-FPN: fuse P3 and P4 features with depthwise-separable convs."""
    c4_up = keras.layers.UpSampling2D(2, name=f"{name}_c4_up")(c4)
    c4_up = _conv_bn_act(c4_up, out_channels, kernel=1, name=f"{name}_c4_proj")
    c3_proj = _conv_bn_act(c3, out_channels, kernel=1, name=f"{name}_c3_proj")
    fused = keras.layers.Add(name=f"{name}_add")([c3_proj, c4_up])
    fused = _depthwise_separable_conv(fused, out_channels, name=f"{name}_fuse")
    return fused


def _box_head(features, name="box_head"):
    """Decoupled head: box regression + confidence.

    Returns dict with:
      conf: (B, 1) sigmoid confidence.
      box:  (B, 4) [cx, cy, w, h] in [0, 1].
    """
    x = _depthwise_separable_conv(features, 64, name=f"{name}_trunk_1")
    x = _depthwise_separable_conv(x, 64, name=f"{name}_trunk_2")
    gap = keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)

    conf = keras.layers.Dense(1, activation="sigmoid", name=f"{name}_conf")(gap)
    box = keras.layers.Dense(4, activation="sigmoid", name=f"{name}_box")(gap)

    return {"conf": conf, "box": box}


def build_obb_model(config: OBBConfig | None = None) -> keras.Model:
    """Build the OBB model with MobileNetV3-Small + Lite-FPN.

    For circular gauges, we only need:
      - box: [cx, cy, w, h] in [0, 1]
      - confidence: scalar in [0, 1]

    No angle prediction (circular gauges are rotationally symmetric).
    """
    if config is None:
        config = OBBConfig()

    inputs = keras.Input(
        shape=(config.input_size, config.input_size, 3), name="image",
    )

    # ── Backbone: MobileNetV3-Small ────────────────────────────────────
    backbone = keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet" if config.pretrained else None,
        input_shape=(config.input_size, config.input_size, 3),
        alpha=config.alpha,
    )
    backbone.trainable = config.backbone_trainable
    backbone_out = backbone(inputs)  # 14x14xC

    # For multi-scale, upsample backbone output to get P3.
    p3 = keras.layers.UpSampling2D(size=2, interpolation="bilinear",
                                     name="p3_upsample")(backbone_out)
    p4 = backbone_out

    # ── Neck: Lite-FPN ─────────────────────────────────────────────────
    fpn_out = _lite_fpn_neck(p3, p4, out_channels=config.fpn_channels)

    # ── Head: box + confidence ────────────────────────────────────────
    outputs = _box_head(fpn_out)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"obb_v2_mnv3small_a{int(config.alpha*100):03d}_box",
    )
    return model


# ──────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────

def wise_iou_loss(pred_box: tf.Tensor, target_box: tf.Tensor) -> tf.Tensor:
    """Wise-IoU loss for bounding box regression.

    pred_box: (B, 4) [cx, cy, w, h] in [0, 1].
    target_box: (B, 4) same format.
    """
    pred_x1 = pred_box[..., 0] - pred_box[..., 2] / 2
    pred_y1 = pred_box[..., 1] - pred_box[..., 3] / 2
    pred_x2 = pred_box[..., 0] + pred_box[..., 2] / 2
    pred_y2 = pred_box[..., 1] + pred_box[..., 3] / 2

    tgt_x1 = target_box[..., 0] - target_box[..., 2] / 2
    tgt_y1 = target_box[..., 1] - target_box[..., 3] / 2
    tgt_x2 = target_box[..., 0] + target_box[..., 2] / 2
    tgt_y2 = target_box[..., 1] + target_box[..., 3] / 2

    inter_x1 = tf.maximum(pred_x1, tgt_x1)
    inter_y1 = tf.maximum(pred_y1, tgt_y1)
    inter_x2 = tf.minimum(pred_x2, tgt_x2)
    inter_y2 = tf.minimum(pred_y2, tgt_y2)
    inter = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)
    union = pred_area + tgt_area - inter + 1e-7
    iou = inter / union

    enc_x1 = tf.minimum(pred_x1, tgt_x1)
    enc_y1 = tf.minimum(pred_y1, tgt_y1)
    enc_x2 = tf.maximum(pred_x2, tgt_x2)
    enc_y2 = tf.maximum(pred_y2, tgt_y2)
    enc_w = enc_x2 - enc_x1
    enc_h = enc_y2 - enc_y1
    enc_w_sq = enc_w ** 2 + 1e-7
    enc_h_sq = enc_h ** 2 + 1e-7

    pred_cx = (pred_x1 + pred_x2) / 2
    pred_cy = (pred_y1 + pred_y2) / 2
    tgt_cx = (tgt_x1 + tgt_x2) / 2
    tgt_cy = (tgt_y1 + tgt_y2) / 2

    rw = ((pred_cx - tgt_cx) ** 2) / enc_w_sq
    rh = ((pred_cy - tgt_cy) ** 2) / enc_h_sq

    wise_iou = 1.0 - iou + tf.exp(rw + rh) - 1.0
    return tf.reduce_mean(wise_iou)


def focal_loss(
    pred_conf: tf.Tensor, target_conf: tf.Tensor,
    alpha: float = 0.25, gamma: float = 2.0,
) -> tf.Tensor:
    """Sigmoid focal loss for classification."""
    bce = tf.keras.backend.binary_crossentropy(
        target_conf, pred_conf, from_logits=False,
    )
    p = pred_conf
    p_t = p * target_conf + (1 - p) * (1 - target_conf)
    alpha_t = alpha * target_conf + (1 - alpha) * (1 - target_conf)
    focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)
    return tf.reduce_mean(focal_weight * bce)


def obb_loss(
    y_true: dict, y_pred: dict,
    *,
    conf_weight: float = 1.0,
    box_weight: float = 5.0,
) -> dict:
    """Combined OBB loss: focal (conf) + Wise-IoU (box).

    y_true: dict with keys "conf" (B,1), "box" (B,4).
    y_pred: dict with keys "conf" (B,1), "box" (B,4).
    """
    conf_loss = focal_loss(y_pred["conf"], y_true["conf"])
    box_loss = wise_iou_loss(y_pred["box"], y_true["box"])

    total = conf_weight * conf_loss + box_weight * box_loss
    return {
        "loss": total,
        "conf_loss": conf_loss,
        "box_loss": box_loss,
    }
