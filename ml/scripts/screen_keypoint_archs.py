#!/usr/bin/env python3
"""Fast architecture screen for the center/tip keypoint model.

Screens several encoder/head variants in ONE process on a capped training
subset with short epochs, scoring each on fixed small slices of the three
test sets (FP32 only).  Contract: 224x224 grayscale crop -> 56x56x2
heatmaps [center, tip] (the ``tip_focus`` deployment contract).

Variants (all QAT-safe Conv+BN+ReLU):
- unet_v1:   the deployed keypoint UNet (32/48/64/96/128, alpha=1.0)
- unet_wide: alpha=1.5 width multiplier
- unet_deep: extra encoder stage (bottleneck 7x7 -> 4x4) with fixed decoder
- unet_eca:  ECA channel attention in the encoder
- unet_noskip: decoder without skip connections
- unet_focal: unet_v1 with focal-loss head (loss ablation, same weights)

Memory safeguards (AGENTS.md): uint8 image storage, capped shuffle buffer,
per-sample float conversion, preflight estimate, launch through
``run_wsl_guarded.sh``.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embedded_gauge_reading_tinyml.keypoint_unet_224 import (  # noqa: E402
    _conv_bn_relu,
    _decoder_stage,
    _encoder_stage,
)

INPUT_SIZE = 224
HEATMAP_SIZE = 56
SEED = 42
SHUFFLE_BUFFER = 4096
MEMORY_BUDGET_MB = 40000
DATA = ROOT / "data" / "gauge_keypoint_224"


def configure_gpu() -> None:
    """Cap TensorFlow's first GPU at 15 GB so WSL retains headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------

def build_unet_v1(alpha: float = 1.0) -> keras.Model:
    """Reference: the deployed keypoint UNet (re-imported builder)."""
    from embedded_gauge_reading_tinyml.keypoint_unet_224 import build_keypoint_unet_224

    return build_keypoint_unet_224((INPUT_SIZE, INPUT_SIZE, 1), alpha=alpha)


def build_unet_eca(alpha: float = 1.0) -> keras.Model:
    """UNet v1 + ECA channel attention in every encoder stage."""
    layers = keras.layers

    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), name="image")
    e1 = _encoder_stage(inputs, w(32), "e1", downsample=True)
    e2 = _encoder_stage(e1, w(48), "e2")
    e3 = _encoder_stage(e2, w(64), "e3")
    e4 = _encoder_stage(e3, w(96), "e4")
    b = _encoder_stage(e4, w(128), "e5")

    # why: ECA on the bottleneck and the two lowest skips is cheap and
    # QAT-safe (GlobalAvgPool + 1D conv + multiply).
    def _eca(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        gap = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
        k = max(3, int((np.log2(filters) + 1) // 2))
        if k % 2 == 0:
            k += 1
        attn = layers.Reshape((1, 1, filters), name=f"{name}_reshape")(gap)
        attn = layers.Conv2D(1, k, padding="same", activation="sigmoid", name=f"{name}_conv")(attn)
        return layers.Multiply(name=f"{name}_mul")([x, attn])

    e2a = _eca(e2, w(48), "eca2")
    e3a = _eca(e3, w(64), "eca3")
    e4a = _eca(e4, w(96), "eca4")
    ba = _eca(b, w(128), "eca5")

    d1 = _decoder_stage(ba, e4a, w(96), "d1")
    d2 = _decoder_stage(d1, e3a, w(64), "d2")
    d3 = _decoder_stage(d2, e2a, w(48), "d3")
    x = _conv_bn_relu(d3, w(32), name="head_refine")
    outputs = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)
    return keras.Model(inputs, outputs, name="keypoint_unet_224_eca")


def build_unet_deep(alpha: float = 1.0) -> keras.Model:
    """UNet v1 + extra encoder stage (bottleneck 7x7 -> 4x4)."""
    layers = keras.layers

    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), name="image")
    e1 = _encoder_stage(inputs, w(32), "e1", downsample=True)
    e2 = _encoder_stage(e1, w(48), "e2")
    e3 = _encoder_stage(e2, w(64), "e3")
    e4 = _encoder_stage(e3, w(96), "e4")
    b = _encoder_stage(e4, w(128), "e5")
    # why: the 7x7 bottleneck is odd-sized, so a stride-2 stage would break
    # the doubling chain; add depth with two stride-1 refine blocks instead.
    b2 = _conv_bn_relu(b, w(160), name="e6a")
    b2 = _conv_bn_relu(b2, w(160), name="e6b")

    d1 = _decoder_stage(b2, e4, w(96), "d1")
    d2 = _decoder_stage(d1, e3, w(64), "d2")
    d3 = _decoder_stage(d2, e2, w(48), "d3")
    x = _conv_bn_relu(d3, w(32), name="head_refine")
    outputs = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)
    return keras.Model(inputs, outputs, name="keypoint_unet_224_deep")


def build_unet_noskip(alpha: float = 1.0) -> keras.Model:
    """UNet v1 with the decoder skip connections removed."""
    layers = keras.layers

    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), name="image")
    e1 = _encoder_stage(inputs, w(32), "e1", downsample=True)
    e2 = _encoder_stage(e1, w(48), "e2")
    e3 = _encoder_stage(e2, w(64), "e3")
    e4 = _encoder_stage(e3, w(96), "e4")
    b = _encoder_stage(e4, w(128), "e5")

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="d0_up")(b)
    x = _conv_bn_relu(x, w(96), name="d0")
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="d1_up")(x)
    x = _conv_bn_relu(x, w(64), name="d1")
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="d2_up")(x)
    x = _conv_bn_relu(x, w(48), name="d2")
    x = _conv_bn_relu(x, w(32), name="head_refine")
    outputs = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)
    return keras.Model(inputs, outputs, name="keypoint_unet_224_noskip")


def build_unet_offset(alpha: float = 1.0) -> keras.Model:
    """UNet wide + CenterNet-style tip-offset head.

    Literature: Zhou et al. 2019 regresses a sub-cell offset at the keypoint
    location; here the tip is predicted as a 2D offset from the center, so
    the model must learn needle length explicitly instead of underestimating
    it (the documented heatmap failure mode).

    Outputs: [heatmaps (56,56,2), tip_offset (2)].
    """

    layers = keras.layers

    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), name="image")
    e1 = _encoder_stage(inputs, w(32), "e1", downsample=True)
    e2 = _encoder_stage(e1, w(48), "e2")
    e3 = _encoder_stage(e2, w(64), "e3")
    e4 = _encoder_stage(e3, w(96), "e4")
    b = _encoder_stage(e4, w(128), "e5")

    d1 = _decoder_stage(b, e4, w(96), "d1")
    d2 = _decoder_stage(d1, e3, w(64), "d2")
    d3 = _decoder_stage(d2, e2, w(48), "d3")
    x = _conv_bn_relu(d3, w(32), name="head_refine")
    heatmaps = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)

    # why: tip offset in normalized crop units [-1, 1]; pooled from fine
    # features so the head sees the whole needle, not a local window.
    gap = layers.GlobalAveragePooling2D(name="offset_gap")(x)
    offset_shared = layers.Dense(w(64), activation="relu", name="offset_shared")(gap)
    tip_offset = layers.Dense(2, name="tip_offset")(offset_shared)

    return keras.Model(inputs, [heatmaps, tip_offset], name="keypoint_unet_224_offset")


def build_unet_polar(alpha: float = 1.0) -> keras.Model:
    """UNet wide + polar tip head (length, sin, cos).

    Literature: the research doc's section 2.7 recommends polar coordinates
    (angle, length) from the center; length is constrained to the needle,
    and the model learns direction and length instead of an absolute tip.

    Outputs: [heatmaps (56,56,2), tip_polar (3)] with polar = (length,
    sin_angle, cos_angle), length in normalized crop units [0, ~0.8].
    """

    layers = keras.layers

    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), name="image")
    e1 = _encoder_stage(inputs, w(32), "e1", downsample=True)
    e2 = _encoder_stage(e1, w(48), "e2")
    e3 = _encoder_stage(e2, w(64), "e3")
    e4 = _encoder_stage(e3, w(96), "e4")
    b = _encoder_stage(e4, w(128), "e5")

    d1 = _decoder_stage(b, e4, w(96), "d1")
    d2 = _decoder_stage(d1, e3, w(64), "d2")
    d3 = _decoder_stage(d2, e2, w(48), "d3")
    x = _conv_bn_relu(d3, w(32), name="head_refine")
    heatmaps = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)

    gap = layers.GlobalAveragePooling2D(name="polar_gap")(x)
    polar_shared = layers.Dense(w(64), activation="relu", name="polar_shared")(gap)
    length = layers.Dense(1, activation="sigmoid", name="tip_length")(polar_shared)
    direction = layers.Dense(2, activation="tanh", name="tip_direction")(polar_shared)
    tip_polar = layers.Concatenate(name="tip_polar")([length, direction])

    return keras.Model(inputs, [heatmaps, tip_polar], name="keypoint_unet_224_polar")


def build_unet_offmap(alpha: float = 1.0) -> keras.Model:
    """UNet + CenterNet-style tip-offset MAP (spatial, not GAP).

    Literature: Zhou et al. 2019 regresses a 2D offset map at the same
    resolution as the heatmap; the offset is read at the center peak, so the
    head keeps spatial structure.  The GAP-pooled offset head failed in
    screen v2 (14-55px tip) because pooling erased position.

    Outputs: [heatmaps (56,56,2), offset_map (56,56,2)].
    """

    layers = keras.layers

    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), name="image")
    e1 = _encoder_stage(inputs, w(32), "e1", downsample=True)
    e2 = _encoder_stage(e1, w(48), "e2")
    e3 = _encoder_stage(e2, w(64), "e3")
    e4 = _encoder_stage(e3, w(96), "e4")
    b = _encoder_stage(e4, w(128), "e5")

    d1 = _decoder_stage(b, e4, w(96), "d1")
    d2 = _decoder_stage(d1, e3, w(64), "d2")
    d3 = _decoder_stage(d2, e2, w(48), "d3")
    x = _conv_bn_relu(d3, w(32), name="head_refine")
    heatmaps = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)

    # why: the offset map is a 2-channel (dx, dy) regression at 56x56,
    # supervised only near the center peak so the model learns the
    # center->tip vector at the right location.
    offset_map = layers.Conv2D(2, 1, padding="same", name="offset_map")(x)

    return keras.Model(inputs, [heatmaps, offset_map], name="keypoint_unet_224_offmap")

def build_unet_stride2_lean(alpha: float = 1.0) -> keras.Model:
    """Stride-2 UNet with reduced high-res channels to fit N6 NPU SRAM.

    GPT's N6 packaging probe: the alpha=1.0 stride-2 model peaks at
    3,268,496 bytes (3.12 MiB) vs the 2,883,584-byte (2.75 MiB) no-HyperRAM
    pool.  The dominant tensor is the 112x112 decoder concat (d3-up 48ch +
    e1-skip 32ch = 80ch = ~1.0 MiB).  This lean variant cuts the high-res
    channels (e1 24, d3 32, d4/head 24) so the 112x112 concat drops to
    56ch (~0.70 MiB) and every 112x112 tensor to 24ch (~0.30 MiB).

    Keeps the 112x112x2 output (the tip-accuracy win) and alpha=1.0
    encoder depth.  Weights ~0.9 MB; peak activation target < 2.5 MiB.
    """

    layers = keras.layers

    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), name="image")
    e1 = _encoder_stage(inputs, w(24), "e1", downsample=True)   # 224→112, 24ch
    e2 = _encoder_stage(e1, w(32), "e2")                        # 112→ 56, 32ch
    e3 = _encoder_stage(e2, w(48), "e3")                        #  56→ 28, 48ch
    e4 = _encoder_stage(e3, w(64), "e4")                        #  28→ 14, 64ch
    b = _encoder_stage(e4, w(96), "e5")                         #  14→  7, 96ch (bottleneck)

    d1 = _decoder_stage(b, e4, w(64), "d1")   # 7→14
    d2 = _decoder_stage(d1, e3, w(48), "d2")  # 14→28
    d3 = _decoder_stage(d2, e2, w(32), "d3")  # 28→56 (lean: 32 not 48)
    d4 = _decoder_stage(d3, e1, w(24), "d4")  # 56→112 (lean: 24 not 32)
    x = _conv_bn_relu(d4, w(24), name="head_refine")
    outputs = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)
    return keras.Model(inputs, outputs, name="keypoint_unet_224_stride2_lean")


def build_unet_stride2(alpha: float = 1.5) -> keras.Model:
    """UNet with a stride-2 output head: 112x112 heatmaps (2px cells).

    Literature: the research doc notes stride-2 heatmaps halve keypoint
    quantization error (4px -> 2px at 224 input).  The DARK-decode failure
    showed the 56x56 int8 heatmaps are the coarse binding constraint; a
    112x112 output addresses it directly.  Decoder adds one more upsample
    (28->56->112) with the e1 skip, matching the deployed head contract
    (output shape 112x112x2).

    NOTE: GPT's N6 probe found this 112x112 output needs 3.12 MiB peak
    activations (incl. 980 KiB HyperRAM) — the LEAN variant above is the
    board-fit alternative.
    """

    layers = keras.layers

    def w(base: int) -> int:
        return max(16, int(base * alpha))

    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 1), name="image")
    e1 = _encoder_stage(inputs, w(32), "e1", downsample=True)
    e2 = _encoder_stage(e1, w(48), "e2")
    e3 = _encoder_stage(e2, w(64), "e3")
    e4 = _encoder_stage(e3, w(96), "e4")
    b = _encoder_stage(e4, w(128), "e5")

    d1 = _decoder_stage(b, e4, w(96), "d1")
    d2 = _decoder_stage(d1, e3, w(64), "d2")
    d3 = _decoder_stage(d2, e2, w(48), "d3")   # 56x56
    d4 = _decoder_stage(d3, e1, w(32), "d4")   # 112x112 (stride 2)
    x = _conv_bn_relu(d4, w(32), name="head_refine")
    outputs = layers.Conv2D(2, 1, padding="same", activation="sigmoid", name="heatmaps")(x)
    return keras.Model(inputs, outputs, name="keypoint_unet_224_stride2")


def decode_heatmap_112(heatmap: np.ndarray) -> tuple[float, float]:
    """Soft-argmax decode of a 112x112 heatmap to (x, y) in crop pixels."""
    size = heatmap.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    weights = np.maximum(heatmap - 0.05, 0.0) ** 4.0
    total = max(float(weights.sum()), 1e-6)
    x = float((weights * xx).sum() / total)
    y = float((weights * yy).sum() / total)
    return x * (INPUT_SIZE / size), y * (INPUT_SIZE / size)


def evaluate_heatmaps_112(pred: np.ndarray, gt_center: np.ndarray, gt_tip: np.ndarray) -> dict[str, float]:
    """Center/tip MAE for 112x112 outputs (GT heatmaps stay 56x56)."""
    center_errs: list[float] = []
    tip_errs: list[float] = []
    for i in range(len(pred)):
        cx, cy = decode_heatmap_112(pred[i, :, :, 0])
        tx, ty = decode_heatmap_112(pred[i, :, :, 1])
        gy, gx = np.unravel_index(np.argmax(gt_center[i]), gt_center[i].shape)
        gt_cx = gx * (INPUT_SIZE / HEATMAP_SIZE)
        gt_cy = gy * (INPUT_SIZE / HEATMAP_SIZE)
        ty_p, tx_p = np.unravel_index(np.argmax(gt_tip[i]), gt_tip[i].shape)
        gt_tx = tx_p * (INPUT_SIZE / HEATMAP_SIZE)
        gt_ty = ty_p * (INPUT_SIZE / HEATMAP_SIZE)
        center_errs.append(np.hypot(cx - gt_cx, cy - gt_cy))
        tip_errs.append(np.hypot(tx - gt_tx, ty - gt_ty))
    center_errs = np.asarray(center_errs)
    tip_errs = np.asarray(tip_errs)
    return {
        "center_mae_px": float(np.mean(center_errs)),
        "center_pct_le_8px": float(np.mean(center_errs <= 8.0)),
        "tip_mae_px": float(np.mean(tip_errs)),
        "tip_pct_le_8px": float(np.mean(tip_errs <= 8.0)),
    }


ARCHES = {
    # name: (builder, alpha, loss_mode, target_mode, decode_mode)
    # why: board-fit candidates — stride-2 (112x112) output for tip
    # accuracy, but lean decoder channels so the calibrated N6 peak
    # activation stays under 2.5 MiB (GPT probe: stride2_s = 3.12 MiB
    # fails; lean alpha=1.0 = 2.18 MiB, alpha=1.1 = 2.38 MiB both fit).
    "unet_wide": (build_unet_v1, 1.5, "focal", "heatmap", "heatmap"),
    "unet_stride2_lean": (build_unet_stride2_lean, 1.0, "focal", "heatmap112", "heatmap112"),
    "unet_stride2_lean11": (build_unet_stride2_lean, 1.1, "focal", "heatmap112", "heatmap112"),
    "unet_s": (build_unet_v1, 1.0, "focal", "heatmap", "heatmap"),
}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_heatmap_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    center_weight: float = 1.0,
    tip_weight: float = 8.0,
    gamma: float = 4.0,
) -> tf.Tensor:
    """Focal-style heatmap loss with per-channel weights (center < tip).

    Matches the deployed keypoint trainer's recipe: tip is sparser and gets
    more weight.
    """
    weights = tf.constant([center_weight, tip_weight], dtype=tf.float32)
    weights = weights[None, None, None, :]
    pos = -tf.pow(1.0 - y_pred, gamma) * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0))
    neg = -tf.pow(y_pred, gamma) * tf.math.log(tf.clip_by_value(1.0 - y_pred, 1e-7, 1.0))
    loss = weights * (y_true * pos + (1.0 - y_true) * neg)
    return tf.reduce_mean(loss)


def offset_aux_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    offset_weight: float = 0.05,
) -> tf.Tensor:
    """L1 on the CenterNet-style tip-offset head (second output only).

    why: the focal heatmap loss is ~0.005; a 4.0 offset weight dominated
    training and starved the heatmaps (center degraded to 11px in screen
    v2).  A small weight keeps the offset a gentle geometric regularizer.
    """
    return offset_weight * tf.reduce_mean(tf.abs(y_true - y_pred))


def polar_aux_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    length_weight: float = 0.05,
    direction_weight: float = 0.02,
) -> tf.Tensor:
    """Length L1 + cosine direction loss for the polar head (second output).

    why: same balance lesson as the offset head — keep the aux terms small
    so the heatmaps remain the primary supervision.
    """
    length_l1 = tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))
    pred_dir = y_pred[:, 1:]
    true_dir = y_true[:, 1:]
    cos_sim = tf.reduce_sum(pred_dir * true_dir, axis=-1) / (
        tf.norm(pred_dir, axis=-1) * tf.norm(true_dir, axis=-1) + 1e-6
    )
    direction_loss = tf.reduce_mean(1.0 - cos_sim)
    return length_weight * length_l1 + direction_weight * direction_loss


# ---------------------------------------------------------------------------
# Decode + metrics (center/tip error in crop pixels)
# ---------------------------------------------------------------------------

def decode_heatmap(heatmap: np.ndarray) -> tuple[float, float]:
    """Soft-argmax decode of one 56x56 heatmap to (x, y) in crop pixels."""
    yy, xx = np.mgrid[0:HEATMAP_SIZE, 0:HEATMAP_SIZE].astype(np.float32)
    weights = np.maximum(heatmap - 0.05, 0.0) ** 4.0
    total = max(float(weights.sum()), 1e-6)
    x = float((weights * xx).sum() / total)
    y = float((weights * yy).sum() / total)
    return x * (INPUT_SIZE / HEATMAP_SIZE), y * (INPUT_SIZE / HEATMAP_SIZE)


def evaluate_heatmaps(pred: np.ndarray, gt_center: np.ndarray, gt_tip: np.ndarray) -> dict[str, float]:
    """Return center/tip MAE and pct<=8px in crop pixels."""
    center_errs: list[float] = []
    tip_errs: list[float] = []
    for i in range(len(pred)):
        cx, cy = decode_heatmap(pred[i, :, :, 0])
        tx, ty = decode_heatmap(pred[i, :, :, 1])
        # why: GT heatmaps are (56,56) 2D planes; argmax over the whole map
        # gives the peak pixel, then scale to 224 crop pixels.
        gy, gx = np.unravel_index(np.argmax(gt_center[i]), gt_center[i].shape)
        gt_cx = gx * (INPUT_SIZE / HEATMAP_SIZE)
        gt_cy = gy * (INPUT_SIZE / HEATMAP_SIZE)
        ty_p, tx_p = np.unravel_index(np.argmax(gt_tip[i]), gt_tip[i].shape)
        gt_tx = tx_p * (INPUT_SIZE / HEATMAP_SIZE)
        gt_ty = ty_p * (INPUT_SIZE / HEATMAP_SIZE)
        center_errs.append(np.hypot(cx - gt_cx, cy - gt_cy))
        tip_errs.append(np.hypot(tx - gt_tx, ty - gt_ty))
    center_errs = np.asarray(center_errs)
    tip_errs = np.asarray(tip_errs)
    return {
        "center_mae_px": float(np.mean(center_errs)),
        "center_pct_le_8px": float(np.mean(center_errs <= 8.0)),
        "tip_mae_px": float(np.mean(tip_errs)),
        "tip_pct_le_8px": float(np.mean(tip_errs <= 8.0)),
    }


def _gt_peaks(gt_center: np.ndarray, gt_tip: np.ndarray, index: int) -> tuple[float, float, float, float]:
    """Return GT center and tip in 224-crop pixels for one sample."""
    gy, gx = np.unravel_index(np.argmax(gt_center[index]), gt_center[index].shape)
    ty, tx = np.unravel_index(np.argmax(gt_tip[index]), gt_tip[index].shape)
    scale = INPUT_SIZE / HEATMAP_SIZE
    return gx * scale, gy * scale, tx * scale, ty * scale


def decode_offset_outputs(
    preds: list[np.ndarray],
    gt_center: np.ndarray,
    gt_tip: np.ndarray,
) -> dict[str, float]:
    """Decode [heatmaps, tip_offset]: center from heatmap, tip = center + offset.

    The offset head predicts (dx, dy) in normalized crop units; multiply by
    INPUT_SIZE to get crop pixels.
    """
    heatmaps, offsets = preds
    center_errs: list[float] = []
    tip_errs: list[float] = []
    for i in range(len(heatmaps)):
        cx, cy = decode_heatmap(heatmaps[i, :, :, 0])
        gt_cx, gt_cy, gt_tx, gt_ty = _gt_peaks(gt_center, gt_tip, i)
        tx = cx + offsets[i, 0] * INPUT_SIZE
        ty = cy + offsets[i, 1] * INPUT_SIZE
        center_errs.append(np.hypot(cx - gt_cx, cy - gt_cy))
        tip_errs.append(np.hypot(tx - gt_tx, ty - gt_ty))
    center_errs = np.asarray(center_errs)
    tip_errs = np.asarray(tip_errs)
    return {
        "center_mae_px": float(np.mean(center_errs)),
        "center_pct_le_8px": float(np.mean(center_errs <= 8.0)),
        "tip_mae_px": float(np.mean(tip_errs)),
        "tip_pct_le_8px": float(np.mean(tip_errs <= 8.0)),
    }


def decode_polar_outputs(
    preds: list[np.ndarray],
    gt_center: np.ndarray,
    gt_tip: np.ndarray,
) -> dict[str, float]:
    """Decode [heatmaps, tip_polar]: center from heatmap, tip = center + polar.

    Polar = (length, sin, cos); the direction vector is normalized to unit
    length at decode time.
    """
    heatmaps, polars = preds
    center_errs: list[float] = []
    tip_errs: list[float] = []
    for i in range(len(heatmaps)):
        cx, cy = decode_heatmap(heatmaps[i, :, :, 0])
        gt_cx, gt_cy, gt_tx, gt_ty = _gt_peaks(gt_center, gt_tip, i)
        length = float(polars[i, 0])
        s, c = polars[i, 1], polars[i, 2]
        norm = max(float(np.hypot(s, c)), 1e-6)
        tx = cx + length * (c / norm) * INPUT_SIZE
        ty = cy + length * (s / norm) * INPUT_SIZE
        center_errs.append(np.hypot(cx - gt_cx, cy - gt_cy))
        tip_errs.append(np.hypot(tx - gt_tx, ty - gt_ty))
    center_errs = np.asarray(center_errs)
    tip_errs = np.asarray(tip_errs)
    return {
        "center_mae_px": float(np.mean(center_errs)),
        "center_pct_le_8px": float(np.mean(center_errs <= 8.0)),
        "tip_mae_px": float(np.mean(tip_errs)),
        "tip_pct_le_8px": float(np.mean(tip_errs <= 8.0)),
    }


def make_offset_targets(center: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Build tip-offset targets (dx, dy) in normalized crop units.

    GT peaks are at heatmap pixels; convert to normalized [0,1] crop coords
    by dividing by (HEATMAP_SIZE - 1), then offset = tip_norm - center_norm.
    """
    scale = 1.0 / (HEATMAP_SIZE - 1)
    offsets = []
    for i in range(len(center)):
        gy, gx = np.unravel_index(np.argmax(center[i]), center[i].shape)
        ty, tx = np.unravel_index(np.argmax(tip[i]), tip[i].shape)
        offsets.append([(tx - gx) * scale, (ty - gy) * scale])
    return np.asarray(offsets, dtype=np.float32)


def make_polar_targets(center: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Build polar targets (length, sin, cos) in normalized crop units."""
    scale = 1.0 / (HEATMAP_SIZE - 1)
    polars = []
    for i in range(len(center)):
        gy, gx = np.unravel_index(np.argmax(center[i]), center[i].shape)
        ty, tx = np.unravel_index(np.argmax(tip[i]), tip[i].shape)
        dx, dy = (tx - gx) * scale, (ty - gy) * scale
        length = float(np.hypot(dx, dy))
        if length > 1e-6:
            polars.append([length, dy / length, dx / length])
        else:
            polars.append([0.0, 0.0, 1.0])
    return np.asarray(polars, dtype=np.float32)




def make_offmap_targets(center: np.ndarray, tip: np.ndarray, sigma_bins: float = 2.0) -> np.ndarray:
    """Build a (N,56,56,2) offset map: (dx, dy) masked near the center peak.

    Each pixel holds the center->tip vector in heatmap-pixel units; pixels
    far from the center peak get zeros and are masked out of the loss.
    """
    scale = INPUT_SIZE / HEATMAP_SIZE
    yy, xx = np.mgrid[0:HEATMAP_SIZE, 0:HEATMAP_SIZE].astype(np.float32)
    maps = []
    for i in range(len(center)):
        gy, gx = np.unravel_index(np.argmax(center[i]), center[i].shape)
        ty, tx = np.unravel_index(np.argmax(tip[i]), tip[i].shape)
        dx = (tx - gx) / scale  # normalized crop units
        dy = (ty - gy) / scale
        dist2 = (xx - gx) ** 2 + (yy - gy) ** 2
        mask = np.exp(-dist2 / (2.0 * sigma_bins**2))[..., None]
        maps.append(np.concatenate([np.full((56, 56, 1), dx), np.full((56, 56, 1), dy)], axis=-1) * mask)
    return np.asarray(maps, dtype=np.float32)


def offmap_aux_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    mask_threshold: float = 1e-3,
    offset_weight: float = 0.05,
) -> tf.Tensor:
    """L1 on the offset map, masked to the Gaussian neighborhood of the center.

    why: pixels far from the center carry no useful offset signal; masking
    them keeps the loss focused on the needle direction at the pivot.  The
    weight is small (like the other aux heads) so the heatmaps stay the
    primary supervision.
    """
    mask = tf.cast(y_true > mask_threshold, tf.float32)
    masked_error = mask * tf.abs(y_true - y_pred)
    return offset_weight * tf.reduce_sum(masked_error) / (tf.reduce_sum(mask) + 1e-6)


def decode_offmap_outputs(
    preds: list[np.ndarray],
    gt_center: np.ndarray,
    gt_tip: np.ndarray,
) -> dict[str, float]:
    """Decode [heatmaps, offset_map]: center from heatmap peak, then read
    the offset vector AT the peak location."""
    heatmaps, offset_maps = preds
    center_errs: list[float] = []
    tip_errs: list[float] = []
    for i in range(len(heatmaps)):
        cx, cy = decode_heatmap(heatmaps[i, :, :, 0])
        gt_cx, gt_cy, gt_tx, gt_ty = _gt_peaks(gt_center, gt_tip, i)
        # read offset at the heatmap peak (rounded to the 56-grid)
        py = int(round(cy / (INPUT_SIZE / HEATMAP_SIZE)))
        px = int(round(cx / (INPUT_SIZE / HEATMAP_SIZE)))
        py = min(max(py, 0), HEATMAP_SIZE - 1)
        px = min(max(px, 0), HEATMAP_SIZE - 1)
        dx = float(offset_maps[i, py, px, 0])
        dy = float(offset_maps[i, py, px, 1])
        tx = cx + dx * INPUT_SIZE
        ty = cy + dy * INPUT_SIZE
        center_errs.append(np.hypot(cx - gt_cx, cy - gt_cy))
        tip_errs.append(np.hypot(tx - gt_tx, ty - gt_ty))
    center_errs = np.asarray(center_errs)
    tip_errs = np.asarray(tip_errs)
    return {
        "center_mae_px": float(np.mean(center_errs)),
        "center_pct_le_8px": float(np.mean(center_errs <= 8.0)),
        "tip_mae_px": float(np.mean(tip_errs)),
        "tip_pct_le_8px": float(np.mean(tip_errs <= 8.0)),
    }


def _augment_keypoint_sample(
    image: np.ndarray,
    center_hm: np.ndarray,
    tip_hm: np.ndarray,
    rng: np.random.Generator,
    flip_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augment one keypoint sample: hflip/rotate/scale + regenerate heatmaps.

    why (2026-07-30 lesson): rotating a Gaussian heatmap directly spreads
    the peak and corrupts it.  The correct approach is to transform the
    IMAGE, transform the keypoint coordinates, then re-render the heatmaps
    from the new coordinates.
    """
    from PIL import Image, ImageEnhance

    gy, gx = np.unravel_index(np.argmax(center_hm), center_hm.shape)
    ty, tx = np.unravel_index(np.argmax(tip_hm), tip_hm.shape)
    scale = INPUT_SIZE / HEATMAP_SIZE
    cx, cy = gx * scale, gy * scale
    tip_x, tip_y = tx * scale, ty * scale

    img = Image.fromarray(image[..., 0])
    if rng.random() < flip_prob:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        cx = INPUT_SIZE - cx
        tip_x = INPUT_SIZE - tip_x

    angle = rng.uniform(-10.0, 10.0)
    if abs(angle) > 0.5:
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=int(np.mean(image)))
        rad = np.deg2rad(angle)
        c = np.cos(rad)
        s = np.sin(rad)
        center = INPUT_SIZE / 2.0
        cx, cy = (center + (cx - center) * c - (cy - center) * s,
                  center + (cx - center) * s + (cy - center) * c)
        tip_x, tip_y = (center + (tip_x - center) * c - (tip_y - center) * s,
                        center + (tip_x - center) * s + (tip_y - center) * c)

    scale_f = rng.uniform(0.9, 1.1)
    if abs(scale_f - 1.0) > 0.01:
        # why: PIL resizes around the top-left, so to zoom about the image
        # center we must transform the keypoints about the same origin.
        new_size = int(round(INPUT_SIZE * scale_f))
        img = img.resize((new_size, new_size), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (INPUT_SIZE, INPUT_SIZE), color=int(np.mean(image)))
        offset = (INPUT_SIZE - new_size) // 2
        canvas.paste(img, (offset, offset))
        img = canvas
        cx = cx * scale_f + offset
        cy = cy * scale_f + offset
        tip_x = tip_x * scale_f + offset
        tip_y = tip_y * scale_f + offset

    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.85, 1.15)))
    aug_image = np.asarray(img, dtype=np.uint8)[..., None]

    new_center = _render_gaussian(cx, cy)
    new_tip = _render_gaussian(tip_x, tip_y)
    return aug_image, new_center, new_tip


def _render_gaussian(x: float, y: float) -> np.ndarray:
    """Render a 56x56 Gaussian heatmap at crop-pixel (x, y)."""
    hm_x = x / (INPUT_SIZE / HEATMAP_SIZE)
    hm_y = y / (INPUT_SIZE / HEATMAP_SIZE)
    yy, xx = np.mgrid[0:HEATMAP_SIZE, 0:HEATMAP_SIZE].astype(np.float32)
    return np.exp(-((xx - hm_x) ** 2 + (yy - hm_y) ** 2) / (2.0 * 2.0**2)).astype(np.float32)


def _augment_keypoint_set(
    images: np.ndarray,
    center: np.ndarray,
    tip: np.ndarray,
    flip_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Double the set with augmented copies (image + regenerated heatmaps).

    why: the keypoint pipeline has ZERO augmentation today; the ellipse
    pipeline showed hflip alone cured a held-out-family failure (test_2
    28->6.3px).  Augmenting with flip/rotate/scale + re-rendered heatmaps is
    the highest-leverage fix for tip generalization.  ``flip_prob=0``
    disables flips (mirrored dial text is an impossible view that hurt
    test_3 in the full-data run).
    """
    rng = np.random.default_rng(SEED)
    aug_images = np.empty_like(images)
    aug_center = np.empty_like(center)
    aug_tip = np.empty_like(tip)
    for index in range(len(images)):
        aug_images[index], aug_center[index], aug_tip[index] = _augment_keypoint_sample(
            images[index], center[index], tip[index], rng, flip_prob=flip_prob
        )
    return (
        np.concatenate([images, aug_images], axis=0),
        np.concatenate([center, aug_center], axis=0),
        np.concatenate([tip, aug_tip], axis=0),
    )


def _memory_preflight(n_samples: int) -> None:
    image_bytes = n_samples * INPUT_SIZE * INPUT_SIZE  # uint8
    heatmap_bytes = n_samples * HEATMAP_SIZE * HEATMAP_SIZE * 2 * 4  # float32
    shuffle_bytes = min(n_samples, SHUFFLE_BUFFER) * (
        INPUT_SIZE * INPUT_SIZE + HEATMAP_SIZE * HEATMAP_SIZE * 2 * 4
    )
    total_mb = (image_bytes + heatmap_bytes + shuffle_bytes) / 1e6
    print(f"memory preflight: {total_mb / 1024:.1f} GiB estimated for {n_samples} samples", flush=True)
    if total_mb > MEMORY_BUDGET_MB:
        raise SystemExit(
            f"aborting: estimated {total_mb / 1024:.1f} GiB exceeds {MEMORY_BUDGET_MB / 1024:.0f} GiB budget"
        )


def _load_split_images(split: str) -> np.ndarray:
    """Load all images of one split as a uint8 array.

    The prepared keypoint dataset stores images as ``images/000000.jpg``
    files with matching ``center.npy``/``tip.npy`` heatmaps; this loads them
    in filename order so indices line up with the heatmap arrays.
    """
    from PIL import Image

    img_dir = DATA / split / "images"
    paths = sorted(img_dir.glob("*.jpg"))
    images = np.stack(
        [np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in paths]
    )[..., None]
    return images


def main() -> None:
    """Screen all keypoint architectures on the capped subset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "keypoint_screen_fast")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--test-1-limit", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--arches", type=str, default=",".join(ARCHES.keys()))
    parser.add_argument("--augment", action="store_true",
                        help="Double the training set with flip/rotate/scale "
                             "augmentation (image + regenerated heatmaps).")
    parser.add_argument("--flip-prob", type=float, default=0.5,
                        help="Probability of horizontal flip in augmentation "
                             "(0 disables flips: mirrored dial text is an "
                             "impossible view that hurt test_3).")
    args = parser.parse_args()

    configure_gpu()
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data once; cap the train slice deterministically.
    train_images = _load_split_images("train")
    train_center = np.load(DATA / "train" / "center.npy")
    train_tip = np.load(DATA / "train" / "tip.npy")
    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(train_images), min(args.max_train, len(train_images)), replace=False)
    train_images = train_images[keep]
    train_center = train_center[keep]
    train_tip = train_tip[keep]
    if args.augment:
        # why: double the set with augmented copies BEFORE slicing the
        # target arrays so the aux targets (offset/polar/offmap) are built
        # on the augmented keypoints too.
        train_images, train_center, train_tip = _augment_keypoint_set(
            train_images, train_center, train_tip, flip_prob=args.flip_prob
        )
        print(f"augmented train slice: {len(train_images)} images (flip_prob={args.flip_prob})", flush=True)
    images_u8 = train_images
    del train_images
    print(f"train slice: {len(images_u8)} images", flush=True)

    test_slices: dict[str, dict[str, np.ndarray]] = {}
    split_images = _load_split_images("test")
    split_center = np.load(DATA / "test" / "center.npy")
    split_tip = np.load(DATA / "test" / "tip.npy")
    # why: deterministic 150-image slice of test_1 plus the small board
    # test sets are tiny already; take fixed first-N for speed.
    test_slices["test_1"] = {
        "images": split_images[: args.test_1_limit],
        "center": split_center[: args.test_1_limit],
        "tip": split_tip[: args.test_1_limit],
    }
    # test_2/test_3 are separate archives but share the same output dir here;
    # note: the staged test dir merges all test zips, so evaluate on the
    # fixed slice only (screening is for ranking, not acceptance).
    _memory_preflight(len(images_u8))

    # why: center/tip are (N,56,56) planes; stack on a new channel axis so
    # the target is (N,56,56,2) like the model output.
    heatmap_targets = np.stack([train_center, train_tip], axis=-1)

    leaderboard: dict[str, object] = {}
    steps_per_epoch = max(1, len(images_u8) // args.batch_size)

    # Precompute the aux targets once (offset/polar/offmap variants share them).
    offset_targets = make_offset_targets(train_center, train_tip)
    polar_targets = make_polar_targets(train_center, train_tip)
    offmap_targets = make_offmap_targets(train_center, train_tip)

    def _build_dataset(target_mode: str) -> tf.data.Dataset:
        """Build a generator dataset whose targets match the arch's heads."""
        if target_mode == "heatmap112":
            # why: GT heatmaps are 56x56; upsample to 112x112 with bilinear
            # so the stride-2 head supervises at its own resolution (the
            # Gaussian peaks stay centered because upsampling is symmetric).
            from PIL import Image as PILImage

            targets_112 = np.stack(
                [
                    np.asarray(
                        PILImage.fromarray(c).resize((112, 112), PILImage.Resampling.BILINEAR)
                    )
                    for c in train_center
                ],
                axis=0,
            )[..., None]
            tips_112 = np.stack(
                [
                    np.asarray(
                        PILImage.fromarray(t).resize((112, 112), PILImage.Resampling.BILINEAR)
                    )
                    for t in train_tip
                ],
                axis=0,
            )[..., None]
            heatmap112 = np.concatenate([targets_112, tips_112], axis=-1).astype(np.float32)

            def samples() -> object:
                for index in range(len(images_u8)):
                    yield images_u8[index].astype(np.float32) / 255.0, heatmap112[index]

            signature = (
                tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 1), tf.float32),
                tf.TensorSpec((112, 112, 2), tf.float32),
            )
        elif target_mode == "offmap":
            def samples() -> object:
                for index in range(len(images_u8)):
                    yield images_u8[index].astype(np.float32) / 255.0, (
                        heatmap_targets[index], offmap_targets[index],
                    )
            signature = (
                tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 1), tf.float32),
                (
                    tf.TensorSpec((HEATMAP_SIZE, HEATMAP_SIZE, 2), tf.float32),
                    tf.TensorSpec((HEATMAP_SIZE, HEATMAP_SIZE, 2), tf.float32),
                ),
            )
        elif target_mode == "offset":
            def samples() -> object:
                for index in range(len(images_u8)):
                    yield images_u8[index].astype(np.float32) / 255.0, (
                        heatmap_targets[index], offset_targets[index],
                    )
            signature = (
                tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 1), tf.float32),
                (
                    tf.TensorSpec((HEATMAP_SIZE, HEATMAP_SIZE, 2), tf.float32),
                    tf.TensorSpec((2,), tf.float32),
                ),
            )
        elif target_mode == "polar":
            def samples() -> object:
                for index in range(len(images_u8)):
                    yield images_u8[index].astype(np.float32) / 255.0, (
                        heatmap_targets[index], polar_targets[index],
                    )
            signature = (
                tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 1), tf.float32),
                (
                    tf.TensorSpec((HEATMAP_SIZE, HEATMAP_SIZE, 2), tf.float32),
                    tf.TensorSpec((3,), tf.float32),
                ),
            )
        else:
            def samples() -> object:
                for index in range(len(images_u8)):
                    yield images_u8[index].astype(np.float32) / 255.0, heatmap_targets[index]
            signature = (
                tf.TensorSpec((INPUT_SIZE, INPUT_SIZE, 1), tf.float32),
                tf.TensorSpec((HEATMAP_SIZE, HEATMAP_SIZE, 2), tf.float32),
            )
        return (
            tf.data.Dataset.from_generator(samples, output_signature=signature)
            .shuffle(min(SHUFFLE_BUFFER, len(images_u8)), seed=SEED, reshuffle_each_iteration=True)
            .batch(args.batch_size)
            .prefetch(1)
        )

    for name in args.arches.split(","):
        name = name.strip()
        if name not in ARCHES:
            continue
        print(f"\n=== {name} ===", flush=True)
        builder, alpha, loss_mode, target_mode, decode_mode = ARCHES[name]
        # why: every architecture must start from the SAME weight-init seed.
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        model = builder(alpha)
        n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
        # why: match the deployed trainer's cosine schedule; a flat 1e-3 LR
        # undertrains the sparse tip channel at screen-scale steps, which
        # made the first screen passes unrankable (v6 gives 2.9/6.5px
        # through this exact eval path).
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,
            decay_steps=max(1, args.epochs * steps_per_epoch),
            alpha=0.01,
        )
        if loss_mode == "offmap":
            loss_fn = [
                lambda yt, yp: focal_heatmap_loss(yt, yp),
                offmap_aux_loss,
            ]
        elif loss_mode == "offset":
            # why: tf_keras calls each loss once per output with that
            # output's own tensors, so pass a per-output list.
            loss_fn = [
                lambda yt, yp: focal_heatmap_loss(yt, yp),
                offset_aux_loss,
            ]
        elif loss_mode == "polar":
            loss_fn = [
                lambda yt, yp: focal_heatmap_loss(yt, yp),
                polar_aux_loss,
            ]
        elif loss_mode == "focal_tipw":
            loss_fn = lambda yt, yp: focal_heatmap_loss(yt, yp, tip_weight=16.0)
        else:
            loss_fn = lambda yt, yp: focal_heatmap_loss(yt, yp)
        model.compile(
            optimizer=keras.optimizers.AdamW(lr_schedule, weight_decay=1e-4),
            loss=loss_fn,
        )
        model.fit(_build_dataset(target_mode), epochs=args.epochs, verbose=2)

        # why: the model is trained on float32 [0,1] inputs, but the staged
        # images are uint8 [0,255]; feeding raw uint8 to predict() saturated
        # the BN running stats and produced all-ones garbage heatmaps (the
        # deployed v6 sanity check scored 2.9px only after normalizing).
        test_inputs = test_slices["test_1"]["images"].astype(np.float32) / 255.0
        pred = model.predict(test_inputs, batch_size=16, verbose=0)
        if decode_mode == "heatmap112":
            metrics = evaluate_heatmaps_112(pred, test_slices["test_1"]["center"], test_slices["test_1"]["tip"])
        elif decode_mode == "offmap":
            metrics = decode_offmap_outputs(pred, test_slices["test_1"]["center"], test_slices["test_1"]["tip"])
        elif decode_mode == "offset":
            metrics = decode_offset_outputs(pred, test_slices["test_1"]["center"], test_slices["test_1"]["tip"])
        elif decode_mode == "polar":
            metrics = decode_polar_outputs(pred, test_slices["test_1"]["center"], test_slices["test_1"]["tip"])
        else:
            metrics = evaluate_heatmaps(pred, test_slices["test_1"]["center"], test_slices["test_1"]["tip"])
        leaderboard[name] = {
            "params": n_params,
            "test_1_metrics": metrics,
            "score": metrics["center_mae_px"] + metrics["tip_mae_px"],
        }
        print(f"  center_mae {metrics['center_mae_px']:.2f}px tip_mae {metrics['tip_mae_px']:.2f}px "
              f"params {n_params:,}", flush=True)

        del model
        keras.backend.clear_session()
        gc.collect()

    ranked = sorted(leaderboard.items(), key=lambda item: item[1]["score"])
    print("\n=== LEADERBOARD ===", flush=True)
    for name, entry in ranked:
        m = entry["test_1_metrics"]
        print(f"{entry['score']:6.2f}px  {name:14s} center {m['center_mae_px']:5.2f} tip {m['tip_mae_px']:5.2f} {entry['params']:>9,} params", flush=True)
    (args.output / "leaderboard.json").write_text(
        json.dumps({"ranked": ranked, "selected": args.arches}, indent=2) + "\n"
    )


def _iter_samples(images: np.ndarray, targets: np.ndarray) -> object:
    """Return a generator over (image, heatmap pair) host arrays (uint8)."""

    def samples() -> object:
        """Yield one sample at a time from the loaded host arrays."""
        for index in range(len(images)):
            yield images[index].astype(np.float32) / 255.0, targets[index]

    return samples


if __name__ == "__main__":
    main()
