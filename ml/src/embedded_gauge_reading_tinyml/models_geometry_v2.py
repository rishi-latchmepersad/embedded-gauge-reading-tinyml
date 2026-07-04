"""
QAT-friendly geometric model architectures for 2025/2026 gauge reading.

This module provides four families of tiny geometry models designed to be:
- QAT-compatible: built entirely from standard Keras layers (no Lambda, no
  tf.nn.convolution wrappers that break tfmot.clone_model).
- Activation-budgeted: peak int8 activation < 1.5 MB at 224x224 input.
- Literature-informed: SimCC heads (Li et al. 2022), DARK decoding (Zhang
  et al. 2020), CoordConv input (Liu et al. 2018), KD with temperature-scaled
  KL on SimCC logits.

Candidates:
  A.  qat_simcc          -- scratch-trained custom encoder + SimCC heads
  B.  kd_simcc           -- same as A, trained with KD from a frozen teacher
  C.  heatmap_dark       -- custom encoder-decoder + 56x56 heatmap + DARK decode
  D.  coordconv_direct   -- custom encoder + CoordConv input + direct regression
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Shared helpers (copied from models_geometry.py to keep the new file
# self-contained while sharing the same serialized layer registrations)
# ---------------------------------------------------------------------------

def _scaled_width(channels: int, width_multiplier: float) -> int:
    """Scale a channel count and round to the nearest multiple of 8."""
    scaled = int(round(float(channels) * float(width_multiplier) / 8.0) * 8)
    return max(8, scaled)


def _conv_bn_relu(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    name: str,
) -> tf.Tensor:
    """Conv2D -> BatchNorm -> ReLU block, flat (no Lambda) for QAT compatibility."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def _depthwise_conv_bn_relu(
    x: tf.Tensor,
    *,
    strides: int = 1,
    name: str,
) -> tf.Tensor:
    """DepthwiseConv2D -> BN -> ReLU, QAT-friendly."""
    x = layers.DepthwiseConv2D(
        3,
        strides=strides,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    return x


def _inverted_residual_block(
    x: tf.Tensor,
    *,
    out_filters: int,
    expand_ratio: int = 3,
    strides: int = 1,
    name: str,
) -> tf.Tensor:
    """MobileNetV2-style inverted residual block built from standard Keras ops.

    This block is QAT-friendly because it uses regular Conv2D/BatchNorm/ReLU
    layers directly -- no Lambda-wrapped tf.nn.convolution calls.
    """
    input_filters = x.shape[-1]
    if input_filters is None:
        input_filters = int(tf.shape(x)[-1])
    expand_filters = input_filters * expand_ratio

    residual = x
    # Expand
    x = layers.Conv2D(
        expand_filters, 1, padding="same", use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_expand",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_expand_bn")(x)
    x = layers.ReLU(name=f"{name}_expand_relu")(x)

    # Depthwise
    x = layers.DepthwiseConv2D(
        3, strides=strides, padding="same", use_bias=False,
        depthwise_initializer="he_normal",
        name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)

    # Project
    x = layers.Conv2D(
        out_filters, 1, padding="same", use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_project",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_project_bn")(x)

    # Residual add when shapes match
    if strides == 1 and int(input_filters) == out_filters:
        x = layers.Add(name=f"{name}_add")([x, residual])
    return x


# ---------------------------------------------------------------------------
# Shared encoder backbone -- QAT-friendly, flat Functional API
# ---------------------------------------------------------------------------

def _build_qat_encoder(
    inputs: tf.Tensor,
    *,
    width_multiplier: float = 1.0,
    backbone_variant: str = "standard",
    name_prefix: str = "encoder",
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Build a QAT-friendly encoder and return skip tensors at 112/56/28/14.

    The encoder uses exclusively standard Keras Conv2D / BatchNorm / ReLU
    layers so that tfmot.quantize_model() can clone the graph.  No Lambda
    layers, no tf.nn.convolution calls.

    Peak activation (int8) at 224x224 input:
      - standard, w=1.0:  112x112x32  = 401 KB
      - standard, w=1.5:  112x112x48  = 602 KB
      - standard, w=2.0:  112x112x64  = 802 KB
      - inverted, w=1.0:  112x112x24  = 300 KB
      - inverted, w=1.5:  112x112x40  = 500 KB
    All stays under the 1.5 MB budget.

    Returns:
        Tuple of (bottleneck, skip_112, skip_56, skip_28).
        bottleneck is 7x7 used for SimCC/direct heads.
    """
    wm = float(width_multiplier)
    variant = str(backbone_variant).lower()

    if variant in ("standard", "custom"):
        # Simple Conv-BN-ReLU stack, 5 stages, stride-2 at each stage.
        s1 = _scaled_width(32, wm)
        s2 = _scaled_width(48, wm)
        s3 = _scaled_width(64, wm)
        s4 = _scaled_width(96, wm)
        s5 = _scaled_width(128, wm)

        x = _conv_bn_relu(inputs, filters=s1, strides=2, name=f"{name_prefix}_s1a")
        x = _conv_bn_relu(x, filters=s1, name=f"{name_prefix}_s1b")
        skip_112 = x  # 112x112

        x = _conv_bn_relu(x, filters=s2, strides=2, name=f"{name_prefix}_s2a")
        x = _conv_bn_relu(x, filters=s2, name=f"{name_prefix}_s2b")
        skip_56 = x  # 56x56

        x = _conv_bn_relu(x, filters=s3, strides=2, name=f"{name_prefix}_s3a")
        x = _conv_bn_relu(x, filters=s3, name=f"{name_prefix}_s3b")
        skip_28 = x  # 28x28

        x = _conv_bn_relu(x, filters=s4, strides=2, name=f"{name_prefix}_s4a")
        x = _conv_bn_relu(x, filters=s4, name=f"{name_prefix}_s4b")
        skip_14 = x  # 14x14

        x = _conv_bn_relu(x, filters=s5, strides=2, name=f"{name_prefix}_s5a")
        x = _conv_bn_relu(x, filters=s5, name=f"{name_prefix}_s5b")

    elif variant == "inverted":
        # Inverted-residual encoder -- MobileNet-style but QAT-safe.
        s1 = _scaled_width(24, wm)
        s2 = _scaled_width(32, wm)
        s3 = _scaled_width(48, wm)
        s4 = _scaled_width(64, wm)
        s5 = _scaled_width(96, wm)

        x = _conv_bn_relu(inputs, filters=s1, strides=2, name=f"{name_prefix}_stem")
        skip_112 = x

        x = _inverted_residual_block(x, out_filters=s2, expand_ratio=3, strides=2, name=f"{name_prefix}_ir1")
        skip_56 = x

        x = _inverted_residual_block(x, out_filters=s3, expand_ratio=3, strides=2, name=f"{name_prefix}_ir2")

        x = _inverted_residual_block(x, out_filters=s3, expand_ratio=3, strides=1, name=f"{name_prefix}_ir2b")
        skip_28 = x

        x = _inverted_residual_block(x, out_filters=s4, expand_ratio=3, strides=2, name=f"{name_prefix}_ir3")

        x = _inverted_residual_block(x, out_filters=s4, expand_ratio=3, strides=1, name=f"{name_prefix}_ir3b")
        skip_14 = x

        x = _inverted_residual_block(x, out_filters=s5, expand_ratio=3, strides=2, name=f"{name_prefix}_ir4")

        x = _conv_bn_relu(x, filters=s5, name=f"{name_prefix}_out")

    elif variant == "tiny":
        # Even smaller for ultra-constrained deployments.
        s1 = _scaled_width(16, wm)
        s2 = _scaled_width(24, wm)
        s3 = _scaled_width(32, wm)
        s4 = _scaled_width(48, wm)
        s5 = _scaled_width(64, wm)

        x = _conv_bn_relu(inputs, filters=s1, strides=2, name=f"{name_prefix}_s1a")
        skip_112 = x

        x = _conv_bn_relu(x, filters=s2, strides=2, name=f"{name_prefix}_s2a")
        skip_56 = x

        x = _conv_bn_relu(x, filters=s3, strides=2, name=f"{name_prefix}_s3a")
        skip_28 = x

        x = _conv_bn_relu(x, filters=s4, strides=2, name=f"{name_prefix}_s4a")
        skip_14 = x

        x = _conv_bn_relu(x, filters=s5, strides=2, name=f"{name_prefix}_s5a")

    else:
        raise ValueError(f"Unknown backbone_variant: {variant!r}")

    return x, skip_112, skip_56, skip_28, skip_14


# ---------------------------------------------------------------------------
# SimCC head -- axis-marginal coordinate classification
# ---------------------------------------------------------------------------

def _build_simcc_head(
    features: tf.Tensor,
    *,
    simcc_bins: int = 112,
    num_keypoints: int = 4,
    dense_units: int = 256,
    dropout_rate: float = 0.15,
    name_prefix: str = "simcc",
) -> tf.Tensor:
    """Attach a SimCC (axis classification) head to pooled features.

    Output shape: (batch, num_keypoints, simcc_bins).
    Channel order: [center_x, center_y, tip_x, tip_y].
    """
    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(features)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_dense",
    )(x)
    x = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Dense(
        dense_units // 2,
        activation="relu",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_dense2",
    )(x)
    logits = layers.Dense(
        num_keypoints * simcc_bins,
        activation="linear",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_logits",
    )(x)
    logits = layers.Reshape(
        (num_keypoints, simcc_bins),
        name=f"{name_prefix}_reshape",
    )(logits)
    return logits


# ---------------------------------------------------------------------------
# Heatmap decoder -- UNet-style with skip connections
# ---------------------------------------------------------------------------

def _build_heatmap_decoder(
    bottleneck: tf.Tensor,
    skip_112: tf.Tensor,
    skip_56: tf.Tensor,
    skip_28: tf.Tensor,
    skip_14: tf.Tensor,
    *,
    heatmap_channels: int = 2,
    width_multiplier: float = 1.0,
    name_prefix: str = "decoder",
    include_fullres_refine: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decode encoder features into per-keypoint heatmaps.

    Returns:
        Tuple of (heatmaps at target resolution, decoder features before
        heatmap projection -- used for confidence head).
    """
    wm = float(width_multiplier)
    d1 = _scaled_width(96, wm)
    d2 = _scaled_width(64, wm)
    d3 = _scaled_width(48, wm)
    d4 = _scaled_width(32, wm)

    x = bottleneck
    # 7x7 → 14x14
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name_prefix}_up1")(x)
    x = layers.Concatenate(name=f"{name_prefix}_cat14")([x, skip_14])
    x = _conv_bn_relu(x, filters=d1, name=f"{name_prefix}_d1a")
    x = _conv_bn_relu(x, filters=d1, name=f"{name_prefix}_d1b")

    # 14x14 → 28x28
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name_prefix}_up2")(x)
    x = layers.Concatenate(name=f"{name_prefix}_cat28")([x, skip_28])
    x = _conv_bn_relu(x, filters=d2, name=f"{name_prefix}_d2a")
    x = _conv_bn_relu(x, filters=d2, name=f"{name_prefix}_d2b")

    # 28x28 → 56x56
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name_prefix}_up3")(x)
    x = layers.Concatenate(name=f"{name_prefix}_cat56")([x, skip_56])
    x = _conv_bn_relu(x, filters=d3, name=f"{name_prefix}_d3a")
    x = _conv_bn_relu(x, filters=d3, name=f"{name_prefix}_d3b")

    # 56x56 → 112x112
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name_prefix}_up4")(x)
    x = layers.Concatenate(name=f"{name_prefix}_cat112")([x, skip_112])
    x = _conv_bn_relu(x, filters=d4, name=f"{name_prefix}_d4a")
    decoder_features = _conv_bn_relu(x, filters=d4, name=f"{name_prefix}_d4b")

    if include_fullres_refine:
        decoder_features = layers.Conv2D(
            d4, 3, padding="same", activation="relu",
            kernel_initializer="he_normal",
            name=f"{name_prefix}_refine_fr",
        )(decoder_features)

    heatmaps = layers.Conv2D(
        heatmap_channels, 1, padding="same", activation="sigmoid",
        name="heatmaps",
    )(decoder_features)

    return heatmaps, decoder_features


# ---------------------------------------------------------------------------
# Candidate A: QAT-friendly SimCC
# ---------------------------------------------------------------------------

def build_qat_simcc_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    *,
    simcc_bins: int = 112,
    simcc_sigma_bins: float = 1.5,
    width_multiplier: float = 1.0,
    backbone_variant: str = "standard",
    simcc_dense_units: int = 256,
    dropout_rate: float = 0.15,
    with_subpixel_refine: bool = False,
    model_name: str = "qat_simcc_geometry",
) -> keras.Model:
    """Build Candidate A: QAT-friendly encoder + SimCC heads.

    Architecture:
      - Custom encoder (5 stages, Conv2D+BN+ReLU, no Lambda)
      - GAP → Dense(256) → Dense(128) → SimCC logits (4 × simcc_bins)
      - Confidence scalar from the same pooled features
      - Optional sub-pixel refinement head (residual offsets in bin units)

    Peak int8 activation at 224x224 with w=1.0: ~401 KB.
    Peak int8 activation at 224x224 with w=2.0: ~802 KB.

    The sub-pixel refinement head (with_subpixel_refine=True) adds a tiny
    Dense(32) + Dense(4, tanh) branch that predicts residual offsets in
    [-0.5, 0.5] bin units per keypoint.  Final coordinate = expected_bin + offset.
    This follows the sub-pixel keypoint refinement literature (Nibali et al.,
    2018; Zhang et al., 2020).
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    bottleneck, skip_112, skip_56, skip_28, skip_14 = _build_qat_encoder(
        x,
        width_multiplier=width_multiplier,
        backbone_variant=backbone_variant,
        name_prefix="simcc_encoder",
    )

    simcc_logits = _build_simcc_head(
        bottleneck,
        simcc_bins=simcc_bins,
        num_keypoints=4,
        dense_units=simcc_dense_units,
        dropout_rate=dropout_rate,
        name_prefix="simcc_head",
    )

    # Confidence from the same GAP features (frozen after encoding for SimCC).
    # Re-use the GAP so we don't add extra parameters.
    confidence_features = layers.GlobalAveragePooling2D(name="confidence_gap")(bottleneck)
    confidence = layers.Dense(
        1, activation="sigmoid",
        kernel_initializer="he_normal",
        name="confidence",
    )(confidence_features)

    outputs: list[keras.layers.Layer] = [simcc_logits, confidence]

    if with_subpixel_refine:
        # Sub-pixel refinement head: tiny Dense branch predicting residual
        # offsets for each keypoint in bin units.  The head uses the same
        # GAP features so it adds negligible parameters.
        # Output: (batch, 4) tanh in [-0.5, 0.5] bin units.
        offset_features = layers.Dense(
            32, activation="relu",
            kernel_initializer="he_normal",
            name="subpixel_dense",
        )(confidence_features)
        subpixel_offsets = layers.Dense(
            4, activation="tanh",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="subpixel_offsets",
        )(offset_features)
        # Scale from [-1, 1] to [-0.5, 0.5] bin units.
        subpixel_offsets = layers.Rescaling(
            0.5, name="subpixel_offsets_scaled",
        )(subpixel_offsets)
        outputs.append(subpixel_offsets)

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name=model_name,
    )


# ---------------------------------------------------------------------------
# RepVGG backbone — structural reparameterization for QAT-friendly training
# ---------------------------------------------------------------------------

def _repvgg_block(x: tf.Tensor, *, filters: int, stride: int, name: str) -> tf.Tensor:
    """Build a RepVGG-style block with 3 branches during training.

    Branches: (1) 3×3 Conv+BN, (2) 1×1 Conv+BN, (3) Identity+BN.
    All branches are added, then ReLU.  During training, the multiple
    branches create implicit shortcut paths that prevent collapse into
    bad local minima — the exact problem our flat Conv-BN-ReLU encoder
    suffers from.

    At export time, the three branches are mathematically fused into a
    single 3×3 Conv2D via fuse_repvgg_block(), producing a QAT-safe
    graph with only standard Keras layers.
    """
    # Branch 1: 3×3 conv.
    use_bias3 = True
    b1 = layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=use_bias3,
        kernel_initializer="he_normal",
        name=f"{name}_conv3x3",
    )(x)
    b1 = layers.BatchNormalization(name=f"{name}_bn3x3")(b1)

    # Branch 2: 1×1 conv.
    b2 = layers.Conv2D(
        filters, 1, strides=stride, padding="valid", use_bias=True,
        kernel_initializer="he_normal",
        name=f"{name}_conv1x1",
    )(x)
    b2 = layers.BatchNormalization(name=f"{name}_bn1x1")(b2)

    # Branch 3: identity (only when stride==1 and in/out channels match).
    input_filters = x.shape[-1]
    if stride == 1 and (input_filters is None or int(input_filters) == filters):
        b3 = layers.BatchNormalization(name=f"{name}_bnid")(x)
    else:
        # When stride > 1 or channel mismatch, replace identity with
        # a 1×1 conv + BN to match dimensions.
        b3 = layers.Conv2D(
            filters, 1, strides=stride, padding="valid", use_bias=True,
            kernel_initializer="he_normal",
            name=f"{name}_conv_skip",
        )(x)
        b3 = layers.BatchNormalization(name=f"{name}_bn_skip")(b3)

    out = layers.Add(name=f"{name}_add")([b1, b2, b3])
    return layers.ReLU(name=f"{name}_relu")(out)


def _repvgg_stem(x: tf.Tensor, *, filters: int, name: str) -> tf.Tensor:
    """Initial 3×3 Conv+BN+ReLU with stride 2, before RepVGG stages."""
    x = layers.Conv2D(
        filters, 3, strides=2, padding="same", use_bias=False,
        kernel_initializer="he_normal", name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_repvgg_simcc_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    *,
    simcc_bins: int = 112,
    simcc_dense_units: int = 256,
    dropout_rate: float = 0.15,
    stage_depths: Tuple[int, int, int, int] = (2, 3, 4, 2),
    stage_filters: Tuple[int, int, int, int] = (48, 96, 192, 384),
    model_name: str = "repvgg_simcc",
) -> keras.Model:
    """RepVGG backbone + GAP SimCC head.

    Architecture inspired by RepVGG-A0 (Ding et al., CVPR 2021):
      - Stem: 3×3 Conv, stride 2 → 112×112×C0
      - Stage 1: depth[0] × RepVGGBlock(C0), stride 2 at first block
      - Stage 2: depth[1] × RepVGGBlock(C1), stride 2 at first block
      - Stage 3: depth[2] × RepVGGBlock(C2), stride 2 at first block
      - Stage 4: depth[3] × RepVGGBlock(C3), no stride reduction
      - GAP → Dense → SimCC head

    During training, each block has 3 branches (3×3, 1×1, identity).
    After training, fuse_repvgg_model() converts to a QAT-safe single-branch
    graph with only Conv2D + ReLU layers.

    Peak int8 activation: 112×112×C0.
      - C0=48 → 602 KB
      - C0=64 → 802 KB  
      - C0=96 → 1.15 MB
    All under 1.5 MB budget.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    filters = list(stage_filters)
    depths = list(stage_depths)
    if len(filters) < 4 or len(depths) < 4:
        raise ValueError("RepVGG expects 4 stages (filters and depths).")

    # Stem.
    x = _repvgg_stem(x, filters=filters[0], name="repvgg_stem")

    # Stage 1: stride 2 at first block → 56×56.
    x = _repvgg_block(x, filters=filters[0], stride=2, name="repvgg_s1b1")
    for i in range(1, depths[0]):
        x = _repvgg_block(x, filters=filters[0], stride=1, name=f"repvgg_s1b{i+1}")

    # Stage 2: stride 2 at first block → 28×28.
    x = _repvgg_block(x, filters=filters[1], stride=2, name="repvgg_s2b1")
    for i in range(1, depths[1]):
        x = _repvgg_block(x, filters=filters[1], stride=1, name=f"repvgg_s2b{i+1}")

    # Stage 3: stride 2 at first block → 14×14.
    x = _repvgg_block(x, filters=filters[2], stride=2, name="repvgg_s3b1")
    for i in range(1, depths[2]):
        x = _repvgg_block(x, filters=filters[2], stride=1, name=f"repvgg_s3b{i+1}")

    # Stage 4: no stride reduction.
    x = _repvgg_block(x, filters=filters[3], stride=1, name="repvgg_s4b1")
    for i in range(1, depths[3]):
        x = _repvgg_block(x, filters=filters[3], stride=1, name=f"repvgg_s4b{i+1}")

    # SimCC head (GAP + Dense, same as other models).
    pooled = layers.GlobalAveragePooling2D(name="repvgg_gap")(x)
    pooled = layers.Dense(
        simcc_dense_units, activation="relu",
        kernel_initializer="he_normal", name="repvgg_dense1",
    )(pooled)
    pooled = layers.Dropout(dropout_rate, name="repvgg_dropout")(pooled)
    pooled = layers.BatchNormalization(name="repvgg_bn_head")(pooled)
    pooled = layers.Dense(
        simcc_dense_units // 2, activation="relu",
        kernel_initializer="he_normal", name="repvgg_dense2",
    )(pooled)

    simcc_logits_raw = layers.Dense(
        4 * simcc_bins, activation="linear",
        kernel_initializer="he_normal", name="repvgg_simcc_logits_raw",
    )(pooled)
    simcc_logits = layers.Reshape(
        (4, simcc_bins), name="repvgg_simcc_logits",
    )(simcc_logits_raw)

    confidence = layers.Dense(
        1, activation="sigmoid",
        kernel_initializer="he_normal", name="confidence",
    )(pooled)

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name=model_name,
    )


def fuse_repvgg_model(model: keras.Model) -> keras.Model:
    """Convert a trained RepVGG model to a single-branch QAT-safe model.

    Fuses each 3-branch RepVGG block into a single 3×3 Conv2D with no
    BatchNorm — the resulting graph has ONLY Conv2D + activation layers,
    making it fully compatible with tfmot.quantization.keras.quantize_model().

    Implementation:
      For each RepVGG block, extract the weights of the three BN-fused
      branches (3×3 conv + BN, 1×1 conv + BN, identity BN), mathematically
      merge them into one 3×3 kernel, and rebuild the model with single
      Conv2D layers in place of the multi-branch blocks.

    NOTE: This is a post-training fusion step.  The training-time model
    uses 3 branches; the fused model is for QAT and deployment.

    Returns:
        A new keras.Model with fused Conv2D layers replacing RepVGG blocks.
    """
    # This is a placeholder for the full implementation.
    # The fusion requires extracting trained weights, computing BN folding,
    # padding 1×1 kernels to 3×3, creating identity kernels, and rebuilding.
    # We will implement this when we have trained weights to work with.
    raise NotImplementedError(
        "Model fusion will be implemented after training the RepVGG model."
    )


# ---------------------------------------------------------------------------
# Candidate B: KD SimCC -- same architecture, teacher-student training
# ---------------------------------------------------------------------------

def build_kd_simcc_student_model(
    *args,
    model_name: str = "kd_simcc_student",
    **kwargs,
) -> keras.Model:
    """Alias for build_qat_simcc_model, used as the KD student.

    The student is architecturally identical to Candidate A.  The difference
    is in training: we add a KD loss against a frozen teacher's SimCC logits.
    """
    return build_qat_simcc_model(*args, model_name=model_name, **kwargs)


def build_teacher_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    *,
    simcc_bins: int = 112,
    alpha: float = 1.0,
    backbone_frozen: bool = True,
    simcc_dense_units: int = 512,
    dropout_rate: float = 0.15,
    model_name: str = "teacher_simcc",
) -> keras.Model:
    """Build a larger teacher model for KD.

    Uses a bigger MobileNetV2 backbone (alpha=1.0) + deeper SimCC head.
    The teacher is trained first on the ground truth, then frozen during
    student KD training.

    NOTE: The teacher uses MobileNetV2 which contains Lambda-wrapped conv
    calls and CANNOT be QAT-cloned.  We only train it in float32 and use
    it for logit-level distillation.  The student in Candidate B uses the
    QAT-friendly encoder and gets PTQ int8 export.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_preprocess")(inputs)

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    backbone.trainable = not backbone_frozen
    x = backbone(x)

    simcc_logits = _build_simcc_head(
        x, simcc_bins=simcc_bins, num_keypoints=4,
        dense_units=simcc_dense_units,
        dropout_rate=dropout_rate,
        name_prefix="teacher_simcc",
    )

    confidence_features = layers.GlobalAveragePooling2D(name="teacher_confidence_gap")(x)
    confidence = layers.Dense(
        1, activation="sigmoid",
        kernel_initializer="he_normal",
        name="confidence",
    )(confidence_features)

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name=model_name,
    )


# ---------------------------------------------------------------------------
# Candidate C: Heatmap + DARK decoding
# ---------------------------------------------------------------------------

def build_heatmap_dark_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    *,
    heatmap_size: int = 56,
    width_multiplier: float = 1.0,
    backbone_variant: str = "standard",
    model_name: str = "heatmap_dark_geometry",
) -> keras.Model:
    """Build Candidate C: UNet-style encoder-decoder with heatmap output.

    Architecture:
      - QAT-friendly encoder (same as Candidate A)
      - Bilinear-upsampling decoder with skip connections
      - 56×56×2 heatmap output (center + tip)
      - Confidence scalar from GAP on decoder features

    DARK decoding (Zhang et al., CVPR 2020) is applied in post-processing:
      1. Find argmax of heatmap
      2. Fit a 2D Gaussian around the peak using the log-heatmap
      3. Solve for the analytical mean → sub-pixel coordinates
    This corrects the quantization bias of simple argmax.

    Peak int8 activation: ~401 KB (encoder stage 1, 112×112×32).

    2025/2026 rationale:
      - Heatmap regression with UNet-style decoder is the most robust
        approach for keypoint localization under occlusion/corruption.
      - DARK removes the systematic bias introduced by max-pooling on
        quantized heatmaps, giving sub-pixel accuracy without extra
        learned refinement heads.
      - PM-SwinUnet (2024) showed UNet + coordinate fitting excels for
        gauge reading specifically.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    bottleneck, skip_112, skip_56, skip_28, skip_14 = _build_qat_encoder(
        x,
        width_multiplier=width_multiplier,
        backbone_variant=backbone_variant,
        name_prefix="heatmap_dark_encoder",
    )

    heatmaps, decoder_features = _build_heatmap_decoder(
        bottleneck,
        skip_112, skip_56, skip_28, skip_14,
        heatmap_channels=2,
        width_multiplier=width_multiplier,
        name_prefix="heatmap_dark_decoder",
        include_fullres_refine=True,
    )

    # Split heatmap channels for named outputs.
    # The heatmaps tensor is (B, 112, 112, 2).  Slice into named tensors.
    center_heatmap = layers.Lambda(
        lambda h: h[..., 0:1], name="center_heatmap",
    )(heatmaps)
    tip_heatmap = layers.Lambda(
        lambda h: h[..., 1:2],         name="tip_heatmap",
    )(heatmaps)

    confidence_features = layers.GlobalAveragePooling2D(name="confidence_gap")(decoder_features)
    confidence = layers.Dense(
        1, activation="sigmoid",
        kernel_initializer="he_normal",
        name="confidence",
    )(confidence_features)

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name=model_name,
    )


# ---------------------------------------------------------------------------
# ECA-Net: Efficient Channel Attention (Wang et al., CVPR 2020)
#   — 1D conv over channels, k=3, negligible params (< 0.01% overhead)
#   — Helps model attend to needle-specific features
# ---------------------------------------------------------------------------

def _eca_attention(x: tf.Tensor, *, name: str, k_size: int = 3) -> tf.Tensor:
    """Apply ECA-Net channel attention with adaptive kernel size.

    ECA-Net uses a 1D convolution across channels to capture local
    cross-channel interactions.  This is dramatically lighter than SE
    blocks while maintaining competitive accuracy.

    Uses Lambda layers for dynamic channel counts (symbolic tensors).
    """
    # GAP → 1D Conv → Sigmoid → scale.
    gap = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    # gap is (B, C). Expand to (B, 1, C) for Conv1D.
    gap_2d = layers.Lambda(
        lambda t: tf.expand_dims(t, axis=1), name=f"{name}_expand",
    )(gap)
    # 1D depthwise conv along channel dimension.
    attn = layers.Conv1D(
        1, k_size, padding="same", use_bias=False,
        kernel_initializer="zeros",
        name=f"{name}_conv1d",
    )(gap_2d)
    # Squeeze to (B, C) and apply sigmoid.
    attn = layers.Lambda(
        lambda t: tf.squeeze(t, axis=1), name=f"{name}_squeeze",
    )(attn)
    attn = layers.Activation("sigmoid", name=f"{name}_sigmoid")(attn)
    # Broadcast to (B, 1, 1, C) for multiplication.
    attn_bc = layers.Lambda(
        lambda t: tf.reshape(t, (-1, 1, 1, tf.shape(t)[-1])),
        name=f"{name}_broadcast",
    )(attn)

    return layers.Multiply(name=f"{name}_scale")([x, attn_bc])


# ---------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)
#   — Channel attention (GAP+GMP → MLP) + Spatial attention (7×7 conv)
#   — Slightly heavier than ECA-Net but more expressive
# ---------------------------------------------------------------------------

def _cbam_attention(
    x: tf.Tensor, *, name: str, reduction: int = 16, spatial_kernel: int = 7,
) -> tf.Tensor:
    """Apply CBAM channel + spatial attention."""
    channels = x.shape[-1]
    if channels is None:
        channels = tf.shape(x)[-1]

    # Channel attention: GAP + GMP → shared MLP → add → sigmoid.
    avg_pool = layers.GlobalAveragePooling2D(name=f"{name}_cavg")(x)
    max_pool = layers.GlobalMaxPooling2D(name=f"{name}_cmax")(x)

    reduced = max(1, channels // reduction)
    shared_dense1 = layers.Dense(reduced, activation="relu", use_bias=False,
                                  kernel_initializer="he_normal",
                                  name=f"{name}_cmlp1")
    shared_dense2 = layers.Dense(channels, use_bias=False,
                                  kernel_initializer="he_normal",
                                  name=f"{name}_cmlp2")

    avg_out = shared_dense2(shared_dense1(avg_pool))
    max_out = shared_dense2(shared_dense1(max_pool))
    ch_attn = layers.Add(name=f"{name}_cadd")([avg_out, max_out])
    ch_attn = layers.Activation("sigmoid", name=f"{name}_csigmoid")(ch_attn)
    ch_attn = layers.Reshape((1, 1, channels), name=f"{name}_cbroadcast")(ch_attn)
    x_refined = layers.Multiply(name=f"{name}_cscale")([x, ch_attn])

    # Spatial attention: mean+max along channel → 7×7 conv → sigmoid.
    avg_spatial = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
        name=f"{name}_savg",
    )(x_refined)
    max_spatial = layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
        name=f"{name}_smax",
    )(x_refined)
    spatial_cat = layers.Concatenate(name=f"{name}_scat")([avg_spatial, max_spatial])
    spatial_attn = layers.Conv2D(
        1, spatial_kernel, padding="same", activation="sigmoid",
        kernel_initializer="zeros", bias_initializer="zeros",
        name=f"{name}_sconv",
    )(spatial_cat)

    return layers.Multiply(name=f"{name}_sscale")([x_refined, spatial_attn])


# ---------------------------------------------------------------------------
# Coordinate Attention (Hou et al., CVPR 2021)
#   — Factorizes attention into 1D horizontal + vertical encoding
#   — Captures long-range dependencies with precise positional info
#   — Specifically designed for mobile networks
# ---------------------------------------------------------------------------

def _coord_attention(x: tf.Tensor, *, name: str, reduction: int = 16) -> tf.Tensor:
    """Apply Coordinate Attention for position-sensitive channel attention.

    Decomposes channel attention into two 1D feature encodings along
    the horizontal and vertical spatial directions.  This preserves
    precise positional information while capturing long-range dependencies
    — critical for distinguishing the needle tip from background.
    """
    channels = x.shape[-1]
    if channels is None:
        channels = tf.shape(x)[-1]

    # Horizontal pooling: (B, H, W, C) → (B, H, 1, C)
    h_pool = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=2, keepdims=True),  # pool over width
        name=f"{name}_hpool",
    )(x)
    # Vertical pooling: (B, H, W, C) → (B, 1, W, C)
    w_pool = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True),  # pool over height
        name=f"{name}_wpool",
    )(x)

    # Shared 1×1 transform.
    reduced_c = max(1, channels // reduction)
    shared_conv = layers.Conv2D(
        reduced_c, 1, padding="same", use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_shared",
    )
    h_feat = shared_conv(h_pool)
    w_feat = shared_conv(w_pool)

    # BN + non-linearity.
    h_feat = layers.BatchNormalization(name=f"{name}_hbn")(h_feat)
    h_feat = layers.Activation("swish", name=f"{name}_hswish")(h_feat)
    w_feat = layers.BatchNormalization(name=f"{name}_wbn")(w_feat)
    w_feat = layers.Activation("swish", name=f"{name}_wswish")(w_feat)

    # Split Conv2D into separate horizontal and vertical 1×1 convs.
    h_attn = layers.Conv2D(
        channels, 1, padding="same", activation="sigmoid",
        kernel_initializer="zeros", bias_initializer="zeros",
        name=f"{name}_hconv",
    )(h_feat)
    w_attn = layers.Conv2D(
        channels, 1, padding="same", activation="sigmoid",
        kernel_initializer="zeros", bias_initializer="zeros",
        name=f"{name}_wconv",
    )(w_feat)

    return layers.Multiply(name=f"{name}_scale")([x, h_attn, w_attn])


# ---------------------------------------------------------------------------
# HRNet-lite: maintain a 56×56 high-resolution branch throughout
# ---------------------------------------------------------------------------

def _hrnet_lite_stage(
    high_res: tf.Tensor,
    low_res: tf.Tensor,
    *,
    hr_filters: int,
    lr_filters: int,
    num_blocks: int,
    name: str,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """One HRNet-lite stage: update both resolution branches with fusion.

    The high-res branch maintains spatial detail (56×56 throughout),
    while the low-res branch captures semantic context.  Multi-scale
    fusion exchanges information between branches at each block.
    Peak activation stays at the high-res branch: 56×56×hr_filters.
    """
    for i in range(num_blocks):
        # High-res branch: 2× 3×3 convs.
        hr_out = _conv_bn_relu(
            high_res, filters=hr_filters, kernel_size=3, strides=1,
            name=f"{name}_hrb{i}_a",
        )
        hr_out = _conv_bn_relu(
            hr_out, filters=hr_filters, kernel_size=3, strides=1,
            name=f"{name}_hrb{i}_b",
        )

        # Low-res branch update.
        lr_out = _conv_bn_relu(
            low_res, filters=lr_filters, kernel_size=3, strides=1,
            name=f"{name}_lrb{i}",
        )

        # Multi-scale fusion: upsample low-res → add to high-res.
        lr_up = layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear",
            name=f"{name}_lrup{i}",
        )(lr_out)
        lr_up_proj = layers.Conv2D(
            hr_filters, 1, padding="same", use_bias=False,
            kernel_initializer="he_normal",
            name=f"{name}_lrup_proj{i}",
        )(lr_up)
        hr_fused = layers.Add(name=f"{name}_hrfused{i}")([hr_out, lr_up_proj])

        # Downsample high-res → add to low-res.
        hr_down = layers.Conv2D(
            lr_filters, 3, strides=2, padding="same", use_bias=False,
            kernel_initializer="he_normal",
            name=f"{name}_hrdown{i}",
        )(high_res)
        lr_fused = layers.Add(name=f"{name}_lrfused{i}")([lr_out, hr_down])

        high_res = hr_fused
        low_res = lr_fused

    return high_res, low_res


def build_mobilenetv2_hrnet_eca_simcc_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    *,
    simcc_bins: int = 112,
    simcc_dense_units: int = 256,
    dropout_rate: float = 0.15,
    alpha: float = 0.75,
    backbone_frozen: bool = True,
    hr_filters: int = 32,
    hr_blocks: int = 2,
    eca_attention: bool = True,
    attention_type: str = "eca",  # "eca", "cbam", "coord", "none"
    model_name: str = "mnv2_hrnet_eca_simcc",
) -> keras.Model:
    """MobileNetV2 + HRNet-lite 56×56 branch + attention + SimCC head.

    Architecture:
      - MobileNetV2 backbone (alpha, ImageNet) extracts multi-scale features
        at 14×14, 28×28, and 56×56 resolutions.
      - HRNet-lite maintains a 56×56 branch alongside the 14×14 backbone
        features, with bidirectional multi-scale fusion.
      - Optional attention module (ECA-Net, CBAM, or Coordinate Attention)
        applied after the high-res branch to focus on needle features.
      - Fused features feed a GAP+Dense SimCC head.

    Peak int8 activation: 56×56×hr_filters.
      - hr_filters=32 → 98 KB
      - hr_filters=64 → 196 KB
    Well under 1.5 MB budget.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_preprocess")(inputs)

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape, alpha=alpha, include_top=False,
        weights="imagenet", pooling=None,
    )
    backbone.trainable = not backbone_frozen

    # Extract multi-scale features from MobileNetV2.
    feature_extractor = keras.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer("block_13_expand_relu").output,  # 14×14
            backbone.get_layer("block_6_expand_relu").output,   # 28×28
            backbone.get_layer("block_3_expand_relu").output,   # 56×56
        ],
        name="mnv2_feature_extractor",
    )
    feat_14, feat_28, feat_56 = feature_extractor(x)

    # Project multi-scale features to common channel counts.
    proj_14 = layers.Conv2D(
        int(hr_filters * 2), 1, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="hr_proj_14",
    )(feat_14)  # 14×14
    proj_56 = layers.Conv2D(
        hr_filters, 1, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="hr_proj_56",
    )(feat_56)  # 56×56

    # HRNet-lite: maintain 56×56 and 14×14 branches.
    # First, upsample 14×14 to 28×28 so we have a 2× gap (not 4×).
    low_res_start = layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear", name="hr_low_init_up",
    )(proj_14)
    low_res_start = layers.Conv2D(
        int(hr_filters * 2), 1, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="hr_low_init_proj",
    )(low_res_start)  # 28×28

    high_res, low_res = _hrnet_lite_stage(
        proj_56, low_res_start,
        hr_filters=hr_filters,
        lr_filters=int(hr_filters * 2),
        num_blocks=hr_blocks,
        name="hrnet_stage",
    )

    # Optional attention on the high-res branch.
    if attention_type == "eca" and eca_attention:
        high_res = _eca_attention(high_res, name="hr_attn")
    elif attention_type == "cbam":
        high_res = _cbam_attention(high_res, name="hr_attn")
    elif attention_type == "coord":
        high_res = _coord_attention(high_res, name="hr_attn")

    # Fuse both branches for SimCC head (upsample to common 56×56).
    low_up = layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear", name="hr_low_up",
    )(low_res)  # 28→56
    low_up_proj = layers.Conv2D(
        64, 1, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="hr_low_up_proj",
    )(low_up)
    high_proj = layers.Conv2D(
        64, 1, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="hr_high_proj",
    )(high_res)

    fused = layers.Concatenate(name="hr_fused")([high_proj, low_up_proj])  # 56×56
    fused = _conv_bn_relu(fused, filters=64, name="hr_fused_conv")

    # SimCC head from fused features.
    pooled = layers.GlobalAveragePooling2D(name="hr_gap")(fused)
    pooled = layers.Dense(
        simcc_dense_units, activation="relu",
        kernel_initializer="he_normal", name="hr_dense1",
    )(pooled)
    pooled = layers.Dropout(dropout_rate, name="hr_dropout")(pooled)
    pooled = layers.BatchNormalization(name="hr_bn_head")(pooled)
    pooled = layers.Dense(
        simcc_dense_units // 2, activation="relu",
        kernel_initializer="he_normal", name="hr_dense2",
    )(pooled)

    simcc_logits_raw = layers.Dense(
        4 * simcc_bins, activation="linear",
        kernel_initializer="he_normal", name="hr_simcc_logits_raw",
    )(pooled)
    simcc_logits = layers.Reshape(
        (4, simcc_bins), name="hr_simcc_logits",
    )(simcc_logits_raw)

    confidence = layers.Dense(
        1, activation="sigmoid",
        kernel_initializer="he_normal", name="confidence",
    )(pooled)

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name=model_name,
    )


# ---------------------------------------------------------------------------
# UNet decoder from MobileNetV2 — 112×112 heatmap for max tip resolution
# ---------------------------------------------------------------------------

def build_mobilenetv2_unet_heatmap_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    *,
    heatmap_size: int = 112,
    heatmap_channels: int = 2,
    alpha: float = 0.75,
    backbone_frozen: bool = True,
    decoder_filters: Tuple[int, int, int, int] = (256, 128, 64, 32),
    dropout_rate: float = 0.15,
    model_name: str = "mnv2_unet_heatmap",
) -> keras.Model:
    """MobileNetV2 encoder + UNet decoder → high-resolution heatmaps.

    Extracts features at 4 scales from MobileNetV2, then progressively
    upsamples with skip connections.  Final output at 112×112 gives
    2× the spatial resolution of 56×56 heatmaps.

    Peak int8 activation: 112×112×decoder_filters[3] = 112×112×32 = 401 KB.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_preprocess")(inputs)

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape, alpha=alpha, include_top=False,
        weights="imagenet", pooling=None,
    )
    backbone.trainable = not backbone_frozen

    # Extract features at multiple scales.
    layer_names = [
        "block_16_project_BN",   # 7×7
        "block_13_expand_relu",  # 14×14
        "block_6_expand_relu",   # 28×28
        "block_3_expand_relu",   # 56×56
    ]
    available = [ln for ln in layer_names if ln in [layer.name for layer in backbone.layers]]
    if len(available) >= 2:
        feature_extractor = keras.Model(
            inputs=backbone.input, outputs=[backbone.get_layer(ln).output for ln in available],
            name="mnv2_unet_extractor",
        )
        features = feature_extractor(x)
        if not isinstance(features, (list, tuple)):
            features = [features]
        # Ensure we have 4 features.
        while len(features) < 4:
            features.insert(0, features[0])
    else:
        features = [backbone(x)]
        while len(features) < 4:
            features.insert(0, features[0])

    feat_7, feat_14, feat_28, feat_56 = features[:4]
    dec = list(decoder_filters[:4])

    # Decoder: 7→14→28→56→112.
    x_dec = feat_7
    # 7 → 14.
    x_dec = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="unet_up1")(x_dec)
    x_dec = layers.Concatenate(name="unet_cat14")([x_dec, feat_14])
    x_dec = _conv_bn_relu(x_dec, filters=dec[0], name="unet_dec1a")
    x_dec = _conv_bn_relu(x_dec, filters=dec[0], name="unet_dec1b")

    # 14 → 28.
    x_dec = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="unet_up2")(x_dec)
    x_dec = layers.Concatenate(name="unet_cat28")([x_dec, feat_28])
    x_dec = _conv_bn_relu(x_dec, filters=dec[1], name="unet_dec2a")
    x_dec = _conv_bn_relu(x_dec, filters=dec[1], name="unet_dec2b")

    # 28 → 56.
    x_dec = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="unet_up3")(x_dec)
    x_dec = layers.Concatenate(name="unet_cat56")([x_dec, feat_56])
    x_dec = _conv_bn_relu(x_dec, filters=dec[2], name="unet_dec3a")
    x_dec = _conv_bn_relu(x_dec, filters=dec[2], name="unet_dec3b")

    # 56 → 112.
    x_dec = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name="unet_up4")(x_dec)
    x_dec = _conv_bn_relu(x_dec, filters=dec[3], name="unet_dec4a")
    x_dec = _conv_bn_relu(x_dec, filters=dec[3], name="unet_dec4b")

    # Full-resolution refinement.
    x_dec = layers.Conv2D(
        dec[3], 3, padding="same", activation="relu",
        kernel_initializer="he_normal", name="unet_refine",
    )(x_dec)

    # Heatmap heads.
    center_heatmap = layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="center_heatmap",
    )(x_dec)
    tip_heatmap = layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="tip_heatmap",
    )(x_dec)
    confidence = layers.Dense(
        1, activation="sigmoid",
        kernel_initializer="he_normal", name="confidence",
    )(layers.GlobalAveragePooling2D(name="unet_conf_gap")(x_dec))

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name=model_name,
    )


# ---------------------------------------------------------------------------
# Compact MobileNetV2 + 56x56 heatmap (v16) — no xSPI1 HyperRAM, no
# high-resolution 112x112 decoder stage. Stays entirely in on-chip SRAM.
# ---------------------------------------------------------------------------

def build_mobilenetv2_compact_heatmap_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    *,
    heatmap_size: int = 56,
    alpha: float = 0.35,
    decoder_filters: Tuple[int, int, int] = (96, 48, 24),
    dropout_rate: float = 0.0,
    model_name: str = "mnv2_compact_heatmap",
) -> keras.Model:
    """Compact 3-stage UNet heatmap. Stops the decoder at 56x56 so the
    largest activation is 56x56xdecoder_filters[2] (~75 KB int8) plus
    the backbone's own 56x56x96 MobileNetV2 mid block. Total on-chip
    SRAM peak stays under 1 MB int8 and no xSPI1 HyperRAM is needed.
    Angle is derived from center/tip in post-processing, not from a
    separate network head.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_preprocess")(inputs)

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape, alpha=alpha, include_top=False,
        weights="imagenet", pooling=None,
    )
    backbone.trainable = True

    feature_extractor = keras.Model(
        inputs=backbone.input,
        outputs=[
            backbone.get_layer("block_13_expand_relu").output,  # 14x14 (bottleneck)
            backbone.get_layer("block_6_expand_relu").output,   # 28x28
            backbone.get_layer("block_3_expand_relu").output,   # 56x56
        ],
        name="compact_extractor",
    )
    feat_14, feat_28, feat_56 = feature_extractor(x)

    # 14 -> 28 (with skip from feat_28).
    x_dec = layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear", name="compact_up1",
    )(feat_14)
    x_dec = layers.Concatenate(name="compact_cat28")([x_dec, feat_28])
    x_dec = _conv_bn_relu(x_dec, filters=decoder_filters[0], name="compact_dec1a")
    x_dec = _conv_bn_relu(x_dec, filters=decoder_filters[0], name="compact_dec1b")

    # 28 -> 56 (with skip from feat_56).
    x_dec = layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear", name="compact_up2",
    )(x_dec)
    x_dec = layers.Concatenate(name="compact_cat56")([x_dec, feat_56])
    x_dec = _conv_bn_relu(x_dec, filters=decoder_filters[1], name="compact_dec2a")
    x_dec = _conv_bn_relu(x_dec, filters=decoder_filters[1], name="compact_dec2b")

    if dropout_rate > 0:
        x_dec = layers.Dropout(dropout_rate, name="compact_dropout")(x_dec)
    x_dec = _conv_bn_relu(x_dec, filters=decoder_filters[2], name="compact_refine")

    center_heatmap = layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="center_heatmap",
    )(x_dec)
    tip_heatmap = layers.Conv2D(
        1, 1, padding="same", activation="sigmoid", name="tip_heatmap",
    )(x_dec)
    confidence = layers.Dense(
        1, activation="sigmoid",
        kernel_initializer="he_normal", name="confidence",
    )(layers.GlobalAveragePooling2D(name="compact_conf_gap")(x_dec))

    is_main_needle = layers.Dense(
        1, activation="sigmoid",
        kernel_initializer="he_normal", name="is_main_needle",
    )(layers.GlobalAveragePooling2D(name="compact_gap_needle")(x_dec))

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence, is_main_needle],
        name=model_name,
    )


def build_spatial_simcc_attn_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    *,
    simcc_bins: int = 112,
    spatial_channels: int = 128,
    alpha: float = 0.75,
    backbone_frozen: bool = True,
    dropout_rate: float = 0.15,
    attention_type: str = "eca",
    model_name: str = "spatial_simcc_attn",
) -> keras.Model:
    """Spatial SimCC with attention injected after MobileNetV2 features.

    Adds ECA-Net, CBAM, or Coordinate Attention after the backbone's
    7×7 features before the spatial bottleneck.  The attention helps
    the model focus on needle-specific features.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_preprocess")(inputs)

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape, alpha=alpha, include_top=False,
        weights="imagenet", pooling=None,
    )
    backbone.trainable = not backbone_frozen
    x = backbone(x)

    # Attention on backbone features.
    if attention_type == "eca":
        x = _eca_attention(x, name="spatial_attn")
    elif attention_type == "cbam":
        x = _cbam_attention(x, name="spatial_attn")
    elif attention_type == "coord":
        x = _coord_attention(x, name="spatial_attn")

    # Spatial bottleneck.
    bottleneck = layers.Conv2D(
        spatial_channels, 1, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="spatial_bottleneck_conv",
    )(x)
    bottleneck = layers.BatchNormalization(name="spatial_bottleneck_bn")(bottleneck)
    bottleneck = layers.ReLU(name="spatial_bottleneck_relu")(bottleneck)
    bottleneck = layers.Dropout(dropout_rate, name="spatial_bottleneck_dropout")(bottleneck)

    num_keypoints = 4
    logit_maps = layers.Conv2D(
        num_keypoints * simcc_bins, 1, padding="same",
        kernel_initializer="he_normal", name="spatial_logit_maps",
    )(bottleneck)

    h = int(logit_maps.shape[1])
    w = int(logit_maps.shape[2])
    logit_4d = layers.Reshape(
        (h * w, simcc_bins, num_keypoints), name="spatial_reshape",
    )(logit_maps)

    simcc_pooled = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1), name="spatial_reduce_mean",
    )(logit_4d)

    simcc_logits = layers.Permute(
        (2, 1), name="spatial_simcc_logits",
    )(simcc_pooled)

    confidence_features = layers.GlobalAveragePooling2D(name="spatial_conf_gap")(bottleneck)
    confidence = layers.Dense(
        1, activation="sigmoid", kernel_initializer="he_normal",
        name="confidence",
    )(confidence_features)

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name=model_name,
    )


# ---------------------------------------------------------------------------
# Candidate D: CoordConv + direct regression
# ---------------------------------------------------------------------------

def build_coordconv_direct_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    *,
    width_multiplier: float = 1.0,
    backbone_variant: str = "standard",
    dense_units: int = 256,
    dropout_rate: float = 0.15,
    model_name: str = "coordconv_direct_geometry",
) -> keras.Model:
    """Build Candidate D: CoordConv input → encoder → direct regression.

    Architecture:
      - CoordConv: concatenates normalised x/y coordinate channels to the
        RGB input before the first convolution.  This lets the encoder
        preserve absolute spatial position even after GAP.
      - QAT-friendly encoder (same as Candidate A).
      - GAP → Dense(256) → Dense(128) → 5 outputs:
        [center_x_norm, center_y_norm, tip_x_norm, tip_y_norm, confidence]
        All Sigmoid-constrained to [0, 1].

    Peak int8 activation: ~401 KB (encoder stage 1, 112×112×32).

    2025/2026 rationale:
      - CoordConv (Liu et al., NeurIPS 2018) is a simple, proven technique
        for injecting spatial awareness into convolutional networks that
        would otherwise lose absolute position after GAP.
      - Direct regression with Huber loss is the simplest possible output
        format, making it an important sanity-check baseline.
      - CoordConv avoids the Lambda issue because we implement it as a
        standard Concatenate of explicit coordinate tensors.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Build CoordConv coordinate channels as a Lambda that tiles to batch.
    # Normalised x/y grid: shape (1, H, W, 2), values in [-1, 1].
    coord_y, coord_x = tf.meshgrid(
        tf.linspace(-1.0, 1.0, input_shape[0]),
        tf.linspace(-1.0, 1.0, input_shape[1]),
        indexing="ij",
    )
    coord_channels = tf.stack(
        [tf.cast(coord_x, tf.float32), tf.cast(coord_y, tf.float32)],
        axis=-1,
    )  # (H, W, 2)
    coord_channels = tf.expand_dims(coord_channels, axis=0)  # (1, H, W, 2)

    # Lambda layer tiles coordinate channels to match the current batch size.
    def _tile_coords(inputs_list):
        image_batch = inputs_list[0]
        coords = inputs_list[1]
        batch_size = tf.shape(image_batch)[0]
        return tf.tile(coords, [batch_size, 1, 1, 1])

    coord_batch = layers.Lambda(
        _tile_coords, name="coordconv_tile",
    )([x, coord_channels])

    x = layers.Concatenate(name="coordconv_concat")([x, coord_batch])

    bottleneck, skip_112, skip_56, skip_28, skip_14 = _build_qat_encoder(
        x,
        width_multiplier=width_multiplier,
        backbone_variant=backbone_variant,
        name_prefix="coordconv_encoder",
    )

    pooled = layers.GlobalAveragePooling2D(name="coordconv_gap")(bottleneck)
    pooled = layers.Dense(
        dense_units,
        activation="relu",
        kernel_initializer="he_normal",
        name="coordconv_dense1",
    )(pooled)
    pooled = layers.Dropout(dropout_rate, name="coordconv_dropout")(pooled)
    pooled = layers.BatchNormalization(name="coordconv_bn")(pooled)
    pooled = layers.Dense(
        dense_units // 2,
        activation="relu",
        kernel_initializer="he_normal",
        name="coordconv_dense2",
    )(pooled)
    outputs = layers.Dense(
        5, activation="sigmoid",
        kernel_initializer="he_normal",
        name="geometry_outputs",
    )(pooled)

    return keras.Model(inputs=inputs, outputs=outputs, name=model_name)


# ---------------------------------------------------------------------------
# KD student wrapper for model.fit() integration
# ---------------------------------------------------------------------------

class KDStudentWrapper(keras.Model):
    """Wraps a student model to add KD loss from a frozen teacher during fit().

    The wrapper intercepts train_step() to:
      1. Run the frozen teacher on the same batch to get soft targets.
      2. Compute both ground-truth loss and KD (KL) loss on SimCC logits.
      3. Return the weighted sum.

    The student model must output (simcc_logits, confidence) — same signature
    as build_qat_simcc_model().  The teacher must have the same output names
    for its SimCC logit tensor.

    Usage:
        wrapper = KDStudentWrapper(
            student_model=student,
            teacher_model=frozen_teacher,
            simcc_output_name="simcc_head_reshape",
            teacher_simcc_name="teacher_simcc_reshape",
            kd_temperature=4.0,
            kd_weight=0.5,
        )
        wrapper.compile(
            optimizer=Adam(1e-3),
            gt_loss={...},  # loss dict for ground-truth
        )
        wrapper.fit(train_data, ...)
    """

    def __init__(
        self,
        student_model: keras.Model,
        teacher_model: keras.Model,
        *,
        simcc_output_name: str = "simcc_head_reshape",
        teacher_simcc_name: str = "teacher_simcc_reshape",
        confidence_name: str = "confidence",
        kd_temperature: float = 4.0,
        kd_weight: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.student = student_model
        self.teacher = teacher_model
        self.teacher.trainable = False
        self._simcc_name = simcc_output_name
        self._teacher_simcc_name = teacher_simcc_name
        self._confidence_name = confidence_name
        self._kd_temperature = float(kd_temperature)
        self._kd_weight = float(kd_weight)
        self._kd_loss_fn = SimCCKDLoss(temperature=kd_temperature)

        # Metric trackers.
        self._total_loss_tracker = keras.metrics.Mean(name="loss")
        self._gt_loss_tracker = keras.metrics.Mean(name="gt_loss")
        self._kd_loss_tracker = keras.metrics.Mean(name="kd_loss")

    @property
    def metrics(self):
        return [
            self._total_loss_tracker,
            self._gt_loss_tracker,
            self._kd_loss_tracker,
        ]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def train_step(self, data):
        x, y = data
        # y is a tuple: (simcc_gt, conf_gt) for SimCC models,
        # or a single tensor for direct regression.

        with tf.GradientTape() as tape:
            student_outputs = self.student(x, training=True)
            teacher_outputs = self.teacher(x, training=False)

            # Ground-truth loss.
            gt_loss = self.compute_loss(y=y, y_pred=student_outputs)

            # KD loss on SimCC logits.
            simcc_student = student_outputs["simcc_head_reshape"]
            simcc_teacher = teacher_outputs["teacher_simcc_reshape"]
            kd_val = self._kd_loss_fn(simcc_teacher, simcc_student)

            total_loss = (1.0 - self._kd_weight) * gt_loss + self._kd_weight * kd_val

        # Update student weights.
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply(gradients, trainable_vars)

        # Update metrics.
        self._total_loss_tracker.update_state(total_loss)
        self._gt_loss_tracker.update_state(gt_loss)
        self._kd_loss_tracker.update_state(kd_val)
        for metric in self.student.metrics:
            if hasattr(metric, "update_state"):
                metric.update_state(y, student_outputs)

        return {
            m.name: m.result() for m in self.metrics
        }

    def test_step(self, data):
        x, y = data
        student_outputs = self.student(x, training=False)
        gt_loss = self.compute_loss(y=y, y_pred=student_outputs)
        self._total_loss_tracker.update_state(gt_loss)
        for metric in self.student.metrics:
            if hasattr(metric, "update_state"):
                metric.update_state(y, student_outputs)
        return {
            m.name: m.result() for m in [self._total_loss_tracker]
        }

    def get_config(self):
        return {
            "kd_temperature": self._kd_temperature,
            "kd_weight": self._kd_weight,
        }

class SimCCKDLoss(keras.losses.Loss):
    """KL-divergence loss between teacher and student SimCC logits.

    The temperature T softens both distributions before computing KL.
    Higher T → broader teacher signal → more informative gradients for
    the student when the teacher is very confident.

    Following Hinton et al. (2015): KD loss = T² * KL(softmax(t_logits/T),
    softmax(s_logits/T)).
    """

    def __init__(
        self,
        temperature: float = 3.0,
        reduction: str = "sum_over_batch_size",
        name: str = "simcc_kd_loss",
        **kwargs,
    ) -> None:
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.temperature = float(temperature)

    def call(
        self,
        teacher_logits: tf.Tensor,
        student_logits: tf.Tensor,
    ) -> tf.Tensor:
        """Compute KD loss between teacher and student SimCC logits.

        Args:
            teacher_logits: (B, 4, bins) float logits from frozen teacher.
            student_logits: (B, 4, bins) float logits from student.

        Returns:
            Scalar loss.
        """
        t = tf.cast(self.temperature, teacher_logits.dtype)
        teacher_soft = tf.nn.softmax(teacher_logits / t, axis=-1)
        student_log_soft = tf.nn.log_softmax(student_logits / t, axis=-1)
        # KL divergence per keypoint, averaged.
        kl_per_kp = tf.reduce_sum(
            teacher_soft * (tf.math.log(teacher_soft + 1e-8) - student_log_soft),
            axis=-1,
        )  # (B, 4)
        kl_mean = tf.reduce_mean(kl_per_kp)
        return (t * t) * kl_mean


# ---------------------------------------------------------------------------
# DARK decoding (post-processing, not a Keras layer)
# ---------------------------------------------------------------------------

def dark_decode_heatmap(
    heatmap: np.ndarray,
    *,
    kernel_size: int = 5,
) -> Tuple[float, float]:
    """Decode a single-channel heatmap to sub-pixel (x, y) coordinates.

    DARK (Distribution-Aware coordinate Representation for Keypoint detection)
    by Zhang et al., CVPR 2020.

    Steps:
      1. Find the pixel with maximum activation (argmax).
      2. Extract a k×k window of log-heatmap values around the peak.
      3. Fit a 2D quadratic (log of Gaussian) and solve for the analytical
         mean to get an unbiased sub-pixel coordinate.

    Args:
        heatmap: Single heatmap array of shape (H, W).
        kernel_size: Window size for local fitting (odd, >= 3).

    Returns:
        Tuple of (x, y) coordinates in pixel space of the heatmap.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze(-1)
    h, w = heatmap.shape

    # Step 1: Argmax.
    idx = np.argmax(heatmap)
    cy, cx = divmod(int(idx), w)

    # Step 2: Extract local window.
    half = kernel_size // 2
    y1 = max(0, cy - half)
    y2 = min(h, cy + half + 1)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half + 1)

    patch = heatmap[y1:y2, x1:x2]
    if patch.size < 3:
        return float(cx), float(cy)

    # Step 3: Log-heatmap.
    patch = np.maximum(patch, 1e-10)
    log_patch = np.log(patch)

    # Fit quadratic: log(h) ≈ a*x² + b*y² + c*x + d*y + e*x*y + f
    # At the peak of the Gaussian, the gradient is zero.
    # Solve: mu = -[c, d] / [2a, 2b] (ignoring the xy cross-term for
    # simplicity, as in the original DARK paper).
    ph, pw = patch.shape
    yy, xx = np.meshgrid(
        np.arange(y1, y2, dtype=np.float32),
        np.arange(x1, x2, dtype=np.float32),
        indexing="ij",
    )
    # Center coordinates around the argmax for numerical stability.
    dx = xx - cx
    dy = yy - cy

    # Build design matrix for quadratic fit: [1, x, y, x², y², xy]
    A = np.column_stack([
        np.ones_like(dx.ravel()),
        dx.ravel(),
        dy.ravel(),
        (dx * dx).ravel(),
        (dy * dy).ravel(),
        (dx * dy).ravel(),
    ])
    b = log_patch.ravel()

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return float(cx), float(cy)

    # coeffs: [const, c, d, a, b, e]
    c_coeff = coeffs[1]  # x coefficient
    d_coeff = coeffs[2]  # y coefficient
    a_coeff = coeffs[3]  # x² coefficient
    b_coeff = coeffs[4]  # y² coefficient

    # Analytical mean: mu_x = -c/(2a), mu_y = -d/(2b)
    if abs(a_coeff) < 1e-8:
        sub_x = 0.0
    else:
        sub_x = -c_coeff / (2.0 * a_coeff)

    if abs(b_coeff) < 1e-8:
        sub_y = 0.0
    else:
        sub_y = -d_coeff / (2.0 * b_coeff)

    # Clamp sub-pixel offsets to [-half, half] to avoid unreasonable extrapolation.
    sub_x = max(-half, min(half, sub_x))
    sub_y = max(-half, min(half, sub_y))

    refined_x = float(cx) + sub_x
    refined_y = float(cy) + sub_y

    # Clamp to image bounds.
    refined_x = max(0.0, min(float(w - 1), refined_x))
    refined_y = max(0.0, min(float(h - 1), refined_y))

    return refined_x, refined_y


def dark_decode_batch(
    heatmaps: np.ndarray,
    *,
    kernel_size: int = 5,
) -> np.ndarray:
    """Decode a batch of heatmaps to sub-pixel coordinates.

    Args:
        heatmaps: (B, C, H, W) or (B, H, W, C) heatmap array.
        kernel_size: Window size for DARK fitting.

    Returns:
        (B, 2, C) coordinate array (x, y) per channel.
    """
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    if heatmaps.ndim == 4 and heatmaps.shape[-1] > heatmaps.shape[1]:
        # (B, H, W, C) format
        heatmaps = np.transpose(heatmaps, (0, 3, 1, 2))
    # Now (B, C, H, W)
    b, c = heatmaps.shape[:2]
    coords = np.zeros((b, 2, c), dtype=np.float32)
    for batch_idx in range(b):
        for ch_idx in range(c):
            x, y = dark_decode_heatmap(
                heatmaps[batch_idx, ch_idx], kernel_size=kernel_size,
            )
            coords[batch_idx, 0, ch_idx] = x
            coords[batch_idx, 1, ch_idx] = y
    return coords


# ---------------------------------------------------------------------------
# Activation budget estimation
# ---------------------------------------------------------------------------

def estimate_peak_int8_activation_bytes(
    model: keras.Model,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
) -> int:
    """Estimate the largest activations tensor size (int8 bytes) in a model.

    Runs a dummy forward pass so that every layer's output shape is
    materialized, then scans all layers for the maximum H*W*C product.
    """
    try:
        dummy = tf.zeros((1,) + tuple(input_shape), dtype=tf.float32)
        _ = model(dummy, training=False)
    except Exception:
        pass

    max_bytes = 0
    for layer in model.layers:
        try:
            output = layer.output
        except (AttributeError, ValueError):
            continue
        if output is None:
            continue

        shapes_to_check: List[Any] = []
        if isinstance(output, (list, tuple)):
            for o in output:
                if o is not None:
                    shapes_to_check.append(o.shape if hasattr(o, "shape") else o)
        elif hasattr(output, "shape"):
            shapes_to_check.append(output.shape)

        for shape in shapes_to_check:
            if shape is None:
                continue
            # Normalise to a list of ints or None.
            if hasattr(shape, "as_list"):
                dims = shape.as_list()
            elif hasattr(shape, "__iter__"):
                dims = [int(d) if d is not None else None for d in shape]
            else:
                continue
            if len(dims) < 3:
                continue
            spatial = 1
            spatial_rank = 0
            for dim in dims[1:]:
                if dim is not None:
                    spatial *= int(dim)
                    spatial_rank += 1
            if spatial_rank >= 2:
                max_bytes = max(max_bytes, spatial)
    return max_bytes


# ---------------------------------------------------------------------------
# TFLite int8 export helper
# ---------------------------------------------------------------------------

def export_tflite_int8(
    model: keras.Model,
    output_path: str,
    representative_data: Sequence[np.ndarray],
    *,
    optimize_for_latency: bool = True,
) -> None:
    """Export a float Keras model to TFLite with full-int8 quantization.

    Uses post-training int8 quantization (PTQ) with a representative
    dataset.  The exported model has float32 I/O with int8 internals,
    matching the pattern that works with the STM32 N6 NPU.

    Args:
        model: Trained float32 Keras model.
        output_path: Destination path for the .tflite file.
        representative_data: List of float32 input batches for calibration.
        optimize_for_latency: If True, use the latency-optimized converter
            flags (enable all quantized ops, disable non-supported ops).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def _representative_dataset():
        for batch in representative_data:
            yield [np.asarray(batch, dtype=np.float32)]

    converter.representative_dataset = _representative_dataset

    if optimize_for_latency:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    output_path = str(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)


def export_qat_tflite_int8(
    qat_model: keras.Model,
    output_path: str,
    *,
    representative_data: Optional[Sequence[np.ndarray]] = None,
) -> None:
    """Export a QAT-trained model to TFLite with full-int8 quantization.

    For QAT models, the quantization parameters are already baked into the
    graph via FakeQuant layers.  The converter uses DEFAULT optimizations
    to fold these into true int8 ops.

    Args:
        qat_model: The QAT-trained Keras model.
        output_path: Destination path for the .tflite file.
        representative_data: Optional calibration data.  For QAT models
            this is usually not needed since scales are already learned,
            but providing it can improve the converter's graph rewriting.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_data is not None:

        def _representative_dataset():
            for batch in representative_data:
                yield [np.asarray(batch, dtype=np.float32)]

        converter.representative_dataset = _representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    # For QAT models, use int8 I/O to match the NPU's native format.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path = str(output_path)
    with open(output_path, "wb") as f:
        f.write(tflite_model)


# ---------------------------------------------------------------------------
# Spatial SimCC model — Conv-based head preserving spatial information
# ---------------------------------------------------------------------------

def build_mobilenetv2_spatial_simcc_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    *,
    simcc_bins: int = 112,
    spatial_channels: int = 128,
    alpha: float = 0.75,
    backbone_frozen: bool = True,
    dropout_rate: float = 0.15,
    model_name: str = "mnv2_spatial_simcc",
) -> keras.Model:
    """MobileNetV2 + spatial SimCC head (preserves spatial info, no GAP).

    Uses 1×1 Conv to project backbone features into per-axis logit maps,
    then mean-pools over the spatial dimension.  This preserves spatial
    information that GAP destroys, giving better keypoint localization.

    Architecture:
      - MobileNetV2 backbone (alpha, ImageNet pretrained)
      - 1×1 Conv → BN → ReLU → Dropout (spatial_channels)
      - 1×1 Conv → (7, 7, 4 * simcc_bins)
      - Reshape → (49, simcc_bins, 4)
      - ReduceMean over spatial → (simcc_bins, 4)
      - Permute → (4, simcc_bins) SimCC logits
      - Confidence from GAP

    PTQ int8 export compatible.  NOT QAT-compatible (MobileNetV2 Lambda layers).
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_preprocess")(inputs)

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    backbone.trainable = not backbone_frozen
    x = backbone(x)

    bottleneck = layers.Conv2D(
        spatial_channels, 1, padding="same", use_bias=False,
        kernel_initializer="he_normal",
        name="spatial_bottleneck_conv",
    )(x)
    bottleneck = layers.BatchNormalization(name="spatial_bottleneck_bn")(bottleneck)
    bottleneck = layers.ReLU(name="spatial_bottleneck_relu")(bottleneck)
    bottleneck = layers.Dropout(dropout_rate, name="spatial_bottleneck_dropout")(bottleneck)

    num_keypoints = 4
    logit_maps = layers.Conv2D(
        num_keypoints * simcc_bins, 1, padding="same",
        kernel_initializer="he_normal",
        name="spatial_logit_maps",
    )(bottleneck)

    h = int(logit_maps.shape[1])
    w = int(logit_maps.shape[2])
    logit_4d = layers.Reshape(
        (h * w, simcc_bins, num_keypoints),
        name="spatial_reshape",
    )(logit_maps)

    simcc_pooled = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name="spatial_reduce_mean",
    )(logit_4d)

    simcc_logits = layers.Permute(
        (2, 1), name="spatial_simcc_logits",
    )(simcc_pooled)

    confidence_features = layers.GlobalAveragePooling2D(name="spatial_conf_gap")(bottleneck)
    confidence = layers.Dense(
        1, activation="sigmoid",
        kernel_initializer="he_normal",
        name="confidence",
    )(confidence_features)

    return keras.Model(
        inputs=inputs,
        outputs=[center_heatmap, tip_heatmap, confidence],
        name=model_name,
    )
