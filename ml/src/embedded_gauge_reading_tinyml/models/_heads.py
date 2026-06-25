# Backbone builders for gauge-reading models.
from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
import keras

from ._backbones import _norm


def _build_interval_expectation_head(
    logits: keras.KerasTensor,
    *,
    value_min: float,
    value_max: float,
    bin_width: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Convert interval logits into a scalar expected temperature value."""
    if bin_width <= 0.0:
        raise ValueError("bin_width must be > 0.")
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    span = value_max - value_min
    num_bins = int(math.ceil(span / bin_width))
    if num_bins < 2:
        raise ValueError("Need at least two bins for interval regression.")

    # Place one center per fixed-width interval, anchored at the calibrated range.
    bin_centers = [value_min + (index + 0.5) * bin_width for index in range(num_bins)]
    centers = keras.layers.Dense(
        1,
        use_bias=False,
        trainable=False,
        name="interval_value_projection",
        kernel_initializer=keras.initializers.Constant(
            [[center] for center in bin_centers]
        ),
    )
    probs = keras.layers.Softmax(name="interval_probs")(logits)
    value = centers(probs)
    return value, probs


def _build_ordinal_expectation_head(
    logits: keras.KerasTensor,
    *,
    value_min: float,
    value_max: float,
    threshold_step: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Convert ordinal threshold logits into a scalar expected value."""
    if threshold_step <= 0.0:
        raise ValueError("threshold_step must be > 0.")
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    span = value_max - value_min
    num_thresholds = int(math.ceil(span / threshold_step))
    if num_thresholds < 2:
        raise ValueError("Need at least two thresholds for ordinal regression.")

    probs = keras.layers.Activation("sigmoid", name="ordinal_probs")(logits)
    threshold_count = keras.layers.Dense(
        1,
        use_bias=False,
        trainable=False,
        name="ordinal_threshold_count",
        kernel_initializer=keras.initializers.Constant([[1.0]] * num_thresholds),
    )(probs)
    value = keras.layers.Rescaling(
        threshold_step,
        offset=value_min,
        name="gauge_value",
    )(threshold_count)
    return value, probs


def _build_sweep_fraction_head(
    logits: keras.KerasTensor,
    *,
    value_min: float,
    value_max: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Convert a sweep-fraction logit into both fraction and calibrated value."""
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    fraction = keras.layers.Activation("sigmoid", name="sweep_fraction")(logits)
    span = value_max - value_min
    value = keras.layers.Rescaling(
        span,
        offset=value_min,
        name="gauge_value",
    )(fraction)
    return value, fraction


def _build_keypoint_heatmap_head(
    features: keras.KerasTensor,
    *,
    heatmap_size: int,
    num_keypoints: int = 2,
) -> keras.KerasTensor:
    """Build a small decoder that predicts keypoint heatmaps."""
    if heatmap_size < 4:
        raise ValueError("heatmap_size must be >= 4.")
    if num_keypoints < 1:
        raise ValueError("num_keypoints must be >= 1.")

    x = keras.layers.Conv2D(
        256,
        3,
        padding="same",
        use_bias=False,
    )(features)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)

    x = keras.layers.Conv2D(
        128,
        3,
        padding="same",
        use_bias=False,
    )(x)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)

    x = keras.layers.Conv2D(
        64,
        3,
        padding="same",
        use_bias=False,
    )(x)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Resizing(
        heatmap_size,
        heatmap_size,
        interpolation="bilinear",
        name="keypoint_heatmap_resize",
    )(x)
    heatmaps = keras.layers.Conv2D(
        num_keypoints,
        1,
        activation="sigmoid",
        name="keypoint_heatmaps",
    )(x)
    return heatmaps


def _build_pointer_mask_head(
    features: keras.KerasTensor,
    *,
    mask_size: int,
) -> keras.KerasTensor:
    """Decode a dense pointer-mask prediction from shared spatial features."""
    if mask_size < 4:
        raise ValueError("mask_size must be >= 4.")

    x = keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(features)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)

    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Resizing(
        mask_size,
        mask_size,
        interpolation="bilinear",
        name="pointer_mask_resize",
    )(x)
    pointer_mask = keras.layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="pointer_mask",
    )(x)
    return pointer_mask


def _build_unsharp_mask_branch(
    inputs: keras.KerasTensor,
    *,
    name_prefix: str,
) -> keras.KerasTensor:
    """Create a blur-aware branch by subtracting a fixed Gaussian blur."""
    blur_kernel = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    blur_kernel /= np.sum(blur_kernel)
    blur_kernel = blur_kernel[:, :, np.newaxis, np.newaxis]
    blur_kernel = np.repeat(blur_kernel, repeats=3, axis=2)

    blurred = keras.layers.DepthwiseConv2D(
        3,
        padding="same",
        use_bias=False,
        trainable=False,
        depthwise_initializer=keras.initializers.Constant(blur_kernel),
        name=f"{name_prefix}_fixed_blur",
    )(inputs)
    detail = keras.layers.Subtract(name=f"{name_prefix}_detail")([inputs, blurred])
    # A plain residual sum keeps the enhancement branch serializable while
    # still emphasizing edges and thin pointer structure.
    return keras.layers.Add(name=f"{name_prefix}_unsharp_mask")([inputs, detail])


def _build_mobilenetv2_multi_scale_backbone(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool,
    backbone_trainable: bool,
    alpha: float,
) -> tuple[keras.KerasTensor, list[keras.KerasTensor], keras.Model]:
    """Build a MobileNetV2 backbone that exposes multi-scale feature maps.

    Returns early, mid, and late features so the head can fuse information
    at multiple resolutions — critical for needle detection where both
    global dial context and local pointer detail matter.
    """
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    # The pipeline emits [0, 1] floats; MobileNetV2 expects [-1, 1].
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(
        inputs
    )

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    base_model.trainable = backbone_trainable

    # Build a multi-output feature extractor from the base model.
    # We extract intermediate activations from specific MobileNetV2 blocks:
    #   block_1_project_BN → early features (56x56)
    #   block_3_project_BN → mid features (28x28)
    #   block_6_project_BN → late features (14x14)
    #   final output       → final features (7x7)
    early_layer = base_model.get_layer("block_1_project_BN")
    mid_layer = base_model.get_layer("block_3_project_BN")
    late_layer = base_model.get_layer("block_6_project_BN")

    feature_model = keras.Model(
        inputs=base_model.inputs,
        outputs=[
            early_layer.output,
            mid_layer.output,
            late_layer.output,
            base_model.output,
        ],
        name="mobilenetv2_multiscale",
    )
    feature_model.trainable = backbone_trainable

    early, mid, late, final = feature_model(x, training=backbone_trainable)
    return inputs, [early, mid, late, final], base_model


def _cbam_refine(
    features: list[keras.KerasTensor],
    base_channels: int = 64,
) -> keras.KerasTensor:
    """Fuse multi-scale features with CBAM attention and upsampling.

    Each scale gets CBAM-refined, upsampled to the largest spatial size,
    then concatenated for the regression head.
    """
    refined: list[keras.KerasTensor] = []

    for i, feat in enumerate(features):
        # Project to consistent channel count
        channels = max(base_channels // (2 ** max(i - 1, 0)), 16)
        x = keras.layers.Conv2D(
            channels, 1, padding="same", use_bias=False, name=f"ms_proj_{i}"
        )(feat)
        x = keras.layers.BatchNormalization(
            momentum=0.9, epsilon=1e-3, name=f"ms_proj_bn_{i}"
        )(x)
        x = keras.layers.Activation("swish")(x)
        # CBAM attention
        x = CBAMBlock(reduction_ratio=8, name=f"ms_cbam_{i}")(x)
        # Upsample to match the spatial size of the earliest (largest) feature.
        # We use Resizing with the static spatial dims from the first feature.
        if i > 0:
            target_h = int(features[0].shape[1])
            target_w = int(features[0].shape[2])
            x = keras.layers.Resizing(
                target_h,
                target_w,
                interpolation="bilinear",
                name=f"ms_resize_{i}",
            )(x)
        refined.append(x)

    # Concatenate all refined scales
    fused = keras.layers.Concatenate(name="ms_fused")(refined)
    # Final refinement conv
    fused = keras.layers.Conv2D(
        base_channels * 2, 3, padding="same", use_bias=False, name="ms_fusion_conv"
    )(fused)
    fused = keras.layers.BatchNormalization(
        momentum=0.9, epsilon=1e-3, name="ms_fusion_bn"
    )(fused)
    fused = keras.layers.Activation("swish")(fused)
    return fused
