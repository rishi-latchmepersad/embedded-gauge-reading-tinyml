# Backbone builders for gauge-reading models.
from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
import keras


def _norm(x: keras.KerasTensor) -> keras.KerasTensor:
    """Apply convolution-friendly normalization for image features."""
    return keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-3)(x)


def _conv_norm_swish(
    x: keras.KerasTensor,
    filters: int,
    *,
    kernel_size: int = 3,
    strides: int = 1,
) -> keras.KerasTensor:
    """Apply Conv2D + normalization + swish."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)
    return x


def _residual_separable_block(
    x: keras.KerasTensor,
    filters: int,
    *,
    dropout_rate: float = 0.0,
) -> keras.KerasTensor:
    """Apply a lightweight residual block with separable convolutions."""
    shortcut: keras.KerasTensor = x

    x = keras.layers.SeparableConv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
    )(x)
    x = _norm(x)
    x = keras.layers.Activation("swish")(x)

    x = keras.layers.SeparableConv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
    )(x)
    x = _norm(x)

    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
        )(shortcut)
        shortcut = _norm(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation("swish")(x)

    if dropout_rate > 0.0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def _build_feature_backbone(
    image_height: int, image_width: int
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """Build the shared CNN backbone and return (inputs, pooled_features)."""
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    x = _conv_norm_swish(inputs, 32, strides=2)
    x = _residual_separable_block(x, 32, dropout_rate=0.02)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 64, dropout_rate=0.04)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 96, dropout_rate=0.06)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 128, dropout_rate=0.08)
    x = keras.layers.GlobalAveragePooling2D()(x)
    return inputs, x


def _build_compact_geometry_backbone(
    image_height: int, image_width: int
) -> tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
    """Build compact CNN features for both spatial and pooled geometry heads."""
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    x = _conv_norm_swish(inputs, 32, strides=2)
    x = _residual_separable_block(x, 32, dropout_rate=0.02)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 64, dropout_rate=0.04)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    x = _residual_separable_block(x, 96, dropout_rate=0.06)
    x = keras.layers.MaxPool2D(pool_size=2)(x)

    spatial_features = _residual_separable_block(x, 128, dropout_rate=0.08)
    pooled_features = keras.layers.GlobalAveragePooling2D()(spatial_features)
    return inputs, spatial_features, pooled_features


def _build_mobilenetv2_backbone(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool,
    backbone_trainable: bool,
    alpha: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor, keras.Model]:
    """Build a MobileNetV2 feature backbone and return the pooled feature map."""
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

    x = base_model(x, training=backbone_trainable)
    return inputs, x, base_model


def _build_mobilenetv2_dual_resolution_backbone(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool,
    backbone_trainable: bool,
    alpha: float,
    crop_ratio: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor, keras.Model]:
    """Build a shared-weight full-frame and center-crop MobileNetV2 backbone.

    The full branch preserves global context and dial framing, while the
    center-crop branch zooms into the pointer and inner Celsius arc. Sharing
    the same MobileNetV2 weights keeps the model compact enough for the 4 GB
    GPU while still giving the head two complementary views of the gauge.
    """
    if not (0.1 < crop_ratio <= 1.0):
        raise ValueError("crop_ratio must be in the range (0.1, 1.0].")

    crop_height = max(32, int(round(image_height * crop_ratio)))
    crop_width = max(32, int(round(image_width * crop_ratio)))

    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    # The full-frame branch keeps the entire dial, background cues, and label
    # context that help disambiguate the reading on noisy preview captures.
    full_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="dualres_full_preprocess",
    )(inputs)

    # The center-crop branch zooms in on the dial face and pointer, which is
    # where the hard-case errors tend to hide in our sample images.
    crop_branch = CenterCropResize(
        crop_height,
        crop_width,
        image_height,
        image_width,
        interpolation="bilinear",
        name="dualres_center_crop_resize",
    )(inputs)
    crop_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="dualres_crop_preprocess",
    )(crop_branch)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    base_model.trainable = backbone_trainable

    full_maps = base_model(full_branch, training=backbone_trainable)
    crop_maps = base_model(crop_branch, training=backbone_trainable)

    # Refine each view before pooling so the head sees a cleaner needle-focused
    # representation rather than raw backbone activations.
    full_maps = CBAMBlock(reduction_ratio=8, name="dualres_full_cbam")(full_maps)
    full_maps = CoordinateAttention(
        reduction_ratio=8,
        name="dualres_full_coordattn",
    )(full_maps)
    crop_maps = CBAMBlock(reduction_ratio=8, name="dualres_crop_cbam")(crop_maps)
    crop_maps = CoordinateAttention(
        reduction_ratio=8,
        name="dualres_crop_coordattn",
    )(crop_maps)

    full_gap = keras.layers.GlobalAveragePooling2D(name="dualres_full_gap")(full_maps)
    full_gmp = keras.layers.GlobalMaxPooling2D(name="dualres_full_gmp")(full_maps)
    crop_gap = keras.layers.GlobalAveragePooling2D(name="dualres_crop_gap")(crop_maps)
    crop_gmp = keras.layers.GlobalMaxPooling2D(name="dualres_crop_gmp")(crop_maps)

    # Compare the global view against the zoomed view so the head can learn
    # when the needle geometry disagrees with the surrounding dial context.
    gap_delta = keras.layers.Subtract(name="dualres_gap_delta")([full_gap, crop_gap])
    gmp_delta = keras.layers.Subtract(name="dualres_gmp_delta")([full_gmp, crop_gmp])

    features = keras.layers.Concatenate(name="dualres_features")(
        [full_gap, full_gmp, crop_gap, crop_gmp, gap_delta, gmp_delta]
    )
    features = keras.layers.LayerNormalization(name="dualres_features_norm")(features)
    return inputs, features, base_model



def _build_mobilenetv2_polar_dualview_backbone(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool,
    backbone_trainable: bool,
    alpha: float,
) -> tuple[dict[str, keras.KerasTensor], keras.KerasTensor, keras.Model]:
    """Build a shared-weight full-frame and polar-unwrapped MobileNetV2 backbone.

    The full view preserves the original framing, while the polar view turns the
    circular gauge face into a rectangular angle-versus-radius image. That makes
    the needle position easier for a CNN to model because the circular geometry
    is flattened into a more linear representation.
    """
    full_input = keras.Input(shape=(image_height, image_width, 3), name="full_image")
    polar_input = keras.Input(shape=(image_height, image_width, 3), name="polar_image")

    # Keep the preprocessing identical across both branches so the shared
    # MobileNetV2 trunk sees consistent input scaling.
    full_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="polar_full_preprocess",
    )(full_input)
    polar_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="polar_view_preprocess",
    )(polar_input)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    base_model.trainable = backbone_trainable

    full_maps = base_model(full_branch, training=backbone_trainable)
    polar_maps = base_model(polar_branch, training=backbone_trainable)

    full_gap = keras.layers.GlobalAveragePooling2D(name="polar_full_gap")(full_maps)
    full_gmp = keras.layers.GlobalMaxPooling2D(name="polar_full_gmp")(full_maps)
    polar_gap = keras.layers.GlobalAveragePooling2D(name="polar_view_gap")(polar_maps)
    polar_gmp = keras.layers.GlobalMaxPooling2D(name="polar_view_gmp")(polar_maps)

    # Encourage the head to learn whether the polar view and the raw view agree.
    gap_delta = keras.layers.Subtract(name="polar_gap_delta")([full_gap, polar_gap])
    gmp_delta = keras.layers.Subtract(name="polar_gmp_delta")([full_gmp, polar_gmp])
    gap_mean = keras.layers.Average(name="polar_gap_mean")([full_gap, polar_gap])
    gmp_mean = keras.layers.Average(name="polar_gmp_mean")([full_gmp, polar_gmp])
    gap_abs_delta = keras.layers.Lambda(
        lambda tensors: tf.abs(tensors[0] - tensors[1]),
        name="polar_gap_abs_delta",
    )([full_gap, polar_gap])
    gmp_abs_delta = keras.layers.Lambda(
        lambda tensors: tf.abs(tensors[0] - tensors[1]),
        name="polar_gmp_abs_delta",
    )([full_gmp, polar_gmp])

    features = keras.layers.Concatenate(name="polar_dualview_features")(
        [
            full_gap,
            full_gmp,
            polar_gap,
            polar_gmp,
            gap_delta,
            gmp_delta,
            gap_mean,
            gmp_mean,
            gap_abs_delta,
            gmp_abs_delta,
        ]
    )
    features = keras.layers.LayerNormalization(name="polar_dualview_features_norm")(
        features
    )
    return {"full_image": full_input, "polar_image": polar_input}, features, base_model


def _build_mobilenetv2_polar_backbone(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool,
    backbone_trainable: bool,
    alpha: float,
) -> tuple[keras.KerasTensor, keras.KerasTensor, keras.Model]:
    """Build a single-input polar-unwrapped MobileNetV2 backbone.

    The polar projection keeps the gauge geometry but flattens the circular dial
    into an angle-versus-radius grid. That is simpler than the dual-view branch
    and avoids the extra data plumbing that has been slowing our experiments.
    """
    polar_input = keras.Input(shape=(image_height, image_width, 3), name="polar_image")

    # MobileNetV2 expects the usual [-1, 1] preprocessing.
    polar_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="polar_preprocess",
    )(polar_input)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_height, image_width, 3),
        alpha=alpha,
    )
    base_model.trainable = backbone_trainable

    polar_maps = base_model(polar_branch, training=backbone_trainable)
    polar_gap = keras.layers.GlobalAveragePooling2D(name="polar_gap")(polar_maps)
    polar_gmp = keras.layers.GlobalMaxPooling2D(name="polar_gmp")(polar_maps)

    # Combine average and max pooled views so the head can keep both context and
    # sharp pointer/tick responses.
    features = keras.layers.Concatenate(name="polar_features")([polar_gap, polar_gmp])
    features = keras.layers.LayerNormalization(name="polar_features_norm")(features)
    return polar_input, features, base_model


def _mobilenetv2_model_name(
    *,
    regression_kind: str,
    alpha: float,
    head_units: int,
) -> str:
    """Generate a stable model name for a MobileNetV2 variant."""
    if math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-6) and head_units == 128:
        return f"mobilenetv2_{regression_kind}_regressor"

    alpha_tag: int = int(round(alpha * 100.0))
    return f"mobilenetv2_{regression_kind}_regressor_a{alpha_tag:03d}_h{head_units:03d}"

