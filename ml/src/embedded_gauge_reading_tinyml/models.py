"""Model builders for gauge-reading networks."""

from __future__ import annotations

import math

import keras
import numpy as np
import tensorflow as tf


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class CBAMBlock(keras.layers.Layer):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Applies channel attention followed by spatial attention to refine feature maps.
    This helps the model focus on the needle region and suppress background clutter.
    """

    def __init__(
        self,
        reduction_ratio: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape: tf.TensorShape) -> None:
        channels = int(input_shape[-1])
        self._channel_avg = keras.layers.GlobalAveragePooling2D(
            name=f"{self.name}_ch_avg"
        )
        self._channel_max = keras.layers.GlobalMaxPooling2D(name=f"{self.name}_ch_max")
        reduced = max(channels // self.reduction_ratio, 4)
        self._mlp_shared = keras.Sequential(
            [
                keras.layers.Dense(reduced, activation="swish", use_bias=False),
                keras.layers.Dense(channels, use_bias=False),
            ],
            name=f"{self.name}_ch_mlp",
        )
        self._spatial_conv = keras.layers.Conv2D(
            1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name=f"{self.name}_spatial_conv",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Channel attention: squeeze + excite via shared MLP
        avg_pool = self._channel_avg(inputs)  # (B, C)
        max_pool = self._channel_max(inputs)  # (B, C)
        avg_out = self._mlp_shared(avg_pool)  # (B, C)
        max_out = self._mlp_shared(max_pool)  # (B, C)
        channel_attn = keras.activations.sigmoid(avg_out + max_out)  # (B, C)
        channel_attn = channel_attn[:, tf.newaxis, tf.newaxis, :]  # (B, 1, 1, C)
        x = inputs * channel_attn

        # Spatial attention: concatenate avg/max pooled channels, then conv
        spatial_avg = tf.reduce_mean(x, axis=-1, keepdims=True)  # (B, H, W, 1)
        spatial_max = tf.reduce_max(x, axis=-1, keepdims=True)  # (B, H, W, 1)
        spatial_concat = tf.concat([spatial_avg, spatial_max], axis=-1)  # (B, H, W, 2)
        spatial_attn = self._spatial_conv(spatial_concat)  # (B, H, W, 1)
        x = x * spatial_attn
        return x

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class CoordinateAttention(keras.layers.Layer):
    """Coordinate Attention (Hou et al., CVPR 2021).

    Encodes position information via 1D horizontal and vertical pooling,
    then applies attention to the feature map. Better than CBAM for
    position-sensitive tasks like needle angle regression.
    """

    def __init__(
        self,
        reduction_ratio: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape: tf.TensorShape) -> None:
        _, h, w, c = input_shape
        reduced = max(c // self.reduction_ratio, 4)
        self._conv1 = keras.layers.Conv2D(
            reduced,
            kernel_size=1,
            use_bias=False,
            name=f"{self.name}_conv1",
        )
        self._bn1 = keras.layers.BatchNormalization(
            momentum=0.9, epsilon=1e-3, name=f"{self.name}_bn1"
        )
        self._conv_h = keras.layers.Conv2D(
            c, kernel_size=1, use_bias=False, name=f"{self.name}_conv_h"
        )
        self._conv_w = keras.layers.Conv2D(
            c, kernel_size=1, use_bias=False, name=f"{self.name}_conv_w"
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        _, h, w, c = tf.unstack(tf.shape(inputs))
        # Horizontal (x-direction) pooling: (B, 1, W, C)
        x_h = tf.reduce_mean(inputs, axis=1, keepdims=True)
        # Vertical (y-direction) pooling: (B, H, 1, C)
        x_w = tf.reduce_mean(inputs, axis=2, keepdims=True)
        # Permute for concat: (B, 1, W, C) -> (B, H, 1, C) via transpose trick
        x_w_perm = tf.transpose(x_w, [0, 2, 1, 3])  # (B, 1, H, C)
        # Concat along spatial dim: (B, 1, W+H, C)
        concat = tf.concat([x_h, x_w_perm], axis=2)
        # 1x1 conv + BN + swish
        fused = self._conv1(concat)
        fused = self._bn1(fused)
        fused = keras.activations.swish(fused)
        # Split back
        split_h, split_w = tf.split(fused, [w, h], axis=2)
        # (B, 1, W, C) -> (B, H, W, C) via broadcast
        attn_h = keras.activations.sigmoid(self._conv_h(split_h))
        attn_w = keras.activations.sigmoid(
            self._conv_w(tf.transpose(split_w, [0, 2, 1, 3]))
        )
        return inputs * attn_h * attn_w

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class CenterCropResize(keras.layers.Layer):
    """Center-crop a tensor and resize it back to a fixed model input size.

    Keras' preprocessing `CenterCrop` layer requires a fully static spatial
    shape at call time. Some of our dual-resolution traces surface a symbolic
    spatial shape, so we do the same operation with TensorFlow image ops.
    """

    def __init__(
        self,
        crop_height: int,
        crop_width: int,
        target_height: int,
        target_width: int,
        *,
        interpolation: str = "bilinear",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if crop_height < 1 or crop_width < 1:
            raise ValueError("crop_height and crop_width must be >= 1.")
        if target_height < 1 or target_width < 1:
            raise ValueError("target_height and target_width must be >= 1.")
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width)
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.interpolation = str(interpolation)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply a centered crop and then resize to the model input size."""
        cropped = tf.image.resize_with_crop_or_pad(
            inputs,
            self.crop_height,
            self.crop_width,
        )
        return tf.image.resize(
            cropped,
            [self.target_height, self.target_width],
            method=self.interpolation,
        )

    def get_config(self) -> dict[str, object]:
        """Serialize the crop/resize parameters with the layer."""
        config = super().get_config()
        config.update(
            {
                "crop_height": self.crop_height,
                "crop_width": self.crop_width,
                "target_height": self.target_height,
                "target_width": self.target_width,
                "interpolation": self.interpolation,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class OrderedCornerBox(keras.layers.Layer):
    """Convert two unordered corner pairs into a stable xyxy crop box.

    The direct crop-box head predicts two x coordinates and two y coordinates.
    Sorting each pair keeps the target ordered without forcing the model to
    learn that constraint implicitly.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Sort the x and y corner pairs into `(x0, y0, x1, y1)` order."""
        corners = tf.cast(inputs, tf.float32)
        if corners.shape.rank is not None and corners.shape[-1] != 4:
            raise ValueError("OrderedCornerBox expects a 4D last dimension.")

        x_pair = tf.sort(corners[..., :2], axis=-1)
        y_pair = tf.sort(corners[..., 2:], axis=-1)
        return tf.concat(
            [x_pair[..., :1], y_pair[..., :1], x_pair[..., 1:], y_pair[..., 1:]],
            axis=-1,
        )

    def get_config(self) -> dict[str, object]:
        """Serialize the layer configuration for SavedModel/Keras export."""
        return super().get_config()


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class CornerKeypointsToBox(keras.layers.Layer):
    """Convert four corner keypoints into a normalized axis-aligned crop box."""

    def __init__(self, *, heatmap_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if heatmap_size < 4:
            raise ValueError("heatmap_size must be >= 4.")
        self.heatmap_size = int(heatmap_size)
        self._denominator = float(max(self.heatmap_size - 1, 1))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Reduce four corner coordinates into a clipped xyxy box."""
        keypoints = tf.cast(inputs, tf.float32)
        if keypoints.shape.rank is not None and keypoints.shape[-2:] != (4, 2):
            raise ValueError("CornerKeypointsToBox expects shape (batch, 4, 2).")

        normalized = tf.clip_by_value(keypoints / self._denominator, 0.0, 1.0)
        x_coords = normalized[..., 0]
        y_coords = normalized[..., 1]
        x_min = tf.reduce_min(x_coords, axis=-1, keepdims=True)
        y_min = tf.reduce_min(y_coords, axis=-1, keepdims=True)
        x_max = tf.reduce_max(x_coords, axis=-1, keepdims=True)
        y_max = tf.reduce_max(y_coords, axis=-1, keepdims=True)
        return keras.ops.concatenate([x_min, y_min, x_max, y_max], axis=-1)

    def get_config(self) -> dict[str, object]:
        """Serialize the heatmap sizing used to normalize the keypoints."""
        config = super().get_config()
        config.update({"heatmap_size": self.heatmap_size})
        return config


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


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class SpatialSoftArgmax2D(keras.layers.Layer):
    """Convert per-keypoint heatmaps into soft coordinates."""

    def __init__(
        self,
        *,
        heatmap_size: int,
        num_keypoints: int = 2,
        temperature: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if heatmap_size < 4:
            raise ValueError("heatmap_size must be >= 4.")
        if num_keypoints < 1:
            raise ValueError("num_keypoints must be >= 1.")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.temperature = temperature
        self._grid_x: tf.Tensor | None = None
        self._grid_y: tf.Tensor | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Precompute the heatmap coordinate grid used by the soft argmax."""
        if len(input_shape) != 4:
            raise ValueError("Heatmap tensor must be rank 4.")
        channels = input_shape[-1]
        if channels is not None and int(channels) != self.num_keypoints:
            raise ValueError("Heatmap channel count must match num_keypoints.")

        coords = tf.range(self.heatmap_size, dtype=tf.float32)
        grid_y, grid_x = tf.meshgrid(coords, coords, indexing="ij")
        self._grid_x = tf.reshape(grid_x, [1, -1, 1])
        self._grid_y = tf.reshape(grid_y, [1, -1, 1])
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Turn sigmoid heatmaps into coordinate expectations."""
        if self._grid_x is None or self._grid_y is None:
            raise RuntimeError("SpatialSoftArgmax2D was not built correctly.")

        heatmaps = tf.cast(inputs, tf.float32)
        batch_size = tf.shape(heatmaps)[0]
        flattened = tf.reshape(
            heatmaps,
            [batch_size, self.heatmap_size * self.heatmap_size, self.num_keypoints],
        )
        weights = tf.nn.softmax(flattened * self.temperature, axis=1)
        x_coords = tf.reduce_sum(weights * self._grid_x, axis=1)
        y_coords = tf.reduce_sum(weights * self._grid_y, axis=1)
        return tf.stack([x_coords, y_coords], axis=-1)

    def get_config(self) -> dict[str, object]:
        """Serialize the layer config for saved models."""
        config = super().get_config()
        config.update(
            {
                "heatmap_size": self.heatmap_size,
                "num_keypoints": self.num_keypoints,
                "temperature": self.temperature,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class GaugeValueFromKeypoints(keras.layers.Layer):
    """Convert detected center/tip keypoints into a calibrated gauge value."""

    def __init__(
        self,
        *,
        value_min: float,
        value_max: float,
        min_angle_rad: float,
        sweep_rad: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if value_max <= value_min:
            raise ValueError("value_max must be > value_min.")
        if sweep_rad <= 0.0:
            raise ValueError("sweep_rad must be > 0.")
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.min_angle_rad = float(min_angle_rad)
        self.sweep_rad = float(sweep_rad)
        self._two_pi = float(2.0 * math.pi)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Map a center/tip coordinate pair into the calibrated gauge value."""
        keypoints = tf.cast(inputs, tf.float32)
        center = keypoints[:, 0, :]
        tip = keypoints[:, 1, :]
        dx = tip[:, 0] - center[:, 0]
        dy = tip[:, 1] - center[:, 1]
        raw_angle = tf.atan2(dy, dx)
        shifted = tf.math.floormod(raw_angle - self.min_angle_rad, self._two_pi)
        fraction = tf.clip_by_value(shifted / self.sweep_rad, 0.0, 1.0)
        span = self.value_max - self.value_min
        value = self.value_min + fraction * span
        return tf.expand_dims(value, axis=-1)

    def get_config(self) -> dict[str, object]:
        """Serialize the layer config for saved models."""
        config = super().get_config()
        config.update(
            {
                "value_min": self.value_min,
                "value_max": self.value_max,
                "min_angle_rad": self.min_angle_rad,
                "sweep_rad": self.sweep_rad,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class GaugeValueFromRelationKeypoints(keras.layers.Layer):
    """Infer the gauge value from a learned relation over four geometry landmarks.

    The layer keeps the familiar gauge range calibration, but it exposes the
    network to all four predicted landmarks at once: center, tip, sweep-min,
    and sweep-max. That gives the model a direct place to learn the pointer-to-
    scale relation described in the recent gauge-reading literature.
    """

    def __init__(
        self,
        *,
        value_min: float,
        value_max: float,
        head_units: int = 64,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if value_max <= value_min:
            raise ValueError("value_max must be > value_min.")
        if head_units < 8:
            raise ValueError("head_units must be >= 8.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.head_units = int(head_units)
        self.dropout = float(dropout)
        self._scale = float(self.value_max - self.value_min)
        self._dense_1: keras.layers.Dense | None = None
        self._dense_2: keras.layers.Dense | None = None
        self._dropout: keras.layers.Dropout | None = None
        self._raw_out: keras.layers.Dense | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create the small relation MLP the first time the layer is built."""
        if len(input_shape) != 3 or int(input_shape[-1]) != 2:
            raise ValueError("Relation keypoints must have shape (batch, 4, 2).")
        self._dense_1 = keras.layers.Dense(
            self.head_units,
            activation="swish",
            name=f"{self.name}_dense_1",
        )
        self._dense_2 = keras.layers.Dense(
            max(self.head_units // 2, 8),
            activation="swish",
            name=f"{self.name}_dense_2",
        )
        self._dropout = keras.layers.Dropout(self.dropout)
        self._raw_out = keras.layers.Dense(1, name=f"{self.name}_raw")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Map four landmark coordinates to a normalized scalar value."""
        if self._dense_1 is None or self._dense_2 is None or self._raw_out is None:
            raise RuntimeError(
                "GaugeValueFromRelationKeypoints was not built correctly."
            )

        keypoints = tf.cast(inputs, tf.float32)
        center = keypoints[:, 0, :]
        tip = keypoints[:, 1, :]
        sweep_min = keypoints[:, 2, :]
        sweep_max = keypoints[:, 3, :]

        tip_vec = tip - center
        min_vec = sweep_min - center
        max_vec = sweep_max - center

        def _pair_features(vec_a: tf.Tensor, vec_b: tf.Tensor) -> tf.Tensor:
            dot = tf.reduce_sum(vec_a * vec_b, axis=-1, keepdims=True)
            cross = vec_a[:, 0:1] * vec_b[:, 1:2] - vec_a[:, 1:2] * vec_b[:, 0:1]
            norm_a = tf.norm(vec_a, axis=-1, keepdims=True)
            norm_b = tf.norm(vec_b, axis=-1, keepdims=True)
            angle = tf.atan2(cross, dot)
            return tf.concat([dot, cross, norm_a, norm_b, angle], axis=-1)

        features = tf.concat(
            [
                tip_vec,
                min_vec,
                max_vec,
                _pair_features(tip_vec, min_vec),
                _pair_features(tip_vec, max_vec),
                _pair_features(min_vec, max_vec),
            ],
            axis=-1,
        )
        x = self._dense_1(features)
        if self._dropout is not None:
            x = self._dropout(x, training=training)
        x = self._dense_2(x)
        if self._dropout is not None:
            x = self._dropout(x, training=training)
        raw = self._raw_out(x)
        normalized = keras.activations.sigmoid(raw)
        value = self.value_min + normalized * self._scale
        return tf.cast(value, tf.float32)

    def get_config(self) -> dict[str, object]:
        """Serialize the relation head configuration."""
        config = super().get_config()
        config.update(
            {
                "value_min": self.value_min,
                "value_max": self.value_max,
                "head_units": self.head_units,
                "dropout": self.dropout,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class GaugeValueFromNeedleDirection(keras.layers.Layer):
    """Convert a predicted unit needle direction into a calibrated value.

    The layer keeps the learning problem geometric: the network predicts a
    direction vector, we derive an angle with `atan2`, and then map that angle
    back to the gauge's calibrated value range.
    """

    def __init__(
        self,
        *,
        value_min: float,
        value_max: float,
        min_angle_rad: float,
        sweep_rad: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if value_max <= value_min:
            raise ValueError("value_max must be > value_min.")
        if sweep_rad <= 0.0:
            raise ValueError("sweep_rad must be > 0.")
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.min_angle_rad = float(min_angle_rad)
        self.sweep_rad = float(sweep_rad)
        self._two_pi = float(2.0 * math.pi)
        self._scale = float(self.value_max - self.value_min)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Map a unit direction vector into the calibrated gauge value."""
        needle_xy = tf.cast(inputs, tf.float32)
        needle_xy = tf.math.l2_normalize(needle_xy, axis=-1)
        angle = tf.atan2(needle_xy[:, 1], needle_xy[:, 0])
        shifted = tf.math.floormod(angle - self.min_angle_rad, self._two_pi)
        fraction = tf.clip_by_value(shifted / self.sweep_rad, 0.0, 1.0)
        value = self.value_min + fraction * self._scale
        return tf.expand_dims(value, axis=-1)

    def get_config(self) -> dict[str, object]:
        """Serialize the layer config for saved models."""
        config = super().get_config()
        config.update(
            {
                "value_min": self.value_min,
                "value_max": self.value_max,
                "min_angle_rad": self.min_angle_rad,
                "sweep_rad": self.sweep_rad,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class GaugeValueFromSweepDistribution(keras.layers.Layer):
    """Convert a sweep-distribution logit vector into a calibrated gauge value.

    The layer treats the output logits as a soft distribution across bins that
    span the known gauge range. We then take the expectation over the bin
    centers and map that fraction back into Celsius.
    """

    def __init__(
        self,
        *,
        value_min: float,
        value_max: float,
        num_bins: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if value_max <= value_min:
            raise ValueError("value_max must be > value_min.")
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2.")
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.num_bins = int(num_bins)
        self._bin_centers: tf.Tensor | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Precompute normalized bin centers used by the expectation layer."""
        if len(input_shape) != 2:
            raise ValueError("Sweep distribution logits must be rank 2.")
        channels = input_shape[-1]
        if channels is not None and int(channels) != self.num_bins:
            raise ValueError("Logit dimension must match num_bins.")

        self._bin_centers = tf.linspace(
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(1.0, dtype=tf.float32),
            self.num_bins,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Map a distribution over sweep bins into a Celsius prediction."""
        if self._bin_centers is None:
            raise RuntimeError("GaugeValueFromSweepDistribution was not built.")

        logits = tf.cast(inputs, tf.float32)
        probs = tf.nn.softmax(logits, axis=-1)
        fraction = tf.reduce_sum(probs * self._bin_centers, axis=-1)
        span = self.value_max - self.value_min
        value = self.value_min + fraction * span
        return tf.expand_dims(value, axis=-1)

    def get_config(self) -> dict[str, object]:
        """Serialize the layer config for saved models."""
        config = super().get_config()
        config.update(
            {
                "value_min": self.value_min,
                "value_max": self.value_max,
                "num_bins": self.num_bins,
            }
        )
        return config


def build_regression_model(image_height: int, image_width: int) -> keras.Model:
    """Build a compact residual CNN regressor for scalar gauge value output."""
    inputs, x = _build_feature_backbone(image_height, image_width)

    x = keras.layers.Dense(192, activation="swish")(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(1, name="gauge_value")(x)

    return keras.Model(inputs=inputs, outputs=output, name="gauge_value_regressor")


def build_needle_direction_model(image_height: int, image_width: int) -> keras.Model:
    """Build a compact residual CNN that predicts unit needle direction (dx, dy)."""
    inputs, x = _build_feature_backbone(image_height, image_width)

    x = keras.layers.Dense(192, activation="swish")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(2, name="needle_xy_raw")(x)
    output = keras.layers.UnitNormalization(axis=-1, name="needle_xy")(x)

    return keras.Model(inputs=inputs, outputs=output, name="needle_direction_regressor")


def build_compact_interval_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    bin_width: float = 5.0,
) -> keras.Model:
    """Build a compact CNN that predicts coarse bins plus a local residual."""
    inputs, x = _build_feature_backbone(image_height, image_width)

    x = keras.layers.Dense(192, activation="swish")(x)
    x = keras.layers.Dropout(0.2)(x)

    span = value_max - value_min
    num_bins = int(math.ceil(span / bin_width))
    if num_bins < 2:
        raise ValueError("Compact interval model needs at least two bins.")

    interval_logits = keras.layers.Dense(
        num_bins,
        name="interval_logits",
    )(x)
    gauge_value, interval_probs = _build_interval_expectation_head(
        interval_logits,
        value_min=value_min,
        value_max=value_max,
        bin_width=bin_width,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "interval_logits": interval_logits,
        },
        name="compact_interval_gauge_regressor",
    )
    setattr(model, "_compact_interval_probs", interval_probs)
    return model


def build_compact_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
) -> keras.Model:
    """Build a compact CNN with explicit keypoint heatmaps and geometry value."""
    inputs, spatial_features, _pooled_features = _build_compact_geometry_backbone(
        image_height,
        image_width,
    )
    heatmaps = _build_keypoint_heatmap_head(
        spatial_features,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
        },
        name="compact_geometry_gauge_regressor",
    )
    return model


def build_compact_source_crop_box_model(
    image_height: int,
    image_width: int,
    *,
    head_units: int = 96,
    head_dropout: float = 0.15,
) -> keras.Model:
    """Build a compact CNN that regresses a normalized source-space crop box."""
    inputs, x = _build_feature_backbone(image_height, image_width)

    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="source_crop_box_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="source_crop_box_dropout")(x)

    raw_box = keras.layers.Dense(
        4,
        activation="sigmoid",
        name="source_crop_box_raw",
    )(x)
    source_crop_box = OrderedCornerBox(name="source_crop_box")(raw_box)

    return keras.Model(
        inputs=inputs,
        outputs={"source_crop_box": source_crop_box},
        name="compact_source_crop_box_regressor",
    )


def build_mobilenetv2_regression_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
    linear_output: bool = False,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build a transfer-learning regressor on top of MobileNetV2 features.

    Args:
        linear_output: If True, use a linear output head on the normalized target.
        value_min: Minimum gauge value for output scaling.
        value_max: Maximum gauge value for output scaling.
    """
    if linear_output:
        pass  # linear output enabled for range-compression-free regression
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    span = value_max - value_min
    if linear_output:
        # Predict a normalized fraction directly and then map it back to Celsius.
        x = keras.layers.Dense(1, activation="linear", name="gauge_value_linear")(x)
        output = keras.layers.Rescaling(
            scale=span,
            offset=value_min,
            name="gauge_value",
        )(x)
    else:
        # Sigmoid output bounded to [0,1], then rescaled to value range.
        x = keras.layers.Dense(1, activation="sigmoid", name="gauge_value_sigmoid")(x)
        # Rescale sigmoid [0,1] output to [value_min, value_max].
        output = keras.layers.Rescaling(
            scale=span,
            offset=value_min,
            name="gauge_value",
        )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="gauge",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    # Store the backbone so training can run staged freeze/unfreeze schedules.
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_tiny_regression_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
) -> keras.Model:
    """Build a narrower MobileNetV2 regressor sized for STM32N6 deployment."""
    return build_mobilenetv2_regression_model(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=0.35,
        head_units=64,
        head_dropout=0.15,
    )


def build_mobilenetv2_direction_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a transfer-learning model that predicts unit needle direction."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(2, name="needle_xy_raw")(x)
    output = keras.layers.UnitNormalization(axis=-1, name="needle_xy")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="needle_direction",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_direction_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a MobileNetV2 model that predicts needle direction then value.

    The backbone learns a compact needle direction embedding, the unit-vector
    head keeps the geometry explicit, and the final scalar value is computed
    from that direction using the calibrated sweep mapping.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(2, name="needle_xy_raw")(x)
    needle_xy = keras.layers.UnitNormalization(axis=-1, name="needle_xy")(x)
    gauge_value = GaugeValueFromNeedleDirection(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(needle_xy)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "needle_xy": needle_xy,
        },
        name=_mobilenetv2_model_name(
            regression_kind="needle_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_interval_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    bin_width: float = 5.0,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    span = value_max - value_min
    num_bins = int(math.ceil(span / bin_width))
    if num_bins < 2:
        raise ValueError("Interval model needs at least two bins.")

    interval_logits = keras.layers.Dense(
        num_bins,
        name="interval_logits",
    )(x)
    gauge_value, interval_probs = _build_interval_expectation_head(
        interval_logits,
        value_min=value_min,
        value_max=value_max,
        bin_width=bin_width,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "interval_logits": interval_logits,
        },
        name=_mobilenetv2_model_name(
            regression_kind="interval",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    setattr(model, "_mobilenet_interval_probs", interval_probs)
    return model


def build_mobilenetv2_ordinal_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    threshold_step: float = 2.5,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a transfer-learning regressor with an ordinal threshold head."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    span = value_max - value_min
    num_thresholds = int(math.ceil(span / threshold_step))
    if num_thresholds < 2:
        raise ValueError("Need at least two thresholds for ordinal regression.")

    ordinal_logits = keras.layers.Dense(
        num_thresholds,
        name="ordinal_logits",
    )(x)
    gauge_value, ordinal_probs = _build_ordinal_expectation_head(
        ordinal_logits,
        value_min=value_min,
        value_max=value_max,
        threshold_step=threshold_step,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "ordinal_logits": ordinal_logits,
        },
        name=_mobilenetv2_model_name(
            regression_kind="ordinal",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    setattr(model, "_mobilenet_ordinal_probs", ordinal_probs)
    return model


def build_mobilenetv2_fraction_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a transfer-learning regressor with a normalized sweep-fraction head."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    sweep_fraction_logits = keras.layers.Dense(
        1,
        name="sweep_fraction_logits",
    )(x)
    gauge_value, sweep_fraction = _build_sweep_fraction_head(
        sweep_fraction_logits,
        value_min=value_min,
        value_max=value_max,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "sweep_fraction": sweep_fraction,
        },
        name=_mobilenetv2_model_name(
            regression_kind="fraction",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_dual_resolution_regression_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.35,
    head_units: int = 96,
    head_dropout: float = 0.2,
    crop_ratio: float = 0.78,
    linear_output: bool = True,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build a dual-resolution MobileNetV2 regressor for gauge value."""
    inputs, features, base_model = _build_mobilenetv2_dual_resolution_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
        crop_ratio=crop_ratio,
    )

    x = keras.layers.Dense(head_units * 2, activation="swish")(features)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    span = value_max - value_min
    if linear_output:
        # Predict an unrestricted normalized scalar and map it back to Celsius.
        x = keras.layers.Dense(
            1,
            activation="linear",
            name="gauge_value_linear",
        )(x)
    else:
        # Keep a bounded head when the caller wants the older sigmoid behavior.
        x = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="gauge_value_sigmoid",
        )(x)

    output = keras.layers.Rescaling(
        scale=span,
        offset=value_min,
        name="gauge_value",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="dualres",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_dual_resolution_interval_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    bin_width: float = 5.0,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 0.35,
    head_units: int = 96,
    head_dropout: float = 0.2,
    crop_ratio: float = 0.78,
) -> keras.Model:
    """Build a dual-resolution MobileNetV2 model with interval supervision."""
    inputs, features, base_model = _build_mobilenetv2_dual_resolution_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
        crop_ratio=crop_ratio,
    )

    x = keras.layers.Dense(head_units * 2, activation="swish")(features)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    span = value_max - value_min
    num_bins = int(math.ceil(span / bin_width))
    if num_bins < 2:
        raise ValueError("Interval model needs at least two bins.")

    interval_logits = keras.layers.Dense(
        num_bins,
        name="interval_logits",
    )(x)
    gauge_value, interval_probs = _build_interval_expectation_head(
        interval_logits,
        value_min=value_min,
        value_max=value_max,
        bin_width=bin_width,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "interval_logits": interval_logits,
        },
        name=_mobilenetv2_model_name(
            regression_kind="dualres_interval",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    setattr(model, "_mobilenet_interval_probs", interval_probs)
    return model


def build_mobilenetv2_polar_dualview_regression_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
    linear_output: bool = False,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build a polar-dualview MobileNetV2 regressor for gauge value.

    The full-frame branch keeps the broader scene context while the polar branch
    turns the circular dial into a flattened angle/radius image. This is a
    better fit for the gauge geometry than another center-crop-only variant.
    """
    inputs, features, base_model = _build_mobilenetv2_polar_dualview_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    x = keras.layers.Dense(head_units, activation="swish", name="polar_dense")(features)
    x = keras.layers.Dropout(head_dropout, name="polar_dropout")(x)

    span = value_max - value_min
    if linear_output:
        # Predict an unrestricted normalized scalar and map it back to Celsius.
        x = keras.layers.Dense(
            1,
            activation="linear",
            name="gauge_value_linear",
        )(x)
    else:
        # Keep a bounded head when the caller wants the older sigmoid behavior.
        x = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="gauge_value_sigmoid",
        )(x)

    output = keras.layers.Rescaling(
        scale=span,
        offset=value_min,
        name="gauge_value",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="polar_dualview",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_polar_regression_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
    linear_output: bool = False,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build a single-input polar-unwrapped MobileNetV2 regressor.

    Compared with the dual-view branch, this version keeps one input tensor and
    one shared MobileNetV2 trunk. It is less expressive, but it is much easier
    to train, debug, and export on the constrained WSL/GPU setup we have here.
    """
    inputs, features, base_model = _build_mobilenetv2_polar_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    x = keras.layers.Dense(head_units, activation="swish", name="polar_dense")(features)
    x = keras.layers.Dropout(head_dropout, name="polar_dropout")(x)

    span = value_max - value_min
    if linear_output:
        # Predict a normalized scalar directly and then map it back to Celsius.
        x = keras.layers.Dense(
            1,
            activation="linear",
            name="gauge_value_linear",
        )(x)
    else:
        # Keep the older bounded form if the caller wants a sigmoid output.
        x = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="gauge_value_sigmoid",
        )(x)

    output = keras.layers.Rescaling(
        scale=span,
        offset=value_min,
        name="gauge_value",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="polar",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_polar_sweep_distribution_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    num_bins: int = 81,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a polar gauge reader with a distributional sweep head.

    The polar warp flattens the circular dial into an angle-versus-radius view,
    while the sweep distribution head preserves ordinal structure better than a
    plain scalar regressor. This keeps the model compact but more geometry-aware
    than the earlier polar-only baseline.
    """
    inputs, features, base_model = _build_mobilenetv2_polar_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # The polar backbone already returns a pooled 2D feature vector, so the
    # distribution head can work directly from that compact representation.
    sweep_features = keras.layers.Dropout(head_dropout)(features)
    sweep_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="polar_sweep_dense",
    )(sweep_features)
    sweep_features = keras.layers.Dropout(head_dropout)(sweep_features)
    sweep_distribution_logits = keras.layers.Dense(
        num_bins,
        name="sweep_distribution_logits",
    )(sweep_features)
    gauge_value = GaugeValueFromSweepDistribution(
        value_min=value_min,
        value_max=value_max,
        num_bins=num_bins,
        name="gauge_value",
    )(sweep_distribution_logits)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "sweep_distribution_logits": sweep_distribution_logits,
        },
        name=_mobilenetv2_model_name(
            regression_kind="polar_sweep_distribution",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_sweep_distribution_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    num_bins: int = 81,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a gauge reader that predicts a smooth sweep distribution.

    The network learns a value distribution over the known Celsius range and a
    deterministic expectation layer turns that distribution back into the final
    scalar temperature. This is a more geometry-aware alternative to direct
    scalar regression.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    features = keras.layers.GlobalAveragePooling2D(name="sweep_distribution_gap")(x)
    features = keras.layers.Dropout(head_dropout)(features)
    features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="sweep_distribution_dense",
    )(features)
    features = keras.layers.Dropout(head_dropout)(features)
    sweep_distribution_logits = keras.layers.Dense(
        num_bins,
        name="sweep_distribution_logits",
    )(features)
    gauge_value = GaugeValueFromSweepDistribution(
        value_min=value_min,
        value_max=value_max,
        num_bins=num_bins,
        name="gauge_value",
    )(sweep_distribution_logits)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "sweep_distribution_logits": sweep_distribution_logits,
        },
        name=_mobilenetv2_model_name(
            regression_kind="sweep_distribution",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_keypoint_model(
    image_height: int,
    image_width: int,
    *,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a transfer-learning model with a scalar head plus keypoint heatmaps."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    scalar_features = keras.layers.GlobalAveragePooling2D()(x)
    scalar_features = keras.layers.Dropout(head_dropout)(scalar_features)
    scalar_features = keras.layers.Dense(head_units, activation="swish")(
        scalar_features
    )
    scalar_features = keras.layers.Dropout(head_dropout)(scalar_features)
    gauge_value = keras.layers.Dense(1, name="gauge_value")(scalar_features)

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
        },
        name=_mobilenetv2_model_name(
            regression_kind="keypoint",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_detector_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a detector-first model that turns keypoints into gauge value."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
        },
        name=_mobilenetv2_model_name(
            regression_kind="detector",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a geometry-first model with explicit keypoint and value outputs."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
        },
        name=_mobilenetv2_model_name(
            regression_kind="geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_geometry_uncertainty_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a geometry model with symmetric uncertainty bounds around the value."""
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="geometry_pooled_features"
    )(x)
    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    uncertainty_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="geometry_uncertainty_dense",
    )(pooled_features)
    uncertainty_features = keras.layers.Dropout(head_dropout)(uncertainty_features)
    interval_radius = keras.layers.Dense(
        1,
        activation="softplus",
        name="gauge_value_interval_radius",
    )(uncertainty_features)
    half_interval = keras.layers.Rescaling(
        0.5,
        name="gauge_value_half_interval",
    )(interval_radius)
    gauge_value_lower = keras.layers.Subtract(name="gauge_value_lower")(
        [gauge_value, half_interval]
    )
    gauge_value_upper = keras.layers.Add(name="gauge_value_upper")(
        [gauge_value, half_interval]
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "gauge_value_lower": gauge_value_lower,
            "gauge_value_upper": gauge_value_upper,
        },
        name=_mobilenetv2_model_name(
            regression_kind="geometry_uncertainty",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_obb_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a joint OBB-plus-geometry model for detector-style gauge reading.

    The OBB branch encourages the network to stay aware of the full dial extent,
    while the keypoint branch turns that localized view into an explicit pointer
    geometry and the final Celsius value. This keeps the architecture closer to
    the recent literature than a pure scalar head.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_geometry_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_geometry_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    # Keep the OBB branch compact and directly supervised so it stays useful as
    # a localizer instead of competing with the geometry head for capacity.
    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    # The geometry branch reuses the dense feature map and predicts heatmaps
    # for the needle center/tip, which the soft-argmax layer turns into explicit
    # coordinates. That gives the final value a structural path rather than a
    # pure scalar shortcut.
    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="obb_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_obb_mask_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
    num_keypoints: int = 2,
) -> keras.Model:
    """Build an OBB-plus-mask model that combines localization and segmentation."""
    if num_keypoints < 2:
        raise ValueError("num_keypoints must be >= 2.")

    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_mask_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_mask_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=num_keypoints,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=num_keypoints,
        name="keypoint_coords",
    )(heatmaps)
    pointer_mask = _build_pointer_mask_head(
        x,
        mask_size=heatmap_size,
    )
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "pointer_mask": pointer_mask,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="obb_mask_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_obb_relation_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a blur-aware OBB-plus-relation model.

    This keeps the proven OBB relation family but adds a lightweight unsharp
    branch so low-contrast crops and preview frames have a better chance of
    surfacing the pointer/scale structure the reader needs.
    """
    if heatmap_size < 4:
        raise ValueError("heatmap_size must be >= 4.")

    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    raw_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="obb_relation_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="obb_relation",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="obb_relation_feature_average")(
        [raw_maps, enhanced_maps]
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_relation_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_relation_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=4,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=4,
        name="keypoint_coords",
    )(heatmaps)
    pointer_mask = _build_pointer_mask_head(
        x,
        mask_size=heatmap_size,
    )
    gauge_value = GaugeValueFromRelationKeypoints(
        value_min=value_min,
        value_max=value_max,
        head_units=head_units,
        dropout=head_dropout,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "pointer_mask": pointer_mask,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="obb_relation_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


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


def build_mobilenetv2_bluraware_obb_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a blur-aware OBB-geometry model with raw and sharpened views."""
    inputs, _, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    raw_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="bluraware_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="bluraware",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="bluraware_feature_average")(
        [raw_maps, enhanced_maps]
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_geometry_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_geometry_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=2,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=2,
        name="keypoint_coords",
    )(heatmaps)
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="bluraware_obb_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_bluraware_obb_relation_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a blur-aware OBB reader with relation keypoints and mask output.

    This variant keeps the OBB backbone intact, adds a fixed unsharp-mask
    branch for low-contrast crops, and then learns pointer-mask plus relation
    geometry outputs on top. The design mirrors the recent literature trend:
    localize first, then reason about the pointer/scale relation explicitly.
    """
    if heatmap_size < 4:
        raise ValueError("heatmap_size must be >= 4.")

    inputs, _, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    raw_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="bluraware_relation_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="bluraware_relation",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="bluraware_relation_feature_average")(
        [raw_maps, enhanced_maps]
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_relation_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_relation_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=4,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=4,
        name="keypoint_coords",
    )(heatmaps)
    pointer_mask = _build_pointer_mask_head(
        x,
        mask_size=heatmap_size,
    )
    gauge_value = GaugeValueFromRelationKeypoints(
        value_min=value_min,
        value_max=value_max,
        head_units=head_units,
        dropout=head_dropout,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "pointer_mask": pointer_mask,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="bluraware_obb_relation_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_bluraware_obb_sequence_geometry_model(
    image_height: int,
    image_width: int,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a blur-aware OBB sequence-geometry model.

    This keeps the OBB front-end fixed, adds a cheap unsharp-mask branch for
    preview-like crops, and learns the pointer sequence with four keypoints
    plus a pointer mask. The value head still converts keypoints into the final
    gauge reading, which matches the literature-backed geometry-first pattern.
    """
    if heatmap_size < 4:
        raise ValueError("heatmap_size must be >= 4.")

    inputs, _, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # Blend the raw crop with a lightly sharpened view so low-contrast frames
    # still expose thin pointer structure to the shared MobileNet backbone.
    raw_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="bluraware_sequence_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="bluraware_sequence",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="bluraware_sequence_feature_average")(
        [raw_maps, enhanced_maps]
    )

    pooled_features = keras.layers.GlobalAveragePooling2D(
        name="obb_sequence_pooled_features"
    )(x)
    pooled_features = keras.layers.Dropout(head_dropout)(pooled_features)
    obb_features = keras.layers.Dense(
        head_units,
        activation="swish",
        name="obb_sequence_dense",
    )(pooled_features)
    obb_features = keras.layers.Dropout(head_dropout)(obb_features)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(obb_features)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(obb_features)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(obb_features)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=4,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=4,
        name="keypoint_coords",
    )(heatmaps)
    pointer_mask = _build_pointer_mask_head(
        x,
        mask_size=heatmap_size,
    )
    gauge_value = GaugeValueFromKeypoints(
        value_min=value_min,
        value_max=value_max,
        min_angle_rad=min_angle_rad,
        sweep_rad=sweep_rad,
        name="gauge_value",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
            "pointer_mask": pointer_mask,
            "obb_params": obb_params,
        },
        name=_mobilenetv2_model_name(
            regression_kind="bluraware_obb_sequence_geometry",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_bluraware_reader_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
    linear_output: bool = False,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build a blur-aware reader that fuses raw and unsharp-masked views.

    The recent gauge-reading papers consistently favor geometry-aware readers,
    but our board path already has a strong OBB localizer. This reader keeps the
    crop fixed and spends its capacity on low-contrast detail recovery by fusing
    the raw crop with a lightweight unsharp-mask branch before the scalar head.
    """
    inputs, _, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    raw_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="bluraware_reader_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="bluraware_reader",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="bluraware_reader_feature_average")(
        [raw_maps, enhanced_maps]
    )

    x = keras.layers.GlobalAveragePooling2D(name="bluraware_reader_pooled_features")(x)
    x = keras.layers.Dropout(head_dropout, name="bluraware_reader_dropout_1")(x)
    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="bluraware_reader_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="bluraware_reader_dropout_2")(x)

    span = value_max - value_min
    if linear_output:
        gauge_value = keras.layers.Dense(
            1,
            activation="linear",
            name="gauge_value_linear",
        )(x)
    else:
        gauge_value = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="gauge_value_sigmoid",
        )(x)

    gauge_value = keras.layers.Rescaling(
        scale=span,
        offset=value_min,
        name="gauge_value",
    )(gauge_value)

    model = keras.Model(
        inputs=inputs,
        outputs=gauge_value,
        name=_mobilenetv2_model_name(
            regression_kind="bluraware_reader",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_bluraware_reader_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
    linear_output: bool = False,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build a blur-aware scalar reader with raw and sharpened crop branches."""
    inputs, _, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    raw_branch = keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1.0,
        name="bluraware_reader_raw_preprocess",
    )(inputs)
    enhanced_branch = _build_unsharp_mask_branch(
        raw_branch,
        name_prefix="bluraware_reader",
    )

    raw_maps = base_model(raw_branch, training=backbone_trainable)
    enhanced_maps = base_model(enhanced_branch, training=backbone_trainable)
    x = keras.layers.Average(name="bluraware_reader_feature_average")(
        [raw_maps, enhanced_maps]
    )

    x = keras.layers.GlobalAveragePooling2D(name="bluraware_reader_pooled_features")(x)
    x = keras.layers.Dropout(head_dropout, name="bluraware_reader_dropout_1")(x)
    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="bluraware_reader_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="bluraware_reader_dropout_2")(x)

    span = value_max - value_min
    if linear_output:
        x = keras.layers.Dense(
            1,
            activation="linear",
            name="gauge_value_linear",
        )(x)
    else:
        x = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="gauge_value_sigmoid",
        )(x)

    output = keras.layers.Rescaling(
        scale=span,
        offset=value_min,
        name="gauge_value",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output,
        name=_mobilenetv2_model_name(
            regression_kind="bluraware_reader",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_rectifier_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a compact rectifier model that predicts a normalized dial crop box."""
    _ = pretrained
    _ = backbone_trainable
    _ = alpha

    # Keep the rectifier lightweight so the two-stage STM32 build stays within
    # the board flash budget while still learning a useful crop proposal.
    inputs, x = _build_feature_backbone(image_height, image_width)
    x = keras.layers.LayerNormalization(name="rectifier_pooled_features")(x)
    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="rectifier_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="rectifier_dropout")(x)

    # Predict normalized center-x, center-y, width, height in [0, 1].
    box = keras.layers.Dense(
        4,
        activation="sigmoid",
        name="rectifier_box",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs={"rectifier_box": box},
        name=_mobilenetv2_model_name(
            regression_kind="rectifier",
            alpha=1.0,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", None)
    return model


def build_mobilenetv2_source_crop_box_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a MobileNetV2 crop-box regressor over source-image coordinates.

    The head predicts two x corners and two y corners directly in normalized
    source-frame coordinates. This keeps the learning target closer to the
    offline rectified oracle than the older center/size rectifier.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    x = keras.layers.GlobalAveragePooling2D(name="source_crop_box_gap")(x)
    x = keras.layers.Dropout(head_dropout, name="source_crop_box_dropout_1")(x)
    x = keras.layers.Dense(
        head_units,
        activation="swish",
        name="source_crop_box_dense",
    )(x)
    x = keras.layers.Dropout(head_dropout, name="source_crop_box_dropout_2")(x)

    raw_box = keras.layers.Dense(
        4,
        activation="sigmoid",
        name="source_crop_box_raw",
    )(x)
    source_crop_box = OrderedCornerBox(name="source_crop_box")(raw_box)

    model = keras.Model(
        inputs=inputs,
        outputs={"source_crop_box": source_crop_box},
        name=_mobilenetv2_model_name(
            regression_kind="source_crop_box",
            alpha=1.0,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_source_crop_corner_model(
    image_height: int,
    image_width: int,
    *,
    heatmap_size: int = 28,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a corner-localizer that decodes four heatmaps into a crop box.

    This candidate keeps the localization problem geometric: it learns the
    four crop corners with heatmaps, converts those to corner coordinates, and
    then reduces the coordinates to an axis-aligned source crop box.
    """
    _ = head_units
    _ = head_dropout

    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    heatmaps = _build_keypoint_heatmap_head(
        x,
        heatmap_size=heatmap_size,
        num_keypoints=4,
    )
    keypoints = SpatialSoftArgmax2D(
        heatmap_size=heatmap_size,
        num_keypoints=4,
        name="keypoint_coords",
    )(heatmaps)
    source_crop_canvas_box = CornerKeypointsToBox(
        heatmap_size=heatmap_size,
        name="source_crop_canvas_box",
    )(keypoints)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "source_crop_canvas_box": source_crop_canvas_box,
            "keypoint_heatmaps": heatmaps,
            "keypoint_coords": keypoints,
        },
        name=_mobilenetv2_model_name(
            regression_kind="source_crop_corner",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_obb_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = False,
    alpha: float = 1.0,
    head_units: int = 128,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build a MobileNetV2 localizer that predicts oriented ellipse parameters.

    The head predicts a normalized center and size plus a unit angle vector.
    That gives us a more explicit detector-style target than the earlier
    keypoint heatmap proxies while still staying compact enough for embedded
    experiments.
    """
    inputs, x, base_model = _build_mobilenetv2_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_center_xy",
    )(x)
    size_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        name="obb_size_wh",
    )(x)
    angle_raw = keras.layers.Dense(
        2,
        name="obb_angle_raw",
    )(x)
    angle_sincos = keras.layers.UnitNormalization(
        axis=-1,
        name="obb_angle_sincos",
    )(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")(
        [center_xy, size_wh, angle_sincos]
    )

    model = keras.Model(
        inputs=inputs,
        outputs={"obb_params": obb_params},
        name=_mobilenetv2_model_name(
            regression_kind="obb",
            alpha=alpha,
            head_units=head_units,
        ),
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


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


def build_mobilenetv2_enhanced_regression_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = True,
    alpha: float = 1.0,
    head_units: int = 256,
    head_dropout: float = 0.3,
    value_min: float = -30.0,
    value_max: float = 50.0,
    use_multi_scale: bool = True,
    use_coord_attention: bool = False,
) -> keras.Model:
    """Build an enhanced MobileNetV2 regressor with multi-scale features and attention.

    Architecture improvements over the baseline mobilenet_v2:
    1. Multi-scale feature fusion — combines early/mid/late MobileNetV2 features
       so the head sees both fine needle detail and global dial context.
    2. CBAM attention — channel + spatial attention at each scale to suppress
       background clutter and highlight the needle region.
    3. Wider head — 256 units with 0.3 dropout for more capacity.
    4. LayerNorm + residual connections in the head for training stability.
    5. Linear output — no sigmoid saturation at temperature extremes.
    6. Auxiliary sweep-fraction head — provides geometric supervision.

    Reference: Woo et al. "CBAM" (ECCV 2018), Hou et al. "Coordinate Attention" (CVPR 2021)
    """
    span = value_max - value_min

    if use_multi_scale:
        # Multi-scale backbone with feature pyramid
        inputs, multi_scale_features, base_model = (
            _build_mobilenetv2_multi_scale_backbone(
                image_height,
                image_width,
                pretrained=pretrained,
                backbone_trainable=backbone_trainable,
                alpha=alpha,
            )
        )
        # Fuse multi-scale features with CBAM
        spatial_features = _cbam_refine(multi_scale_features, base_channels=64)
        # Global pooling for regression head
        x = keras.layers.GlobalAveragePooling2D(name="enhanced_gap")(spatial_features)
        # Also keep max-pooled features for sharper responses
        x_max = keras.layers.GlobalMaxPooling2D(name="enhanced_gmp")(spatial_features)
        x = keras.layers.Concatenate(name="enhanced_pool_fusion")([x, x_max])
    else:
        # Standard backbone with coordinate attention on final features
        inputs, final_features, base_model = _build_mobilenetv2_backbone(
            image_height,
            image_width,
            pretrained=pretrained,
            backbone_trainable=backbone_trainable,
            alpha=alpha,
        )
        if use_coord_attention:
            final_features = CoordinateAttention(
                reduction_ratio=16, name="enhanced_coord_attn"
            )(final_features)
        x = keras.layers.GlobalAveragePooling2D(name="enhanced_gap")(final_features)
        x_max = keras.layers.GlobalMaxPooling2D(name="enhanced_gmp")(final_features)
        x = keras.layers.Concatenate(name="enhanced_pool_fusion")([x, x_max])

    # Wider, deeper regression head with LayerNorm for stability
    x = keras.layers.LayerNormalization(name="enhanced_head_norm_0")(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(head_units, activation="swish", name="enhanced_dense_1")(x)
    x = keras.layers.LayerNormalization(name="enhanced_head_norm_1")(x)
    x = keras.layers.Dropout(head_dropout)(x)
    x = keras.layers.Dense(
        head_units // 2, activation="swish", name="enhanced_dense_2"
    )(x)
    x = keras.layers.LayerNormalization(name="enhanced_head_norm_2")(x)
    x = keras.layers.Dropout(head_dropout * 0.5)(x)

    # Main scalar output — linear (no sigmoid) to avoid saturation at extremes
    main_output = keras.layers.Dense(1, name="gauge_value_raw")(x)
    gauge_value = keras.layers.Rescaling(
        scale=span, offset=value_min, name="gauge_value"
    )(main_output)

    # Auxiliary sweep-fraction head for geometric supervision
    fraction_logit = keras.layers.Dense(1, name="sweep_fraction_raw")(x)
    sweep_fraction = keras.layers.Activation("sigmoid", name="sweep_fraction")(
        fraction_logit
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "sweep_fraction": sweep_fraction,
        },
        name=f"mobilenetv2_enhanced_gauge_regressor",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_enhanced_ensemble_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = True,
    alpha: float = 1.0,
    head_units: int = 256,
    head_dropout: float = 0.3,
    value_min: float = -30.0,
    value_max: float = 50.0,
    num_heads: int = 3,
) -> keras.Model:
    """Build an enhanced MobileNetV2 with multi-head ensemble.

    Uses a shared backbone with multiple independent regression heads.
    At inference, heads are averaged for a more robust prediction.
    Each head gets its own dropout pattern for diversity.
    """
    span = value_max - value_min

    inputs, multi_scale_features, base_model = _build_mobilenetv2_multi_scale_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )
    spatial_features = _cbam_refine(multi_scale_features, base_channels=64)
    x = keras.layers.GlobalAveragePooling2D(name="ensemble_gap")(spatial_features)
    x_max = keras.layers.GlobalMaxPooling2D(name="ensemble_gmp")(spatial_features)
    shared = keras.layers.Concatenate(name="ensemble_pool_fusion")([x, x_max])
    shared = keras.layers.LayerNormalization(name="ensemble_shared_norm")(shared)

    head_outputs: list[keras.KerasTensor] = []
    for i in range(num_heads):
        h = keras.layers.Dropout(head_dropout, name=f"ensemble_head_{i}_drop_1")(shared)
        h = keras.layers.Dense(head_units, use_bias=False, name=f"ensemble_dense_{i}")(
            h
        )
        h = keras.layers.LayerNormalization(name=f"ensemble_ln_{i}")(h)
        h = keras.layers.Activation("swish")(h)
        h = keras.layers.Dropout(head_dropout)(h)

        h_linear = keras.layers.Dense(
            1, activation="linear", name=f"gauge_value_head_{i}_linear"
        )(h)
        h_out = keras.layers.Rescaling(
            scale=span, offset=value_min, name=f"gauge_value_head_{i}"
        )(h_linear)
        head_outputs.append(h_out)

    # Average all heads
    if num_heads > 1:
        avg_raw = keras.layers.Average(name="ensemble_avg")(head_outputs)
    else:
        avg_raw = head_outputs[0]

    gauge_value = keras.layers.Rescaling(
        scale=span, offset=value_min, name="gauge_value"
    )(avg_raw)

    model = keras.Model(
        inputs=inputs,
        outputs={"gauge_value": gauge_value},
        name=f"mobilenetv2_enhanced_ensemble_{num_heads}h_gauge_regressor",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_sota_multiscale_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = True,
    alpha: float = 1.0,
    head_units: int = 256,
    head_dropout: float = 0.3,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build SOTA model with multi-scale features and dual attention.

    This model incorporates the best practices from gauge-reading literature:
    1. Multi-scale feature fusion (FPN-style)
    2. CBAM + Coordinate Attention
    3. Wide head with LayerNorm
    4. Linear output (no saturation)
    5. Auxiliary sweep fraction head

    Args:
        image_height: Input image height
        image_width: Input image width
        pretrained: Use ImageNet weights
        backbone_trainable: Fine-tune backbone
        alpha: MobileNetV2 width multiplier
        head_units: Head FC layer units
        head_dropout: Dropout rate
        value_min: Minimum temperature value
        value_max: Maximum temperature value

    Returns:
        Keras model with gauge_value output
    """
    span = value_max - value_min

    # Build multi-scale backbone
    inputs, multi_scale_features, base_model = _build_mobilenetv2_multi_scale_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # Fuse multi-scale features with CBAM
    spatial_features = _cbam_refine(multi_scale_features, base_channels=64)

    # Global pooling fusion
    x = keras.layers.GlobalAveragePooling2D(name="sota_gap")(spatial_features)
    x_max = keras.layers.GlobalMaxPooling2D(name="sota_gmp")(spatial_features)
    x = keras.layers.Concatenate(name="sota_pool_fusion")([x, x_max])

    # Wide head with LayerNorm
    x = keras.layers.Dense(head_units, use_bias=False, name="sota_dense_1")(x)
    x = keras.layers.LayerNormalization(name="sota_ln_1")(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    x = keras.layers.Dense(head_units // 2, use_bias=False, name="sota_dense_2")(x)
    x = keras.layers.LayerNormalization(name="sota_ln_2")(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Dropout(head_dropout * 0.5)(x)

    # Linear output head
    gauge_value_linear = keras.layers.Dense(
        1, activation="linear", name="gauge_value_linear"
    )(x)
    gauge_value = keras.layers.Rescaling(
        scale=span, offset=value_min, name="gauge_value"
    )(gauge_value_linear)

    # Auxiliary sweep fraction head for geometric supervision
    sweep_fraction_logits = keras.layers.Dense(1, name="sweep_fraction_logits")(x)
    sweep_fraction = keras.layers.Activation("sigmoid", name="sweep_fraction")(
        sweep_fraction_logits
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "sweep_fraction": sweep_fraction,
        },
        name="mobilenetv2_sota_multiscale",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_sota_ensemble_model(
    image_height: int,
    image_width: int,
    *,
    num_heads: int = 3,
    pretrained: bool = True,
    backbone_trainable: bool = True,
    alpha: float = 1.0,
    head_units: int = 192,
    head_dropout: float = 0.25,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build ensemble model with multiple independent heads.

    Ensemble averaging reduces variance and improves generalization,
    especially on hard cases. Each head sees the same multi-scale
    features but learns independent weights.
    """
    span = value_max - value_min

    # Build shared multi-scale backbone
    inputs, multi_scale_features, base_model = _build_mobilenetv2_multi_scale_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # Fuse features
    spatial_features = _cbam_refine(multi_scale_features, base_channels=64)
    x = keras.layers.GlobalAveragePooling2D(name="ensemble_gap")(spatial_features)

    # Create multiple independent heads
    head_outputs = []
    for i in range(num_heads):
        h = keras.layers.Dense(head_units, use_bias=False, name=f"ensemble_dense_{i}")(
            x
        )
        h = keras.layers.LayerNormalization(name=f"ensemble_ln_{i}")(h)
        h = keras.layers.Activation("swish")(h)
        h = keras.layers.Dropout(head_dropout)(h)

        h_linear = keras.layers.Dense(
            1, activation="linear", name=f"gauge_value_head_{i}_linear"
        )(h)
        h_out = keras.layers.Rescaling(
            scale=span, offset=value_min, name=f"gauge_value_head_{i}"
        )(h_linear)
        head_outputs.append(h_out)

    # Average ensemble predictions
    if len(head_outputs) > 1:
        gauge_value = keras.layers.Average(name="gauge_value_ensemble")(head_outputs)
    else:
        gauge_value = head_outputs[0]

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            **{f"gauge_value_head_{i}": head_outputs[i] for i in range(num_heads)},
        },
        name=f"mobilenetv2_sota_ensemble_{num_heads}h",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model


def build_mobilenetv2_uncertainty_model(
    image_height: int,
    image_width: int,
    *,
    pretrained: bool = True,
    backbone_trainable: bool = True,
    alpha: float = 1.0,
    head_units: int = 256,
    head_dropout: float = 0.3,
    value_min: float = -30.0,
    value_max: float = 50.0,
) -> keras.Model:
    """Build model with uncertainty estimation via quantile regression.

    Predicts median, lower, and upper quantiles to estimate
    prediction uncertainty. Useful for identifying hard cases
    and low-confidence predictions.

    Args:
        image_height: Input image height
        image_width: Input image width
        pretrained: Use ImageNet weights
        backbone_trainable: Fine-tune backbone
        alpha: MobileNetV2 width multiplier
        head_units: Head FC layer units
        head_dropout: Dropout rate
        value_min: Minimum temperature value
        value_max: Maximum temperature value

    Returns:
        Keras model with gauge_value and uncertainty bounds
    """
    span = value_max - value_min

    # Build multi-scale backbone
    inputs, multi_scale_features, base_model = _build_mobilenetv2_multi_scale_backbone(
        image_height,
        image_width,
        pretrained=pretrained,
        backbone_trainable=backbone_trainable,
        alpha=alpha,
    )

    # Fuse features
    spatial_features = _cbam_refine(multi_scale_features, base_channels=64)
    x = keras.layers.GlobalAveragePooling2D(name="uncertainty_gap")(spatial_features)
    x_max = keras.layers.GlobalMaxPooling2D(name="uncertainty_gmp")(spatial_features)
    x = keras.layers.Concatenate(name="uncertainty_pool_fusion")([x, x_max])

    # Wide head
    x = keras.layers.Dense(head_units, use_bias=False, name="uncertainty_dense_1")(x)
    x = keras.layers.LayerNormalization(name="uncertainty_ln_1")(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Dropout(head_dropout)(x)

    x = keras.layers.Dense(head_units // 2, use_bias=False, name="uncertainty_dense_2")(
        x
    )
    x = keras.layers.LayerNormalization(name="uncertainty_ln_2")(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Dropout(head_dropout * 0.5)(x)

    # Predict median (main value)
    median_linear = keras.layers.Dense(1, activation="linear", name="median_linear")(x)
    gauge_value = keras.layers.Rescaling(
        scale=span, offset=value_min, name="gauge_value"
    )(median_linear)

    # Predict uncertainty bounds (learned offsets)
    # Use softplus to ensure positive intervals
    lower_offset = keras.layers.Dense(1, activation="softplus", name="lower_offset")(x)
    upper_offset = keras.layers.Dense(1, activation="softplus", name="upper_offset")(x)

    # Compute bounds
    gauge_value_lower = keras.layers.Subtract(name="gauge_value_lower")(
        [gauge_value, lower_offset]
    )
    gauge_value_upper = keras.layers.Add(name="gauge_value_upper")(
        [gauge_value, upper_offset]
    )

    model = keras.Model(
        inputs=inputs,
        outputs={
            "gauge_value": gauge_value,
            "gauge_value_lower": gauge_value_lower,
            "gauge_value_upper": gauge_value_upper,
        },
        name="mobilenetv2_sota_uncertainty",
    )
    setattr(model, "_mobilenet_backbone", base_model)
    return model
