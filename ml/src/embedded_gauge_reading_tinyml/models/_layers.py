# Custom Keras layers for gauge-reading models.
from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
import keras


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


@keras.saving.register_keras_serializable(package="embedded_gauge_reading_tinyml")
class PolarEvidenceLayer(keras.layers.Layer):
    """Convert spatial features to per-angle evidence using center-relative
    polar coordinates and differentiable soft binning.

    For each pixel in the feature map this layer:
      1. Computes its angle relative to the predicted center (atan2)
      2. Computes a learned per-pixel evidence score via small conv
      3. Soft-bins pixel scores into angular bins using Gaussian kernels
      4. Applies a radial mask to focus on the annular needle region

    The result is a per-angle evidence array mimicking the classical polar
    spoke voting, fully differentiable w.r.t. both feature values and center.
    """

    def __init__(
        self,
        num_angles: int = 180,
        sigma_angle: float = 0.04,
        radius_mean: float = 0.35,
        radius_sigma: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_angles = num_angles
        self.sigma_angle = sigma_angle
        self.radius_mean = radius_mean
        self.radius_sigma = radius_sigma
        self._bin_centers: tf.Tensor | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        C = input_shape[-1]
        self.score_conv = keras.Sequential(
            [
                keras.layers.Conv2D(max(C // 4, 8), 3, padding="same", activation="relu"),
                keras.layers.Conv2D(1, 1, padding="same", activation="linear"),
            ],
            name="polar_score_conv",
        )
        bc = np.linspace(-math.pi, math.pi, self.num_angles + 1, dtype=np.float32)[:self.num_angles]
        self._bin_centers = tf.constant(bc, dtype=tf.float32)

    def compute_output_spec(self, features, center_norm):
        return keras.KerasTensor(
            shape=(features.shape[0], self.num_angles), dtype=features.dtype
        )

    def call(
        self, features: tf.Tensor, center_norm: tf.Tensor
    ) -> tf.Tensor:
        """Run polar evidence computation.

        Args:
            features: (B, H, W, C) — spatial feature map.
            center_norm: (B, 2) — (cx, cy) in [0, 1] normalized.

        Returns:
            (B, num_angles) — per-angle evidence logits.
        """
        H = features.shape[1]
        W = features.shape[2]
        B = tf.shape(features)[0]

        # 1. Per-pixel center-relative angle and radius with explicit broadcasting
        x_grid = tf.range(W, dtype=tf.float32)[None, None, :]  # (1, 1, W)
        y_grid = tf.range(H, dtype=tf.float32)[None, :, None]  # (1, H, 1)
        cx = center_norm[:, None, None, 0] * tf.cast(W - 1, tf.float32)  # (B, 1, 1)
        cy = center_norm[:, None, None, 1] * tf.cast(H - 1, tf.float32)  # (B, 1, 1)
        dx = tf.broadcast_to(x_grid + 0.5 - cx, (B, H, W))   # (B, H, W)
        dy = tf.broadcast_to(y_grid + 0.5 - cy, (B, H, W))   # (B, H, W)
        angle = tf.math.atan2(dy, dx)
        radius = tf.sqrt(dx * dx + dy * dy)

        # 2. Per-pixel evidence scores from learned conv
        scores = self.score_conv(features)  # (B, H, W, 1)
        scores = tf.squeeze(scores, axis=-1)  # (B, H, W)

        # 3. Radial mask (annular focus)
        radial_mask = tf.exp(
            -((radius - self.radius_mean * tf.cast(W - 1, tf.float32)) ** 2)
            / (2.0 * (self.radius_sigma * tf.cast(W - 1, tf.float32)) ** 2)
        )
        radial_mask = tf.cast(radial_mask, tf.float32)

        # 4. Soft angular binning with Gaussian kernels
        diff = angle[..., None] - self._bin_centers[None, None, None, :]
        diff = tf.math.atan2(
            tf.math.sin(diff), tf.math.cos(diff)
        )  # circular
        weights = tf.exp(
            -(diff ** 2) / (2.0 * self.sigma_angle ** 2)
        )  # (B, H, W, A)

        # 5. Weighted aggregation
        weighted = scores[..., None] * weights * radial_mask[..., None]
        evidence = tf.reduce_sum(weighted, axis=[1, 2])  # (B, A)
        return evidence

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            num_angles=self.num_angles,
            sigma_angle=self.sigma_angle,
            radius_mean=self.radius_mean,
            radius_sigma=self.radius_sigma,
        )
        return config

