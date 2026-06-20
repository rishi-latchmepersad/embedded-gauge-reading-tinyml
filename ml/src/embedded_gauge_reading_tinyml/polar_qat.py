"""Quantization-aware training helpers for polar gauge models.

The deployment model stays clean and exportable.  This module adds a small
training-only wrapper that injects output quantization noise and derives the
engineering value from the quantized mask, which keeps the QAT schedule aligned
with the eventual int8 board path.
"""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow import keras

from embedded_gauge_reading_tinyml.geometry_heatmap_qat_utils import fake_quantize_01_tensor
from embedded_gauge_reading_tinyml.polar_model import PolarAngleToTemperature


def quantize_polar_mask(mask: tf.Tensor, *, noise_stddev: float = 0.0) -> tf.Tensor:
    """Add optional noise and fake-quantize a polar mask in the [0, 1] range."""

    values = tf.cast(mask, tf.float32)
    if noise_stddev > 0.0:
        values = values + tf.random.normal(tf.shape(values), stddev=noise_stddev, dtype=values.dtype)
    values = tf.clip_by_value(values, 0.0, 1.0)
    return fake_quantize_01_tensor(values)


def _angular_profile_distribution(mask: tf.Tensor, *, temperature: float = 10.0) -> tf.Tensor:
    """Collapse a polar mask into a normalized angular distribution."""

    angular_profile = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1)
    angular_profile = tf.squeeze(angular_profile, axis=-1)
    angular_profile = tf.maximum(angular_profile, 0.0)
    angular_profile = tf.nn.softmax(angular_profile * float(temperature), axis=-1)
    return angular_profile


def angle_vector_from_polar_mask(mask: tf.Tensor, *, temperature: float = 10.0) -> tf.Tensor:
    """Convert a polar needle mask into a unit [cos, sin] angle vector."""

    angular_distribution = _angular_profile_distribution(mask, temperature=temperature)
    width = tf.shape(angular_distribution)[-1]
    angles = tf.cast(tf.range(width), tf.float32) * (2.0 * math.pi / tf.cast(width, tf.float32))
    cos_weights = tf.cos(angles)[tf.newaxis, :]
    sin_weights = tf.sin(angles)[tf.newaxis, :]
    cos_component = tf.reduce_sum(angular_distribution * cos_weights, axis=-1, keepdims=True)
    sin_component = tf.reduce_sum(angular_distribution * sin_weights, axis=-1, keepdims=True)
    angle_vector = tf.concat([cos_component, sin_component], axis=-1)
    return tf.math.l2_normalize(angle_vector, axis=-1)


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def angle_vector_cosine_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Penalize disagreement between true and predicted angle unit vectors."""

    true_vector = tf.math.l2_normalize(tf.cast(y_true, tf.float32), axis=-1)
    pred_vector = tf.math.l2_normalize(tf.cast(y_pred, tf.float32), axis=-1)
    cosine_similarity = tf.reduce_sum(true_vector * pred_vector, axis=-1)
    cosine_similarity = tf.clip_by_value(cosine_similarity, -1.0, 1.0)
    return tf.reduce_mean(1.0 - cosine_similarity)


@keras.utils.register_keras_serializable(package="embedded_gauge_reading_tinyml")
def angle_vector_mae_deg(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Measure circular angular error in degrees between two unit vectors."""

    true_angle = tf.math.atan2(y_true[..., 1], y_true[..., 0])
    pred_angle = tf.math.atan2(y_pred[..., 1], y_pred[..., 0])
    wrapped_error = tf.math.floormod(pred_angle - true_angle + math.pi, 2.0 * math.pi) - math.pi
    return tf.reduce_mean(tf.abs(wrapped_error) * (180.0 / math.pi))


class PolarMaskQATTrainingModel(keras.Model):
    """Wrap a clean polar mask model with training-time fake quantization."""

    def __init__(
        self,
        *,
        base_model: keras.Model,
        output_noise_stddev: float = 0.01,
        value_min: float = -30.0,
        value_max: float = 50.0,
        min_angle_deg: float = 135.0,
        sweep_deg: float = 270.0,
        temperature: float = 10.0,
        angle_vector_temperature: float = 5.0,
        profile_head_units: int = 0,
        profile_head_dropout: float = 0.0,
        state_head_units: int = 0,
        state_head_dropout: float = 0.0,
        state_head_bins: int = 0,
        name: str = "polar_mask_qat_training_model",
    ) -> None:
        super().__init__(name=name)
        self.base_model = base_model
        self.output_noise_stddev = float(output_noise_stddev)
        self.temperature_layer = PolarAngleToTemperature(
            value_min=value_min,
            value_max=value_max,
            min_angle_deg=min_angle_deg,
            sweep_deg=sweep_deg,
            temperature=temperature,
            name="qat_gauge_value",
        )
        self.angle_vector_temperature = float(angle_vector_temperature)
        self.profile_head_units = int(profile_head_units)
        self.profile_head_dropout = float(profile_head_dropout)
        self.state_head_bins = int(state_head_bins)
        self.state_head_units = int(state_head_units) if state_head_units > 0 else max(16, self.state_head_bins * 2)
        self.state_head_dropout = float(state_head_dropout)
        self.profile_dense = (
            keras.layers.Dense(
                self.profile_head_units,
                activation="swish",
                name="qat_profile_dense",
            )
            if self.profile_head_units > 0
            else None
        )
        self.profile_dropout = (
            keras.layers.Dropout(self.profile_head_dropout, name="qat_profile_dropout")
            if self.profile_head_units > 0 and self.profile_head_dropout > 0.0
            else None
        )
        self.profile_value_sigmoid = (
            keras.layers.Dense(1, activation="sigmoid", name="qat_profile_value_sigmoid")
            if self.profile_head_units > 0
            else None
        )
        self.profile_value_rescale = (
            keras.layers.Rescaling(
                scale=value_max - value_min,
                offset=value_min,
                name="qat_profile_value",
            )
            if self.profile_head_units > 0
            else None
        )
        self.state_dense = (
            keras.layers.Dense(
                self.state_head_units,
                activation="swish",
                name="qat_state_dense",
            )
            if self.state_head_bins > 0
            else None
        )
        self.state_dropout = (
            keras.layers.Dropout(self.state_head_dropout, name="qat_state_dropout")
            if self.state_head_bins > 0 and self.state_head_dropout > 0.0
            else None
        )
        self.state_logits = (
            keras.layers.Dense(self.state_head_bins, name="qat_state_logits")
            if self.state_head_bins > 0
            else None
        )

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> dict[str, tf.Tensor]:
        """Run the clean model, then fake-quantize the mask before decoding."""

        base_outputs = self.base_model(inputs, training=training)
        if isinstance(base_outputs, dict):
            needle_mask = tf.cast(base_outputs["needle_mask"], tf.float32)
        else:
            needle_mask = tf.cast(base_outputs, tf.float32)

        noise_stddev = self.output_noise_stddev if training else 0.0
        needle_mask = quantize_polar_mask(needle_mask, noise_stddev=noise_stddev)
        gauge_value = self.temperature_layer(needle_mask)
        angular_profile = _angular_profile_distribution(
            needle_mask,
            temperature=self.angle_vector_temperature,
        )
        angle_vector = angle_vector_from_polar_mask(
            needle_mask,
            temperature=self.angle_vector_temperature,
        )
        outputs: dict[str, tf.Tensor] = {
            "needle_mask": needle_mask,
            "gauge_value": gauge_value,
            "needle_angle_vector": angle_vector,
        }

        profile_features = angular_profile
        if self.profile_dense is not None and self.profile_value_sigmoid is not None:
            # Normalize the angular profile and let a small auxiliary head learn
            # a direct scalar readout from the same geometric evidence.
            profile_features = self.profile_dense(angular_profile)
            if self.profile_dropout is not None:
                profile_features = self.profile_dropout(profile_features, training=training)
            profile_value = self.profile_value_sigmoid(profile_features)
            profile_value = self.profile_value_rescale(profile_value)
            outputs["profile_value_aux"] = profile_value

        if self.state_logits is not None and self.state_dense is not None:
            # Feed the shared angular evidence into an ordinal state head so the
            # training loss can preserve local temperature ordering.
            state_features = self.state_dense(profile_features)
            if self.state_dropout is not None:
                state_features = self.state_dropout(state_features, training=training)
            outputs["gauge_value_state"] = self.state_logits(state_features)
        elif self.state_logits is not None:
            outputs["gauge_value_state"] = self.state_logits(profile_features)

        return outputs
