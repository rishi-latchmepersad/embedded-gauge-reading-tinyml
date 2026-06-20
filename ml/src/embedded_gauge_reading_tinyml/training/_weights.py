"""Training sub-module for embedded gauge reading.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf


def _edge_weight(value_norm: float, strength: float) -> float:
    """Return a weight that concentrates loss on the dial edges."""
    if strength <= 0.0:
        return 1.0
    distance = abs(value_norm - 0.5) * 2.0
    return 1.0 + strength * distance



def _range_aware_weight(
    value: float,
    *,
    value_min: float,
    value_max: float,
    cold_tail_fraction: float = 0.15,
    hot_tail_fraction: float = 0.15,
    oversampling_factor: float = 3.0,
) -> float:
    """Return a weight that oversamples cold/hot tails for range-aware training.

    Args:
        value: The normalized gauge value (in the calibrated range).
        value_min: Minimum value in the gauge range.
        value_max: Maximum value in the gauge range.
        cold_tail_fraction: Fraction of range at cold end to oversample (default 15%).
        hot_tail_fraction: Fraction of range at hot end to oversample (default 15%).
        oversampling_factor: How much more to weight tail samples (default 3x).

    Returns:
        A sample weight that emphasizes cold/hot tail regions.
    """
    if value_max <= value_min:
        return 1.0

    span = value_max - value_min
    cold_threshold = value_min + cold_tail_fraction * span
    hot_threshold = value_max - hot_tail_fraction * span

    if value <= cold_threshold:
        # Cold tail region
        return oversampling_factor
    elif value >= hot_threshold:
        # Hot tail region
        return oversampling_factor
    else:
        # Middle region
        return 1.0



def _compute_range_aware_weights(
    examples: list[TrainingExample],
    *,
    value_min: float,
    value_max: float,
    cold_tail_fraction: float = 0.15,
    hot_tail_fraction: float = 0.15,
    oversampling_factor: float = 3.0,
) -> np.ndarray:
    """Compute range-aware sample weights for oversampling cold/hot tails."""
    weights = np.array(
        [
            _range_aware_weight(
                example.value,
                value_min=value_min,
                value_max=value_max,
                cold_tail_fraction=cold_tail_fraction,
                hot_tail_fraction=hot_tail_fraction,
                oversampling_factor=oversampling_factor,
            )
            for example in examples
        ],
        dtype=np.float32,
    )
    return weights



def _compute_edge_weights(
    examples: list[TrainingExample],
    strength: float,
) -> np.ndarray:
    """Map normalized scalar labels into sample weights that emphasize extremes."""
    if strength <= 0.0:
        return np.ones(len(examples), dtype=np.float32)
    weights = np.array(
        [_edge_weight(example.value_norm, strength) for example in examples],
        dtype=np.float32,
    )
    return weights



def _sample_mixup_lambda(alpha: float) -> tf.Tensor:
    """Sample a MixUp interpolation coefficient from a symmetric Beta law."""
    if alpha <= 0.0:
        return tf.constant(1.0, dtype=tf.float32)
    left: tf.Tensor = tf.random.gamma([], alpha=alpha, beta=1.0, dtype=tf.float32)
    right: tf.Tensor = tf.random.gamma([], alpha=alpha, beta=1.0, dtype=tf.float32)
    return tf.cast(left / (left + right + 1e-7), tf.float32)



def _mixup_value_batch(
    images: tf.Tensor,
    targets: tf.Tensor,
    weights: tf.Tensor,
    alpha: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Apply MixUp to a batch of scalar-regression examples."""
    batch_size: tf.Tensor = tf.shape(images)[0]
    perm: tf.Tensor = tf.random.shuffle(tf.range(batch_size))
    mixed_images: tf.Tensor = tf.gather(images, perm)
    mixed_targets: tf.Tensor = tf.gather(targets, perm)
    mixed_weights: tf.Tensor = tf.gather(weights, perm)

    lam: tf.Tensor = _sample_mixup_lambda(alpha)
    lam_img: tf.Tensor = tf.reshape(lam, [1, 1, 1, 1])
    lam_vec: tf.Tensor = tf.reshape(lam, [1])

    images = lam_img * images + (1.0 - lam_img) * mixed_images
    targets = lam_vec * targets + (1.0 - lam_vec) * mixed_targets
    weights = lam_vec * weights + (1.0 - lam_vec) * mixed_weights
    return images, targets, weights


