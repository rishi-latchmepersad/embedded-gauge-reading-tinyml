"""Temperature-aware loss weighting for hot band focus.

This module provides loss functions that up-weight samples in the hot band (35-50°C)
to address the 46°C regression issue.
"""

from __future__ import annotations

import tensorflow as tf


def temperature_aware_loss_weight(true_temp: tf.Tensor) -> tf.Tensor:
    """
    Balanced per-sample loss weights for full-range accuracy.

    All extreme bands get boosted to ensure the model works across
    the ENTIRE gauge range (-30°C to 50°C), not just mid-band.

    Args:
        true_temp: Tensor of true temperature values

    Returns:
        Tensor of loss weights (same shape as true_temp)
    """
    # Hot band: 35-50°C gets 1.8x weight (target 46°C regression)
    hot_mask = tf.cast(tf.logical_and(true_temp >= 35.0, true_temp <= 50.0), tf.float32)
    hot_weight = hot_mask * 1.8

    # Cold band: <0°C gets 1.6x weight (symmetry with hot)
    cold_mask = tf.cast(true_temp < 0.0, tf.float32)
    cold_weight = cold_mask * 1.6

    # Low band: 0-20°C gets 1.2x weight (slight boost)
    low_mask = tf.cast(tf.logical_and(true_temp >= 0.0, true_temp < 20.0), tf.float32)
    low_weight = low_mask * 1.2

    # Mid band: 20-35°C gets 1.0x weight (baseline)
    mid_mask = tf.cast(tf.logical_and(true_temp >= 20.0, true_temp < 35.0), tf.float32)
    mid_weight = mid_mask * 1.0

    return hot_weight + cold_weight + low_weight + mid_weight


def make_weighted_mse_loss():
    """Create a temperature-weighted MSE loss function."""

    def weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute MSE with temperature-aware weighting."""
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        # Compute per-sample squared error
        squared_error = tf.square(y_true_f - y_pred_f)

        # Get temperature-aware weights
        weights = temperature_aware_loss_weight(y_true_f)

        # Apply weights and compute mean
        weighted_error = squared_error * weights
        return tf.reduce_mean(weighted_error)

    weighted_mse.__name__ = "weighted_mse"
    return weighted_mse


def make_weighted_mae_loss():
    """Create a temperature-weighted MAE loss function."""

    def weighted_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute MAE with temperature-aware weighting."""
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        # Compute per-sample absolute error
        abs_error = tf.abs(y_true_f - y_pred_f)

        # Get temperature-aware weights
        weights = temperature_aware_loss_weight(y_true_f)

        # Apply weights and compute mean
        weighted_error = abs_error * weights
        return tf.reduce_mean(weighted_error)

    weighted_mae.__name__ = "weighted_mae"
    return weighted_mae


def make_weighted_scalar_regression_loss(
    monotonic_pair_strength: float = 0.0,
    monotonic_pair_margin: float = 0.0,
    interpolation_pair_strength: float = 0.0,
    interpolation_pair_scale: float = 1.0,
) -> callable:
    """
    Create a temperature-weighted scalar regression loss with optional pair constraints.

    This is a modified version of _make_scalar_regression_loss that adds
    temperature-aware weighting to prioritize the hot band.
    """

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute weighted regression loss."""
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        # Base MSE with temperature weighting
        squared_error = tf.square(y_true_f - y_pred_f)
        weights = temperature_aware_loss_weight(y_true_f)
        weighted_mse = tf.reduce_mean(squared_error * weights)

        total_loss = weighted_mse

        # Note: Pair constraints would need to be adapted for weighted samples
        # For now, we focus on the base weighted MSE

        return total_loss

    loss_fn.__name__ = "weighted_scalar_regression_loss"
    return loss_fn


# Test the weighting function
if __name__ == "__main__":
    import numpy as np

    # Test temperature ranges
    test_temps = tf.constant(
        [-30.0, -15.0, 0.0, 20.0, 35.0, 46.0, 50.0], dtype=tf.float32
    )
    weights = temperature_aware_loss_weight(test_temps)

    print("Temperature-aware loss weights:")
    print("-" * 40)
    for temp, weight in zip(test_temps.numpy(), weights.numpy()):
        print(f"  {temp:6.1f}°C -> {weight:.1f}x weight")

    # Test loss functions
    print("\nTesting loss functions...")
    y_true = tf.constant([46.0, 20.0, -30.0], dtype=tf.float32)
    y_pred = tf.constant(
        [36.3, 20.0, -30.0], dtype=tf.float32
    )  # 46°C under-reads to 36.3

    mse_fn = make_weighted_mse_loss()
    mae_fn = make_weighted_mae_loss()

    mse = mse_fn(y_true, y_pred)
    mae = mae_fn(y_true, y_pred)

    print(f"Weighted MSE: {mse.numpy():.4f}")
    print(f"Weighted MAE: {mae.numpy():.4f}")

    # Compare with standard MSE
    standard_mse = tf.reduce_mean(tf.square(y_true - y_pred))
    print(f"Standard MSE: {standard_mse.numpy():.4f}")
    print(f"Ratio (weighted/std): {mse.numpy() / standard_mse.numpy():.2f}x")
