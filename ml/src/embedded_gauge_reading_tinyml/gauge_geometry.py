"""
Gauge geometry utilities for the inner Celsius dial reader.

This module provides deterministic geometry calculations for converting
between needle angles and temperature readings on the inner Celsius dial.

The inner Celsius dial has:
- Cold end (-30C) at approximately 135 degrees (bottom-left)
- Hot end (+50C) at approximately 45 degrees (top-right)
- Total sweep of approximately 270 degrees counter-clockwise

Image coordinate system:
- Origin (0,0) is at top-left
- X increases to the right
- Y increases downward

This affects angle calculations: standard mathematical angles go counter-clockwise
from the positive X axis, but image Y is inverted. We handle this by computing
angles in image space consistently.
"""

import math
from typing import Any, Optional

import numpy as np


def angle_degrees_from_center_to_tip(
    center_x_pixels: float,
    center_y_pixels: float,
    tip_x_pixels: float,
    tip_y_pixels: float,
) -> float:
    """
    Calculate the angle from dial center to needle tip in degrees.

    This computes the angle of the needle relative to the positive X axis,
    measured counter-clockwise in image space. The result is in the range
    [0, 360) degrees.

    Why image-space angle wrapping matters:
    - In image coordinates, Y increases downward (unlike standard math coordinates)
    - atan2 returns angles in [-pi, pi] radians, which we convert to [0, 360) degrees
    - Consistent angle convention is critical for training and inference to match

    Args:
        center_x_pixels: X coordinate of the dial center in source image pixels
        center_y_pixels: Y coordinate of the dial center in source image pixels
        tip_x_pixels: X coordinate of the needle tip in source image pixels
        tip_y_pixels: Y coordinate of the needle tip in source image pixels

    Returns:
        Angle in degrees from center to tip, in range [0, 360)
    """
    # Compute vector from center to tip
    dx = tip_x_pixels - center_x_pixels
    dy = tip_y_pixels - center_y_pixels

    # atan2 gives angle in radians in range [-pi, pi]
    # In image space: positive X is right, positive Y is down
    # atan2(y, x) gives angle from positive X axis, positive toward positive Y
    angle_radians = math.atan2(dy, dx)

    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)

    # Normalize to [0, 360) range
    # atan2 returns [-180, 180], so we add 360 to negative values
    if angle_degrees < 0:
        angle_degrees += 360.0

    return angle_degrees


def celsius_from_inner_dial_angle_degrees(
    angle_degrees: float,
    cold_angle_degrees: float = 135.0,
    sweep_degrees: float = 270.0,
    minimum_celsius: float = -30.0,
    maximum_celsius: float = 50.0,
) -> float:
    """
    Convert a needle angle to temperature using the inner Celsius dial calibration.

    The inner Celsius dial spans from -30C to +50C over a 270-degree arc.
    The cold end (-30C) is at cold_angle_degrees (default 135, bottom-left).
    The scale increases counter-clockwise.

    Angle mapping (default calibration):
    - 135 -> -30C (cold end, bottom-left)
    - 225 ->   0C (middle, top)
    - 45  -> +50C (hot end, top-right, wraps through 0)

    Why circular angle handling matters:
    - The dial wraps through 0/360, so 45 is "after" 135 in counter-clockwise order
    - We must handle the wraparound correctly when computing position along the arc
    - Simple linear interpolation fails at the wraparound point

    Args:
        angle_degrees: The needle angle in degrees [0, 360)
        cold_angle_degrees: Angle at the cold end (-30C), default 135
        sweep_degrees: Total angular sweep of the dial, default 270
        minimum_celsius: Temperature at cold end, default -30C
        maximum_celsius: Temperature at hot end, default +50C

    Returns:
        Temperature in degrees Celsius
    """
    # Normalize input angle to [0, 360)
    angle_normalized = angle_degrees % 360.0

    # Compute the angular distance from cold_end to the needle angle,
    # moving counter-clockwise along the dial sweep.
    # This requires careful handling of the 0/360 wraparound.

    # Step 1: Compute raw difference (needle - cold_end)
    raw_diff = angle_normalized - cold_angle_degrees

    # Step 2: Normalize to [0, 360) to get counter-clockwise distance
    # If raw_diff is negative, the needle is "before" cold_end in standard angle order,
    # but since we measure counter-clockwise from cold_end, we add 360.
    if raw_diff < 0:
        raw_diff += 360.0

    # Step 3: The fraction of the sweep that the needle has traversed
    # The dial only covers sweep_degrees (270), not the full 360
    sweep_fraction = raw_diff / sweep_degrees

    # Step 4: Map fraction to temperature range
    temperature_range = maximum_celsius - minimum_celsius
    celsius = minimum_celsius + sweep_fraction * temperature_range

    return celsius


def circular_angle_error_degrees(
    predicted_angle_degrees: float,
    true_angle_degrees: float,
) -> float:
    """
    Compute the circular error between two angles.

    This computes the shortest angular distance between two angles,
    correctly handling the 0/360 wraparound.

    Examples:
    - Error between 10 and 20 is 10
    - Error between 359 and 1 is 2 (not 358)
    - Error between 0 and 180 is 180

    Why circular error matters:
    - Angles are periodic: 0 = 360
    - Standard absolute difference fails near the wraparound point
    - For gauge needles, 359 and 1 represent nearly identical positions
    - Training losses and evaluation metrics must respect this circularity

    Args:
        predicted_angle_degrees: The predicted needle angle in degrees
        true_angle_degrees: The ground truth needle angle in degrees

    Returns:
        Absolute angular error in degrees, in range [0, 180]
    """
    # Normalize both angles to [0, 360)
    pred_norm = predicted_angle_degrees % 360.0
    true_norm = true_angle_degrees % 360.0

    # Compute raw difference
    diff = abs(pred_norm - true_norm)

    # Handle wraparound: if diff > 180, the shorter path goes the other way
    # For example, |359 - 1| = 358, but the actual angular distance is 2
    if diff > 180.0:
        diff = 360.0 - diff

    return diff


def angle_degrees_to_radians(angle_degrees: float) -> float:
    """
    Convert angle from degrees to radians.

    Simple wrapper around math.radians for consistency with module naming.

    Args:
        angle_degrees: Angle in degrees

    Returns:
        Angle in radians
    """
    return math.radians(angle_degrees)


def angle_radians_to_degrees(angle_radians: float) -> float:
    """
    Convert angle from radians to degrees.

    Simple wrapper around math.degrees for consistency with module naming.

    Args:
        angle_radians: Angle in radians

    Returns:
        Angle in degrees
    """
    return math.degrees(angle_radians)


def temperature_to_inner_dial_angle_degrees(
    celsius: float,
    cold_angle_degrees: float = 135.0,
    sweep_degrees: float = 270.0,
    minimum_celsius: float = -30.0,
    maximum_celsius: float = 50.0,
) -> float:
    """Convert a temperature to the inner dial needle angle in degrees.

    Reverse of celsius_from_inner_dial_angle_degrees. Maps a temperature
    to the needle angle that a correctly-calibrated gauge would show.

    Args:
        celsius: Temperature in Celsius
        cold_angle_degrees: Angle at the cold end (default 135)
        sweep_degrees: Total angular sweep (default 270)
        minimum_celsius: Temperature at cold end (default -30)
        maximum_celsius: Temperature at hot end (default 50)

    Returns:
        Needle angle in degrees [0, 360)
    """
    temperature_range = maximum_celsius - minimum_celsius
    if temperature_range <= 0.0:
        raise ValueError("maximum_celsius must be > minimum_celsius")
    fraction = (celsius - minimum_celsius) / temperature_range
    fraction = min(max(fraction, 0.0), 1.0)
    angle = cold_angle_degrees + fraction * sweep_degrees
    return angle % 360.0


def angle_logits_to_temperature(
    logits: np.ndarray,
    gauge_spec: Any,
    *,
    num_bins: int = 36,
    decode_topk: int = 3,
) -> float:
    """Decode 36-bin angle logits to temperature via circular expectation.

    Steps:
    1. Softmax over logits → probabilities per angle bin
    2. Circular expectation (Von Mises) → mean angle in radians
    3. Mean angle → fraction of sweep → temperature via gauge_spec

    Args:
        logits: Raw logits array of shape (num_bins,) or (1, num_bins)
        gauge_spec: GaugeSpec with min_angle_rad, sweep_rad, min_value, max_value
        num_bins: Number of angle bins (default 36 for 10° resolution)
        decode_topk: Number of top bins to use for expectation (default 3)

    Returns:
        Temperature in gauge_spec units
    """
    flat = np.asarray(logits, dtype=np.float32).reshape(-1)
    if flat.size < num_bins:
        flat = np.pad(flat, (0, num_bins - flat.size))
    flat = flat[:num_bins]

    # Softmax
    flat -= flat.max()
    exp_vals = np.exp(flat)
    probs = exp_vals / (exp_vals.sum() + 1e-8)

    # Top-k filtering for robustness
    topk = min(decode_topk, num_bins)
    top_indices = np.argsort(probs)[-topk:]
    mask = np.zeros_like(probs)
    mask[top_indices] = 1.0
    probs = probs * mask
    probs /= probs.sum() + 1e-8

    # Circular expectation
    bin_angles = np.linspace(0.0, 2.0 * math.pi, num_bins, endpoint=False, dtype=np.float32)
    sin_sum = float(np.sum(probs * np.sin(bin_angles)))
    cos_sum = float(np.sum(probs * np.cos(bin_angles)))
    mean_angle = math.atan2(sin_sum, cos_sum)
    if mean_angle < 0.0:
        mean_angle += 2.0 * math.pi

    # Map angle to temperature via gauge_spec
    angle_deg = math.degrees(mean_angle)
    cold_angle_deg = math.degrees(gauge_spec.min_angle_rad)
    sweep_deg = math.degrees(gauge_spec.sweep_rad)
    return celsius_from_inner_dial_angle_degrees(
        angle_deg,
        cold_angle_degrees=cold_angle_deg,
        sweep_degrees=sweep_deg,
        minimum_celsius=gauge_spec.min_value,
        maximum_celsius=gauge_spec.max_value,
    )


def temperature_to_angle_bin_distribution(
    celsius: float,
    gauge_spec: Any,
    *,
    num_bins: int = 36,
    sigma_bins: float = 1.5,
) -> np.ndarray:
    """Convert a temperature label to a soft angle-bin target distribution.

    Creates a circular Gaussian centered on the bin corresponding to the
    temperature's needle angle. Used as the training target for the
    angle-vote model's categorical cross-entropy loss.

    Args:
        celsius: Temperature in Celsius
        gauge_spec: GaugeSpec for angle-temperature mapping
        num_bins: Number of angle bins (default 36)
        sigma_bins: Gaussian width in bins (default 1.5)

    Returns:
        Float32 array of shape (num_bins,) summing to 1.0
    """
    angle = temperature_to_inner_dial_angle_degrees(
        celsius,
        cold_angle_degrees=math.degrees(gauge_spec.min_angle_rad),
        sweep_degrees=math.degrees(gauge_spec.sweep_rad),
        minimum_celsius=gauge_spec.min_value,
        maximum_celsius=gauge_spec.max_value,
    )
    bin_width = 360.0 / num_bins
    center_bin = (angle / bin_width) % num_bins
    indices = np.arange(num_bins, dtype=np.float32)
    dist = np.exp(
        -0.5 * ((indices - np.float32(center_bin)) / np.float32(sigma_bins)) ** 2
    )
    # Circular wrap: account for bins at the 0/360 boundary
    dist_wrapped = dist.copy()
    dist_wrapped[:num_bins // 2] += np.exp(
        -0.5 * ((indices[:num_bins // 2] + num_bins - np.float32(center_bin)) / np.float32(sigma_bins)) ** 2
    )
    dist_wrapped[-num_bins // 2:] += np.exp(
        -0.5 * ((indices[-num_bins // 2:] - num_bins - np.float32(center_bin)) / np.float32(sigma_bins)) ** 2
    )
    total = float(np.sum(dist_wrapped))
    if total > 0.0:
        dist_wrapped /= np.float32(total)
    return dist_wrapped.astype(np.float32)


def temperature_to_angle_sincos(
    celsius: float,
    gauge_spec: Any,
) -> tuple[float, float]:
    """Convert temperature to needle-direction unit vector (sin, cos).

    The needle angle is computed from temperature via the gauge spec, then
    converted to (sin(angle), cos(angle)). Used as the regression target
    for angle sin/cos models.

    Args:
        celsius: Temperature in Celsius
        gauge_spec: GaugeSpec with angle→temperature mapping

    Returns:
        Tuple of (sin_angle, cos_angle) representing the unit-length
        needle direction vector.
    """
    angle_deg = temperature_to_inner_dial_angle_degrees(
        celsius,
        cold_angle_degrees=math.degrees(gauge_spec.min_angle_rad),
        sweep_degrees=math.degrees(gauge_spec.sweep_rad),
        minimum_celsius=gauge_spec.min_value,
        maximum_celsius=gauge_spec.max_value,
    )
    angle_rad = math.radians(angle_deg)
    return (math.sin(angle_rad), math.cos(angle_rad))


def angle_sincos_to_temperature(
    sin_val: float,
    cos_val: float,
    gauge_spec: Any,
) -> float:
    """Decode (sin, cos) predictions to temperature via gauge spec.

    Uses atan2 to recover the angle, then maps to temperature.

    Args:
        sin_val: Predicted sin(angle)
        cos_val: Predicted cos(angle)
        gauge_spec: GaugeSpec for angle→temperature mapping

    Returns:
        Temperature in gauge_spec units (Celsius)
    """
    angle_rad = math.atan2(sin_val, cos_val)
    if angle_rad < 0.0:
        angle_rad += 2.0 * math.pi
    angle_deg = math.degrees(angle_rad)
    cold_angle_deg = math.degrees(gauge_spec.min_angle_rad)
    sweep_deg = math.degrees(gauge_spec.sweep_rad)
    return celsius_from_inner_dial_angle_degrees(
        angle_deg,
        cold_angle_degrees=cold_angle_deg,
        sweep_degrees=sweep_deg,
        minimum_celsius=gauge_spec.min_value,
        maximum_celsius=gauge_spec.max_value,
    )
