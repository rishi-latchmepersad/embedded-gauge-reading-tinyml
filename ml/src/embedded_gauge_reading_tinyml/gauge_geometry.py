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
from typing import Optional


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
