"""
Inner-Celsius-only mask for the tip-focus geometry heatmap pipeline.

This mask ensures the model only sees the inner Celsius dial area,
excluding outer Fahrenheit labels, the lower subdial, and the lower
tape blob.  The same mask geometry is used in training, offline replay,
and firmware preprocessing so the model sees consistent pixels.

Mask geometry (on a 224x224 image space — applied AFTER crop+resize in
training, and directly on the 224x224 camera frame in firmware):
  - Keep region: a circle centered at INNER_DIAL_CENTER (112, 100)
    with radius KEEP_RADIUS_PX (62 pixels).  This includes the inner
    Celsius ring and the needle through its full sweep, but excludes
    the outer Fahrenheit labels (~radius 65+) and the corners of the
    frame where shadows/cables appear.
  - Additional lower exclusion: pixels with row >= LOWER_EXCLUDE_Y
    (150) are blanked even if inside the keep circle.  This removes
    the lower inset gauge and the dark tape/blob below the main dial.
"""

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Mask geometry constants (224x224 image space)
# ---------------------------------------------------------------------------

INNER_DIAL_CENTER_X: int = 112
"""X coordinate of the inner Celsius dial center in 224x224 pixel space."""

INNER_DIAL_CENTER_Y: int = 100
"""Y coordinate of the inner Celsius dial center in 224x224 pixel space."""

KEEP_RADIUS_PX: float = 62.0
"""Radius in pixels of the inner-Celsius keep region.

This radius (62 px at 224x224) covers:
  - The inner Celsius ring (tick marks at ~radius 40–55)
  - The Celsius label numerals (just outside the ring, ~radius 55–60)
  - The needle hub (~radius 10) and the full needle tip sweep

It excludes:
  - The outer Fahrenheit ring (starts at ~radius 65+)
  - The extreme corners of the frame (shadows, cables)
"""

LOWER_EXCLUDE_Y: int = 150
"""Row index (exclusive start) above which the image is blanked.

Values at or below this row (the lower ~1/3 of the frame) are set to
the background value, removing the lower inset gauge, the dark taped
blob, and the lower subdial even if they fall inside the keep circle.
"""

BACKGROUND_VALUE_FLOAT: float = 0.0
"""Pixel value to write into masked-out areas for float32 [0,1] images."""

BACKGROUND_VALUE_INT8: int = -128
"""Pixel value to write into masked-out areas for int8 [-128,127] images.


This matches the zero-point of the model's int8 quantisation
(scale=0.003921569, zp=-128), so masked pixels decode to 0.0 in
the float32 representation that the model was trained on.
"""


# ---------------------------------------------------------------------------
# Python mask helpers
# ---------------------------------------------------------------------------


def create_inner_celsius_mask(
    height: int = 224,
    width: int = 224,
) -> np.ndarray:
    """Create a boolean mask with ``True`` for pixels to keep.

    Args:
        height: Image height in pixels (default 224).
        width: Image width in pixels (default 224).

    Returns:
        Boolean array of shape ``(height, width)`` where ``True`` means
        the pixel passes the inner-Celsius-only gate and should NOT be
        blanked.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    dx = xs - INNER_DIAL_CENTER_X
    dy = ys - INNER_DIAL_CENTER_Y
    dist = np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)

    keep = dist <= KEEP_RADIUS_PX

    # Additionally blank the lower exclusion zone even inside the circle
    keep[ys >= LOWER_EXCLUDE_Y] = False

    return keep


def apply_inner_celsius_mask(
    image: np.ndarray,
    background_value: Optional[float] = None,
) -> np.ndarray:
    """Apply the inner-Celsius mask to a float32 [0,1] image.

    Args:
        image: Float32 array of shape ``(height, width, channels)``
            with values in [0, 1].
        background_value: Value to write into masked-out pixels.
            Defaults to ``BACKGROUND_VALUE_FLOAT`` (0.0).

    Returns:
        Masked copy of ``image``.
    """
    if background_value is None:
        background_value = BACKGROUND_VALUE_FLOAT

    mask = create_inner_celsius_mask(
        height=int(image.shape[0]),
        width=int(image.shape[1]),
    )
    result = image.copy()
    for c in range(image.shape[2]):
        result[~mask, c] = background_value
    return result


def apply_inner_celsius_mask_int8(
    image: np.ndarray,
    background_value: Optional[int] = None,
) -> np.ndarray:
    """Apply the inner-Celsius mask to an int8 [-128,127] image.

    This is the firmware representation: int8 NHWC with zero-point -128.

    Args:
        image: Int8 array of shape ``(height, width, channels)`` with
            values in [-128, 127].
        background_value: Value to write into masked-out pixels.
            Defaults to ``BACKGROUND_VALUE_INT8`` (-128).

    Returns:
        Masked copy of ``image``.
    """
    if background_value is None:
        background_value = BACKGROUND_VALUE_INT8

    mask = create_inner_celsius_mask(
        height=int(image.shape[0]),
        width=int(image.shape[1]),
    )
    result = image.copy()
    for c in range(image.shape[2]):
        result[~mask, c] = background_value
    return result
