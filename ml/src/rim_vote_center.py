"""Python port of the rim-vote centre detector from the firmware.

The C reference is in ``app_baseline_runtime.c``:
  - ``AppBaselineRuntime_EstimateDialCenterFromRimVotes`` (coarse-to-fine search)
  - ``AppBaselineRuntime_ScoreDialCenterCandidate`` (per-candidate scoring)
  - ``AppBaselineRuntime_ReadEdgeMagnitude``      (Sobel edge + radial alignment)

Constants match the firmware values at line 60-87.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Firmware constants (app_baseline_runtime.c lines 60-87)
SATURATION_THRESHOLD = 235
SCAN_BORDER_PIXELS = 8
MIN_RADIUS_PIXELS = 16
COARSE_STEP_PIXELS = 8
FINE_STEP_PIXELS = 4
SAMPLE_STEP_PIXELS = 4
RIM_MIN_FRACTION = 0.84
RIM_MAX_FRACTION = 1.04

# Subdial mask (app_baseline_runtime.c lines 67-69)
SUBDIAL_X_FRACTION = 0.35
SUBDIAL_Y_MIN_FRACTION = 0.10
SUBDIAL_Y_MAX_FRACTION = 0.58


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _read_luma(raw: np.ndarray, width_px: int, x: int, y: int) -> float:
    """Read luminance (Y) from a YUV422 packed buffer at pixel (x, y).

    YUV422 byte layout for an even/odd pixel pair at column ``x``:
        offset = (y * stride) + ((x // 2) * 4)
        U = frame[offset + 1]
        Y_even = frame[offset]
        Y_odd  = frame[offset + 2]
        V = frame[offset + 3]
    """
    row_stride = width_px * 2
    pair_offset = (y * row_stride) + ((x & ~1) * 2)
    y_offset = pair_offset + (2 if (x & 1) else 0)
    return float(raw[y_offset])


def _read_edge_magnitude(
    raw: np.ndarray, width_px: int, height_px: int, x: int, y: int
) -> Tuple[float, float, float, float]:
    """Sobel-like 3x3 edge magnitude plus gradient at (x, y).

    Returns ``(magnitude, grad_x, grad_y, background_luma)``.
    Returns zero magnitude at the image border.
    """
    if x < 1 or y < 1 or (x + 1) >= width_px or (y + 1) >= height_px:
        return (0.0, 0.0, 0.0, 0.0)

    tl = _read_luma(raw, width_px, x - 1, y - 1)
    tc = _read_luma(raw, width_px, x, y - 1)
    tr = _read_luma(raw, width_px, x + 1, y - 1)
    ml = _read_luma(raw, width_px, x - 1, y)
    mr = _read_luma(raw, width_px, x + 1, y)
    bl = _read_luma(raw, width_px, x - 1, y + 1)
    bc = _read_luma(raw, width_px, x, y + 1)
    br = _read_luma(raw, width_px, x + 1, y + 1)

    gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)
    gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)
    bg = (tl + tc + tr + ml + mr + bl + bc + br) / 8.0
    mag = math.sqrt(gx * gx + gy * gy)
    return (mag, gx, gy, bg)


def _is_in_subdial_mask(cx: int, cy: int, x: int, y: int, radius_px: float) -> bool:
    """Check if (x, y) falls inside the subdial clutter mask.

    The subdial is a smaller concentric dial in the bottom-right quadrant.
    """
    dx = float(abs(x - cx))
    dy = float(abs(y - cy))
    return (
        dx < SUBDIAL_X_FRACTION * radius_px
        and float(y) > float(cy) + SUBDIAL_Y_MIN_FRACTION * radius_px
        and float(y) < float(cy) + SUBDIAL_Y_MAX_FRACTION * radius_px
        and dy > SUBDIAL_Y_MIN_FRACTION * radius_px
    )


def score_candidate(
    raw: np.ndarray,
    width_px: int,
    height_px: int,
    scan_x_min: int,
    scan_y_min: int,
    scan_x_max: int,
    scan_y_max: int,
    dial_radius_px: float,
    cx: int,
    cy: int,
) -> float:
    """Score a candidate centre ``(cx, cy)`` — higher is better.

    Matches ``AppBaselineRuntime_ScoreDialCenterCandidate``.
    """
    rim_radius_min = dial_radius_px * RIM_MIN_FRACTION
    rim_radius_max = dial_radius_px * RIM_MAX_FRACTION
    crop_center_x = (scan_x_min + scan_x_max) / 2.0
    crop_center_y = (scan_y_min + scan_y_max) / 2.0
    crop_half_diag = math.hypot(
        (scan_x_max - scan_x_min) / 2.0, (scan_y_max - scan_y_min) / 2.0
    )
    score = 0.0
    sample_count = 0

    for y in range(scan_y_min + 1, scan_y_max - 1, SAMPLE_STEP_PIXELS):
        for x in range(scan_x_min + 1, scan_x_max - 1, SAMPLE_STEP_PIXELS):
            dx = float(x) - float(cx)
            dy = float(y) - float(cy)
            radius = math.hypot(dx, dy)
            if radius < rim_radius_min or radius > rim_radius_max:
                continue

            luma = _read_luma(raw, width_px, x, y)
            if luma > SATURATION_THRESHOLD or _is_in_subdial_mask(
                cx, cy, x, y, dial_radius_px
            ):
                continue

            edge_mag, grad_x, grad_y, _bg = _read_edge_magnitude(
                raw, width_px, height_px, x, y
            )
            safe_mag = edge_mag if edge_mag > 1.0 else 1.0
            radial_x = dx / radius
            radial_y = dy / radius
            norm_gx = grad_x / safe_mag
            norm_gy = grad_y / safe_mag
            radial_alignment = abs(norm_gx * radial_x + norm_gy * radial_y)
            rim_dist = abs(radius - dial_radius_px) / (dial_radius_px + 1e-6)
            rim_bias = 1.0 - _clamp(rim_dist, 0.0, 1.0)
            rim_weight = rim_bias * rim_bias
            alignment_weight = radial_alignment * radial_alignment
            vote = edge_mag * alignment_weight * rim_weight
            if vote > 0.0:
                score += vote
                sample_count += 1

    if sample_count == 0:
        return 0.0

    center_dist = math.hypot(float(cx) - crop_center_x, float(cy) - crop_center_y)
    center_prior = _clamp(
        1.0 - 0.25 * (center_dist / (crop_half_diag + 1e-6)), 0.20, 1.0
    )
    return (score / float(sample_count)) * center_prior


def estimate_dial_center(
    yuv_bytes: bytes,
    frame_width_px: int,
    frame_height_px: int,
    dial_radius_px: float | None = None,
    scan_x_min: int | None = None,
    scan_y_min: int | None = None,
    scan_x_max: int | None = None,
    scan_y_max: int | None = None,
) -> Tuple[int, int, float]:
    """Full coarse-to-fine rim-vote centre search.

    Parameters
    ----------
    yuv_bytes:
        Raw YUV422 packed buffer (``frame_width_px * frame_height_px * 2`` bytes).
    frame_width_px, frame_height_px:
        Dimensions of the YUV422 frame.
    dial_radius_px:
        Expected radius of the gauge rim in pixels.
        Defaults to ``0.3076 * frame_height_px`` (the inner Celsius dial ratio).
    scan_* :
        Bounding box for the centre search (defaults to the full image minus
        ``SCAN_BORDER_PIXELS`` on each side).

    Returns
    -------
    ``(best_cx, best_cy, best_quality)``.
    """
    raw = np.frombuffer(yuv_bytes, dtype=np.uint8)

    if dial_radius_px is None:
        dial_radius_px = 0.3076 * float(frame_height_px)
    if scan_x_min is None:
        scan_x_min = SCAN_BORDER_PIXELS
    if scan_y_min is None:
        scan_y_min = SCAN_BORDER_PIXELS
    if scan_x_max is None:
        scan_x_max = frame_width_px - SCAN_BORDER_PIXELS
    if scan_y_max is None:
        scan_y_max = frame_height_px - SCAN_BORDER_PIXELS

    min_cx = scan_x_min + SCAN_BORDER_PIXELS
    min_cy = scan_y_min + SCAN_BORDER_PIXELS
    max_cx = scan_x_max - SCAN_BORDER_PIXELS - 1
    max_cy = scan_y_max - SCAN_BORDER_PIXELS - 1

    if min_cx >= max_cx or min_cy >= max_cy or dial_radius_px < float(MIN_RADIUS_PIXELS):
        return (0, 0, 0.0)

    # Coarse search
    best_cx, best_cy = 0, 0
    best_quality = -1.0
    found = False
    for cy in range(min_cy, max_cy + 1, COARSE_STEP_PIXELS):
        for cx in range(min_cx, max_cx + 1, COARSE_STEP_PIXELS):
            q = score_candidate(
                raw, frame_width_px, frame_height_px,
                scan_x_min, scan_y_min, scan_x_max, scan_y_max,
                dial_radius_px, cx, cy,
            )
            if q > best_quality:
                best_quality = q
                best_cx, best_cy = cx, cy
                found = True

    if not found:
        return (0, 0, 0.0)

    # Fine search around the best coarse centre
    fine_radius = int(COARSE_STEP_PIXELS)
    fine_min_x = max(int(min_cx), best_cx - fine_radius)
    fine_max_x = min(int(max_cx), best_cx + fine_radius)
    fine_min_y = max(int(min_cy), best_cy - fine_radius)
    fine_max_y = min(int(max_cy), best_cy + fine_radius)

    for cy in range(fine_min_y, fine_max_y + 1, FINE_STEP_PIXELS):
        for cx in range(fine_min_x, fine_max_x + 1, FINE_STEP_PIXELS):
            q = score_candidate(
                raw, frame_width_px, frame_height_px,
                scan_x_min, scan_y_min, scan_x_max, scan_y_max,
                dial_radius_px, cx, cy,
            )
            if q > best_quality:
                best_quality = q
                best_cx, best_cy = cx, cy

    return (best_cx, best_cy, best_quality)


def yuv422_to_rgb(yuv_bytes: bytes, width_px: int, height_px: int) -> np.ndarray:
    """Convert a YUV422 packed buffer to an RGB uint8 numpy array.

    BT.601 conversion with integer arithmetic matching the firmware.
    Returns shape ``(height, width, 3)``.
    """
    raw = np.frombuffer(yuv_bytes, dtype=np.uint8)
    rgb = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    for y in range(height_px):
        for x in range(width_px):
            luma = _read_luma(raw, width_px, x, y)
            # U/V are shared for even/odd pixel pairs
            row_stride = width_px * 2
            pair_offset = (y * row_stride) + ((x & ~1) * 2)
            u = int(raw[pair_offset + 1])
            v = int(raw[pair_offset + 3])

            r = int(luma) + ((v - 128) * 1436) // 1024
            g = int(luma) - ((u - 128) * 352 + (v - 128) * 731) // 1024
            b = int(luma) + ((u - 128) * 1814) // 1024
            r = _clamp(r, 0, 255)
            g = _clamp(g, 0, 255)
            b = _clamp(b, 0, 255)
            rgb[y, x] = (r, g, b)

    return rgb
