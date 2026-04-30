"""Simple line-segment needle detector for analog gauge reading.

This detector uses Canny edge detection + HoughLinesP to find the needle
as a line segment, eliminating the inversion ambiguity of histogram voting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np


@dataclass(frozen=True)
class NeedleDetection:
    """Needle detection result."""

    unit_dx: float
    unit_dy: float
    confidence: float
    tip_x: float
    tip_y: float
    tail_x: float
    tail_y: float


def detect_needle_line_segment(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    min_angle_rad: float = math.radians(135.0),
    sweep_rad: float = math.radians(270.0),
) -> NeedleDetection | None:
    """Detect needle using Canny + HoughLinesP.

    The needle is the longest dark line segment that passes near the center
    and has one endpoint near the hub and the other near the rim.
    """
    if dial_radius_px <= 1.0:
        return None

    center_x, center_y = center_xy
    h_img, w_img = image_bgr.shape[:2]

    # Convert to grayscale and enhance contrast.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Canny edge detection.
    edges = cv2.Canny(blurred, 40, 120, apertureSize=3, L2gradient=True)

    # HoughLinesP to find line segments.
    min_line_length = max(18, int(round(0.30 * dial_radius_px)))
    max_line_gap = max(4, int(round(0.08 * dial_radius_px)))
    threshold = max(14, int(round(0.10 * dial_radius_px)))

    lines = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return None

    best_score = float("-inf")
    best_detection: NeedleDetection | None = None

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len < min_line_length:
            continue

        # Distance from center to the line segment.
        center_dist = _point_to_segment_distance(center_x, center_y, x1, y1, x2, y2)
        if center_dist > 0.30 * dial_radius_px:
            continue  # Must pass near center.

        # Distance from center to each endpoint.
        d1 = math.hypot(x1 - center_x, y1 - center_y)
        d2 = math.hypot(x2 - center_x, y2 - center_y)
        near = min(d1, d2)
        far = max(d1, d2)

        # One end near hub, other near rim.
        if near > 0.30 * dial_radius_px or far < 0.50 * dial_radius_px:
            continue

        # Determine tip (far from center) and tail (near center).
        if d1 > d2:
            tip_x, tip_y = x1, y1
            tail_x, tail_y = x2, y2
        else:
            tip_x, tip_y = x2, y2
            tail_x, tail_y = x1, y1

        # Angle from center to tip.
        angle_rad = math.atan2(tip_y - center_y, tip_x - center_x)

        # Check if angle is within gauge sweep.
        shifted = (angle_rad - min_angle_rad) % (2.0 * math.pi)
        if shifted > sweep_rad + math.radians(10.0):
            continue  # Outside sweep.

        # Score by darkness contrast along the line.
        contrast, dark_fraction = _sample_line_darkness(
            image_bgr, tail_x, tail_y, tip_x, tip_y, center_xy, dial_radius_px
        )
        if contrast <= 0.0:
            continue

        # Score = length * contrast * radial_alignment.
        radiality = 1.0 - min(center_dist / max(0.30 * dial_radius_px, 1.0), 1.0)
        reach = min(far / max(0.95 * dial_radius_px, 1.0), 1.0)
        score = seg_len * contrast * (0.3 + 0.7 * radiality) * (0.3 + 0.7 * reach)

        if score > best_score:
            best_score = score
            unit_len = max(far, 1e-6)
            best_detection = NeedleDetection(
                unit_dx=(tip_x - center_x) / unit_len,
                unit_dy=(tip_y - center_y) / unit_len,
                confidence=score,
                tip_x=tip_x,
                tip_y=tip_y,
                tail_x=tail_x,
                tail_y=tail_y,
            )

    return best_detection


def _point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Shortest distance from point to finite line segment."""
    seg_dx = x2 - x1
    seg_dy = y2 - y1
    seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len_sq <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq
    t = min(max(t, 0.0), 1.0)
    proj_x = x1 + t * seg_dx
    proj_y = y1 + t * seg_dy
    return math.hypot(px - proj_x, py - proj_y)


def _sample_line_darkness(
    image_bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    center_xy: tuple[float, float],
    dial_radius_px: float,
) -> tuple[float, float]:
    """Sample darkness contrast along a line segment."""
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    seg_dx = x2 - x1
    seg_dy = y2 - y1
    seg_len = math.hypot(seg_dx, seg_dy)
    if seg_len <= 1e-6:
        return 0.0, 0.0

    unit_x = seg_dx / seg_len
    unit_y = seg_dy / seg_len
    perp_x = -unit_y
    perp_y = unit_x

    contrasts = []
    dark_hits = 0
    samples = 0

    # Sample middle portion of line (avoid hub and rim edges).
    for fraction in np.linspace(0.35, 0.75, 7, dtype=np.float32):
        samples += 1
        sx = x1 + fraction * seg_dx
        sy = y1 + fraction * seg_dy
        ix = int(round(min(max(sx, 0.0), gray.shape[1] - 1.0)))
        iy = int(round(min(max(sy, 0.0), gray.shape[0] - 1.0)))
        line_px = float(gray[iy, ix])

        # Sample neighbors perpendicular to line.
        neighbor_values = []
        for offset_px in (2.0, 4.0):
            for direction in (-1.0, 1.0):
                nx = sx + direction * offset_px * perp_x
                ny = sy + direction * offset_px * perp_y
                nix = int(round(min(max(nx, 0.0), gray.shape[1] - 1.0)))
                niy = int(round(min(max(ny, 0.0), gray.shape[0] - 1.0)))
                neighbor_values.append(float(gray[niy, nix]))

        if not neighbor_values:
            continue

        local_mean = float(np.mean(neighbor_values))
        contrast = (local_mean - line_px) / 255.0
        contrasts.append(contrast)
        if line_px + 5.0 < local_mean:
            dark_hits += 1

    if not contrasts:
        return 0.0, 0.0

    contrast_mean = float(np.mean(contrasts))
    dark_fraction = dark_hits / samples if samples > 0 else 0.0
    return contrast_mean, dark_fraction


def needle_to_temperature(
    detection: NeedleDetection,
    *,
    min_value: float = -30.0,
    max_value: float = 50.0,
    min_angle_rad: float = math.radians(135.0),
    sweep_rad: float = math.radians(270.0),
) -> float:
    """Convert needle direction to temperature."""
    angle_rad = math.atan2(detection.unit_dy, detection.unit_dx)
    shifted = (angle_rad - min_angle_rad) % (2.0 * math.pi)
    fraction = min(max(shifted / sweep_rad, 0.0), 1.0)
    return min_value + fraction * (max_value - min_value)


def main() -> None:
    """Test the line-segment detector on hard cases."""
    import sys

    # Hard cases from ai-memory.
    test_cases = [
        ("capture_0073.png", 46.0),
        ("capture_2026-04-03_08-20-49.png", 45.0),
        ("capture_p5c.png", 5.0),
        ("capture_p20c_preview.png", 20.0),
        ("capture_2026-04-24_22-24-04.png", 10.0),
        ("capture_2026-04-24_22-30-21.png", 10.0),
        ("capture_0001.png", 0.0),
        ("capture_0002.png", 0.0),
    ]

    # Get project root relative to this script.
    script_dir = Path(__file__).parent.parent.parent.parent
    img_dir = script_dir / "captured_images"

    # Fixed geometry from board prior (simplified).
    center_xy = (120.0, 120.0)
    dial_radius_px = 80.0

    print("Line-Segment Needle Detector Results")
    print("=" * 60)
    print(f"{'Image':<40} {'True':>6} {'Pred':>6} {'Error':>6} {'Conf':>8}")
    print("-" * 60)

    total_error = 0.0
    count = 0

    for filename, true_temp in test_cases:
        img_path = img_dir / filename
        if not img_path.exists():
            print(f"{filename:<40} NOT FOUND")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"{filename:<40} FAILED TO LOAD")
            continue

        detection = detect_needle_line_segment(
            image,
            center_xy=center_xy,
            dial_radius_px=dial_radius_px,
        )

        if detection is None:
            print(f"{filename:<40} {true_temp:>6.1f} {'FAIL':>6} {'-':>6} {'-':>8}")
            continue

        pred_temp = needle_to_temperature(detection)
        error = abs(pred_temp - true_temp)
        total_error += error
        count += 1

        print(
            f"{filename:<40} {true_temp:>6.1f} {pred_temp:>6.1f} {error:>6.1f} "
            f"{detection.confidence:>8.1f}"
        )

    if count > 0:
        mae = total_error / count
        print("-" * 60)
        print(f"{'MAE':<40} {'':>6} {'':>6} {mae:>6.1f}")


if __name__ == "__main__":
    main()
