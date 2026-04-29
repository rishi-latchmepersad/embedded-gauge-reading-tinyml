"""Try multiple classical CV approaches and compare them on hard case images.

Approaches to try:
1. Spoke-vote (gradient alignment in annulus)
2. Polar dark-stripe (what we have)
3. Hough line detection + center alignment
4. Template matching with needle shape
5. Color-based needle detection (orange/red hue threshold)

This helps us find what works best before committing to one approach.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry

REPO_ROOT = PROJECT_ROOT.parent
spec: GaugeSpec = load_gauge_specs()["littlegood_home_temp_gauge_c"]

IMAGES = [
    ("capture_m30c_preview.png", -30.0),
    ("capture_m10c_preview.png", -10.0),
    ("capture_0c_preview.png", 0.0),
    ("capture_p10c_preview.png", 10.0),
    ("capture_p20c_preview.png", 20.0),
    ("capture_p35c_preview.png", 35.0),
    ("capture_p50c_preview.png", 50.0),
    ("capture_0075.png", 19.0),
    ("capture_2026-04-03_08-20-49.png", 45.0),
    ("capture_2026-04-24_22-24-04.png", 10.0),
    ("capture_2026-04-24_22-30-21.png", 10.0),
]


def angle_to_temp(angle_rad: float, spec: GaugeSpec) -> float:
    """Map an angle to temperature using the gauge spec."""
    raw = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
    shifted = (raw - spec.min_angle_rad) % (2.0 * math.pi)
    fraction = min(max(shifted / spec.sweep_rad, 0.0), 1.0)
    return spec.min_value + fraction * (spec.max_value - spec.min_value)


def expected_angle(temp: float, spec: GaugeSpec) -> float:
    """Get the expected needle angle for a temperature."""
    fraction = (temp - spec.min_value) / (spec.max_value - spec.min_value)
    return spec.min_angle_rad + fraction * spec.sweep_rad


# =============================================================================
# Approach 1: Improved spoke-vote with better annulus and subdial handling
# =============================================================================
def spoke_vote_v2(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> Optional[tuple[float, float, float]]:
    """Improved spoke-vote using tighter annulus and better gradient weighting."""
    h, w = gray.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )

    dx = xx - cx
    dy = yy - cy
    rr = np.sqrt(dx**2 + dy**2)

    # Tighter annulus: 20-70% (was 15-75%)
    inner_mask = (rr > 0.20 * radius) & (rr < 0.70 * radius)

    # Better subdial suppression (wider zone)
    dx_sub = np.abs(xx - cx)
    dy_sub = yy - cy
    subdial = (
        (rr > 0.25 * radius)
        & (dx_sub < 0.40 * radius)
        & (dy_sub > 0.08 * radius)
        & (dy_sub < 0.62 * radius)
    )
    inner_mask = inner_mask & ~subdial

    # Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    # Radial and tangential computation
    rr_safe = np.where(rr > 0.5, rr, 1.0)
    radial_x = -dx / rr_safe
    radial_y = -dy / rr_safe

    # Tangential weight (cross product)
    mag_safe = np.where(mag > 1.0, mag, 1.0)
    gx_n = gx / mag_safe
    gy_n = gy / mag_safe
    tang_weight = np.abs(gx_n * radial_y - gy_n * radial_x)

    # Vote weighting
    vote = np.where(inner_mask & (mag > 8.0), mag * tang_weight, 0.0)

    # Spoke angles
    spoke_angle = np.arctan2(dy, dx)

    # Histogram with 720 bins
    num_bins = 720
    angle_bins = ((spoke_angle + math.pi) / (2.0 * math.pi) * num_bins).astype(np.int32)
    angle_bins = np.clip(angle_bins, 0, num_bins - 1)

    vote_flat = vote.ravel()
    bin_flat = angle_bins.ravel()
    histogram = np.zeros(num_bins, dtype=np.float32)
    np.add.at(histogram, bin_flat, vote_flat)

    # Zero outside sweep
    for b in range(num_bins):
        ang = (b / num_bins) * 2.0 * math.pi - math.pi
        shifted = (ang - spec.min_angle_rad) % (2.0 * math.pi)
        if shifted > spec.sweep_rad + math.radians(6.0):
            histogram[b] = 0.0

    # Smooth
    hist_smooth = cv2.GaussianBlur(histogram[np.newaxis, :], (1, 31), 0).ravel()

    best_bin = int(np.argmax(hist_smooth))
    peak_val = float(hist_smooth[best_bin])
    noise = float(np.mean(hist_smooth)) + 1e-6
    snr = peak_val / noise

    if snr < 2.0:
        return None

    best_angle = (best_bin / num_bins) * 2.0 * math.pi - math.pi
    return math.cos(best_angle), math.sin(best_angle), snr


# =============================================================================
# Approach 2: Center-weighted dark arc detection
# =============================================================================
def center_weighted_vote(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> Optional[tuple[float, float, float]]:
    """Vote based on darkness near center + gradient direction."""
    h, w = gray.shape[:2]

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Sobel gradients
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    dx = xx - cx
    dy = yy - cy
    rr = np.sqrt(dx**2 + dy**2)

    # Focus on inner 25-60% where needle is most visible and hub/subdial effects are minimal
    inner_mask = (rr > 0.25 * radius) & (rr < 0.60 * radius)

    # Radial direction from center
    rr_safe = np.where(rr > 0.5, rr, 1.0)
    radial_x = -dx / rr_safe
    radial_y = -dy / rr_safe

    # Gradient direction (should be tangential for needle)
    mag_safe = np.where(mag > 1.0, mag, 1.0)
    gx_n = gx / mag_safe
    gy_n = gy / mag_safe
    tang_weight = np.abs(gx_n * radial_y - gy_n * radial_x)

    # Center-weighting: closer to center gets higher weight
    center_weight = np.where(inner_mask, (1.0 - rr / (0.65 * radius)), 0.0)
    center_weight = np.clip(center_weight, 0.0, 1.0)

    # Dark pixel detection (needle is dark on light background)
    dark_mask = (blurred < 100) & inner_mask

    # Combined vote: dark pixels with tangential gradient weighted by center proximity
    vote = np.where(dark_mask, center_weight * tang_weight * 50.0, 0.0)

    # Spoke angles
    spoke_angle = np.arctan2(dy, dx)

    # Histogram
    num_bins = 720
    angle_bins = ((spoke_angle + math.pi) / (2.0 * math.pi) * num_bins).astype(np.int32)
    angle_bins = np.clip(angle_bins, 0, num_bins - 1)

    vote_flat = vote.ravel()
    bin_flat = angle_bins.ravel()
    histogram = np.zeros(num_bins, dtype=np.float32)
    np.add.at(histogram, bin_flat, vote_flat)

    # Zero outside sweep
    for b in range(num_bins):
        ang = (b / num_bins) * 2.0 * math.pi - math.pi
        shifted = (ang - spec.min_angle_rad) % (2.0 * math.pi)
        if shifted > spec.sweep_rad + math.radians(6.0):
            histogram[b] = 0.0

    # Smooth
    hist_smooth = cv2.GaussianBlur(histogram[np.newaxis, :], (1, 21), 0).ravel()

    best_bin = int(np.argmax(hist_smooth))
    peak_val = float(hist_smooth[best_bin])
    runner = float(np.max(np.roll(hist_smooth, -best_bin)[1:20]))
    peak_ratio = peak_val / max(runner, 1e-6)
    noise = float(np.mean(hist_smooth)) + 1e-6
    snr = peak_val / noise

    if snr < 1.5 or peak_ratio < 1.1:
        return None

    best_angle = (best_bin / num_bins) * 2.0 * math.pi - math.pi
    return math.cos(best_angle), math.sin(best_angle), snr


# =============================================================================
# Approach 3: Line segment detection near needle direction
# =============================================================================
def line_segment_vote(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> Optional[tuple[float, float, float]]:
    """Use LSD (Line Segment Detector) to find needle-like lines."""
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Create mask for inner dial region
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = ((rr > 0.20 * radius) & (rr < 0.75 * radius)).astype(np.uint8) * 255

    # LSD detector
    lsd = cv2.createLineSegmentDetector(1)
    lines = lsd.detect(enhanced, mask)

    if lines is None or len(lines) == 0:
        return None

    # Filter lines: we want lines that pass near the center (needle base)
    # and extend outward at an angle that matches the gauge sweep
    candidates = []
    for line in lines:
        # OpenCV LineSegmentDetector returns lines as:
        # - (N, 1, 4) in older versions
        # - (N, 4) in newer versions
        # We need to handle both cases
        line = np.asarray(line).ravel()
        if line.shape[0] != 4:
            continue
        x1, y1, x2, y2 = line
        # Distance from center to line
        line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_len < 10:
            continue

        # Vector from x1,y1 to x2,y2
        vx = (x2 - x1) / line_len
        vy = (y2 - y1) / line_len

        # Check if line passes near center
        # Distance from center to line (point-to-segment)
        t = max(0.0, min(1.0, ((cx - x1) * vx + (cy - y1) * vy) / (line_len**2 + 1e-6)))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        dist_to_center = np.sqrt((cx - proj_x) ** 2 + (cy - proj_y) ** 2)

        # Line should not be too close to center (needle is a line, not a point)
        # and should not be too far (should be on the dial face)
        if dist_to_center < 3 or dist_to_center > 0.35 * radius:
            continue

        # The direction from center to line midpoint should match needle direction
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dir_x = mid_x - cx
        dir_y = mid_y - cy
        dir_len = np.sqrt(dir_x**2 + dir_y**2)
        if dir_len < 5:
            continue
        dir_x /= dir_len
        dir_y /= dir_len

        # Line direction should be roughly tangential (needle is radial)
        # Actually needle IS radial, so gradient is tangential
        # But we're looking at the line direction itself
        dot = abs(
            vx * dir_x + vy * dir_y
        )  # |cos(theta)| between line direction and radial

        # Want lines that are somewhat radial (not tangential tick marks)
        # For a needle, the line direction should be ~parallel to radial (dot close to 1)
        # or anti-parallel (dot close to -1)
        if abs(dot) < 0.7:
            continue

        # Score based on line length and distance from center
        score = line_len * (1.0 - dist_to_center / (0.35 * radius))

        # The needle direction is from center toward the line's farther end
        # Determine which end is farther from center
        d1 = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        d2 = np.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
        if d2 > d1:
            needle_dir_x, needle_dir_y = vx, vy
        else:
            needle_dir_x, needle_dir_y = -vx, -vy

        candidates.append((score, needle_dir_x, needle_dir_y, line_len))

    if not candidates:
        return None

    # Pick the best candidate by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, dx, dy, length = candidates[0]

    # Verify direction is in sweep
    angle = math.atan2(dy, dx)
    shifted = (angle - spec.min_angle_rad) % (2.0 * math.pi)
    if shifted > spec.sweep_rad + math.radians(6.0):
        # Flip direction (opposite end of line might be the needle tip)
        dx, dy = -dx, -dy
        angle = math.atan2(dy, dx)
        shifted = (angle - spec.min_angle_rad) % (2.0 * math.pi)
        if shifted > spec.sweep_rad + math.radians(6.0):
            return None

    return dx, dy, score / 100.0


# =============================================================================
# Approach 4: Simple radial line detection
# =============================================================================
def radial_line_detection(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> Optional[tuple[float, float, float]]:
    """Look for dark radial lines from center outward.

    This is simpler than the gradient approach: find dark streaks that
    radiate from near the center toward the edge of the dial.
    """
    h, w = gray.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Compute darkness: darker = more negative gradient from local mean
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)
    dark_score = blurred.astype(np.float32)  # Lower is darker

    # Create radial spokes at various angles and sample darkness along each
    num_angles = 360
    angle_scores = np.zeros(num_angles)

    for i in range(num_angles):
        angle = -math.pi + (2.0 * math.pi) * i / num_angles
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Sample points along the spoke (from 25% to 70% of radius)
        total_score = 0.0
        count = 0
        for frac in np.linspace(0.25, 0.70, 20):
            r = frac * radius
            px = int(round(cx + r * cos_a))
            py = int(round(cy + r * sin_a))
            if 0 <= px < w and 0 <= py < h:
                # Darkness is inverted (lower pixel value = darker = more negative diff)
                diff = float(blurred[py, px]) - 128.0  # deviation from mid-gray
                total_score += -diff  # more negative = darker = higher score
                count += 1

        if count > 0:
            angle_scores[i] = total_score / count

    # Smooth scores
    angle_scores_smooth = cv2.GaussianBlur(
        angle_scores[np.newaxis, :], (1, 15), 0
    ).ravel()

    # Zero out angles outside sweep
    valid_mask = np.zeros(num_angles, dtype=bool)
    for i in range(num_angles):
        angle = -math.pi + (2.0 * math.pi) * i / num_angles
        shifted = (angle - spec.min_angle_rad) % (2.0 * math.pi)
        valid_mask[i] = shifted <= spec.sweep_rad + math.radians(6.0)
    angle_scores_smooth = np.where(valid_mask, angle_scores_smooth, 0.0)

    best_i = int(np.argmax(angle_scores_smooth))
    best_score = float(angle_scores_smooth[best_i])
    noise = float(np.std(angle_scores_smooth[valid_mask])) + 1e-6
    snr = best_score / noise

    if snr < 2.0:
        return None

    best_angle = -math.pi + (2.0 * math.pi) * best_i / num_angles
    return math.cos(best_angle), math.sin(best_angle), snr


# =============================================================================
# Run comparison
# =============================================================================
def main() -> None:
    """Test all approaches on hard case images."""
    results = []

    for fname, true_val in IMAGES:
        img_path = REPO_ROOT / "captured_images" / fname
        if not img_path.exists():
            print(f"SKIP {fname}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"SKIP {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        estimated = estimate_dial_geometry(img)
        if estimated is not None:
            (cx, cy), radius = estimated
        else:
            h, w = gray.shape[:2]
            cx, cy = 0.5 * w, 0.5 * h
            radius = 0.45 * min(h, w)

        expected = expected_angle(true_val, spec)
        expected_deg = math.degrees(expected)
        if expected_deg > 180:
            expected_deg -= 360

        row = {"fname": fname, "true": true_val, "expected_deg": expected_deg}

        # Test each approach
        for name, func in [
            ("spoke_vote_v2", spoke_vote_v2),
            ("center_weighted", center_weighted_vote),
            ("line_segment", line_segment_vote),
            ("radial_line", radial_line_detection),
        ]:
            result = func(gray, cx, cy, radius)
            if result is not None:
                dx, dy, conf = result
                angle = math.degrees(math.atan2(dy, dx))
                pred = angle_to_temp(math.atan2(dy, dx), spec)
                err = abs(pred - true_val)
                row[f"{name}_angle"] = f"{angle:.1f}"
                row[f"{name}_pred"] = f"{pred:.1f}"
                row[f"{name}_err"] = f"{err:.1f}"
                row[f"{name}_conf"] = f"{conf:.2f}"
            else:
                row[f"{name}_angle"] = "NONE"
                row[f"{name}_pred"] = "NONE"
                row[f"{name}_err"] = "NONE"
                row[f"{name}_conf"] = "0.00"

        results.append(row)

        # Print summary for this image
        print(f"\n{fname} (true={true_val:.0f}°C, expected angle={expected_deg:.1f}°)")
        for name, _, _, _ in [
            ("spoke_vote_v2", "", "", ""),
            ("center_weighted", "", "", ""),
            ("line_segment", "", "", ""),
            ("radial_line", "", "", ""),
        ]:
            angle_str = row.get(f"{name}_angle", "N/A")
            pred_str = row.get(f"{name}_pred", "N/A")
            err_str = row.get(f"{name}_err", "N/A")
            conf_str = row.get(f"{name}_conf", "N/A")
            if pred_str != "NONE":
                print(
                    f"  {name:20s}: angle={angle_str:8s} pred={pred_str:8s} err={err_str:8s} conf={conf_str}"
                )
            else:
                print(f"  {name:20s}: NO DETECTION")

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(
        f"{'File':30s} {'True':>6s} {'SpokeV2':>12s} {'CtrWt':>12s} {'LineSeg':>12s} {'Radial':>12s}"
    )
    print("-" * 100)

    for r in results:
        fname_short = (
            r["fname"]
            .replace("capture_", "")
            .replace(".png", "")
            .replace("_preview", "")
        )
        true = r["true"]
        sv2 = r.get("spoke_vote_v2_err", "N/A")
        ctr = r.get("center_weighted_err", "N/A")
        seg = r.get("line_segment_err", "N/A")
        rad = r.get("radial_line_err", "N/A")
        print(
            f"{fname_short:30s} {true:>6.0f} {sv2:>12s} {ctr:>12s} {seg:>12s} {rad:>12s}"
        )

    # Compute MAE for each approach
    print("\n" + "=" * 100)
    print("MAE BY APPROACH")
    print("=" * 100)

    for name in ["spoke_vote_v2", "center_weighted", "line_segment", "radial_line"]:
        errors = []
        for r in results:
            err_str = r.get(f"{name}_err", "NONE")
            if err_str not in ("NONE", "N/A"):
                errors.append(float(err_str))
        if errors:
            mae = sum(errors) / len(errors)
            max_err = max(errors)
            over5 = sum(1 for e in errors if e > 5.0)
            print(
                f"{name:20s}: MAE={mae:6.2f}°C  max={max_err:6.2f}°C  over5={over5}/{len(errors)}"
            )


if __name__ == "__main__":
    main()
