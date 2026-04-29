"""Improved classical CV gauge reader combining multiple detection strategies.

Key improvements over the original baseline:
1. Multi-scale detection: try different radius scales and pick the best
2. Better annulus: 20-70% (tighter, avoids hub and outer tick marks)
3. Improved subdial suppression
4. Combined detection: run multiple strategies and pick the best by quality
5. Better confidence scoring
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs
from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    _angle_in_sweep,
    _runner_up_peak_after_suppression,
    NeedleDetection,
)
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry

REPO_ROOT = PROJECT_ROOT.parent
spec: GaugeSpec = load_gauge_specs()["littlegood_home_temp_gauge_c"]


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
# Approach A: Improved spoke-vote with better annulus
# =============================================================================
def spoke_vote_improved(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    snr_threshold: float = 1.8,
) -> NeedleDetection | None:
    """Improved spoke-vote using tighter annulus and better gradient weighting."""
    h, w = gray.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )

    dx = xx - cx
    dy = yy - cy
    rr = np.sqrt(dx**2 + dy**2)

    # Tighter annulus: 20-70%
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

    # Radial and tangential
    rr_safe = np.where(rr > 0.5, rr, 1.0)
    radial_x = -dx / rr_safe
    radial_y = -dy / rr_safe

    # Tangential weight
    mag_safe = np.where(mag > 1.0, mag, 1.0)
    gx_n = gx / mag_safe
    gy_n = gy / mag_safe
    tang_weight = np.abs(gx_n * radial_y - gy_n * radial_x)

    # Vote
    vote = np.where(inner_mask & (mag > 8.0), mag * tang_weight, 0.0)

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
    margin = math.radians(6.0)
    for b in range(num_bins):
        ang = (b / num_bins) * 2.0 * math.pi - math.pi
        if not _angle_in_sweep(ang, spec, margin_rad=margin):
            histogram[b] = 0.0

    # Smooth
    kernel_width = 15
    hist_smooth = cv2.GaussianBlur(
        histogram[np.newaxis, :], (1, kernel_width * 2 + 1), 0
    ).ravel()

    best_bin = int(np.argmax(hist_smooth))
    peak_val = float(hist_smooth[best_bin])
    noise = float(np.mean(histogram)) + 1e-6
    snr = peak_val / noise

    if snr < snr_threshold:
        return None

    runner = _runner_up_peak_after_suppression(
        hist_smooth, best_index=best_bin, suppression_bins=kernel_width
    )
    peak_ratio = peak_val / max(runner, 1e-6)

    best_angle = (best_bin / num_bins) * 2.0 * math.pi - math.pi

    # Validate sweep
    if not _angle_in_sweep(best_angle, spec, margin_rad=margin):
        return None

    return NeedleDetection(
        unit_dx=math.cos(best_angle),
        unit_dy=math.sin(best_angle),
        confidence=snr,
        peak_value=peak_val,
        runner_up_value=runner,
        peak_ratio=peak_ratio,
        peak_margin=peak_val - runner,
    )


# =============================================================================
# Approach B: Center-weighted dark arc detection
# =============================================================================
def center_weighted_detect(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> NeedleDetection | None:
    """Vote based on darkness near center + gradient direction."""
    h, w = gray.shape[:2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    dx = xx - cx
    dy = yy - cy
    rr = np.sqrt(dx**2 + dy**2)

    # Focus on 25-60%
    inner_mask = (rr > 0.25 * radius) & (rr < 0.60 * radius)

    rr_safe = np.where(rr > 0.5, rr, 1.0)
    radial_x = -dx / rr_safe
    radial_y = -dy / rr_safe

    mag_safe = np.where(mag > 1.0, mag, 1.0)
    gx_n = gx / mag_safe
    gy_n = gy / mag_safe
    tang_weight = np.abs(gx_n * radial_y - gy_n * radial_x)

    # Center weighting
    center_weight = np.where(inner_mask, (1.0 - rr / (0.65 * radius)), 0.0)
    center_weight = np.clip(center_weight, 0.0, 1.0)

    # Dark pixels
    dark_mask = (blurred < 100) & inner_mask

    vote = np.where(dark_mask, center_weight * tang_weight * 50.0, 0.0)

    spoke_angle = np.arctan2(dy, dx)

    num_bins = 720
    angle_bins = ((spoke_angle + math.pi) / (2.0 * math.pi) * num_bins).astype(np.int32)
    angle_bins = np.clip(angle_bins, 0, num_bins - 1)

    vote_flat = vote.ravel()
    bin_flat = angle_bins.ravel()
    histogram = np.zeros(num_bins, dtype=np.float32)
    np.add.at(histogram, bin_flat, vote_flat)

    margin = math.radians(6.0)
    for b in range(num_bins):
        ang = (b / num_bins) * 2.0 * math.pi - math.pi
        if not _angle_in_sweep(ang, spec, margin_rad=margin):
            histogram[b] = 0.0

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
    if not _angle_in_sweep(best_angle, spec, margin_rad=margin):
        return None

    return NeedleDetection(
        unit_dx=math.cos(best_angle),
        unit_dy=math.sin(best_angle),
        confidence=snr,
        peak_value=peak_val,
        runner_up_value=runner,
        peak_ratio=peak_ratio,
        peak_margin=peak_val - runner,
    )


# =============================================================================
# Approach C: Multi-scale spoke vote
# =============================================================================
def multi_scale_detect(
    gray: np.ndarray,
    cx: float,
    cy: float,
    base_radius: float,
) -> NeedleDetection | None:
    """Run spoke-vote at multiple radius scales and pick the best."""
    # Try different radius scales
    scales = [0.80, 0.90, 1.00, 1.10, 1.20]
    results = []

    for scale in scales:
        radius = base_radius * scale
        det = spoke_vote_improved(gray, cx, cy, radius, snr_threshold=2.0)
        if det is not None:
            results.append((det, scale, det.confidence))

    if not results:
        return None

    # Sort by quality: confidence * peak_ratio
    results.sort(key=lambda x: x[2] * x[0].peak_ratio, reverse=True)
    return results[0][0]


# =============================================================================
# Approach D: Line segment detection
# =============================================================================
def line_detect(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> NeedleDetection | None:
    """Use LSD to find needle-like lines."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = ((rr > 0.20 * radius) & (rr < 0.75 * radius)).astype(np.uint8) * 255

    lsd = cv2.createLineSegmentDetector(1)
    lines = lsd.detect(enhanced, mask)

    if lines is None or len(lines) == 0:
        return None

    candidates = []
    for line in lines:
        line = np.asarray(line).ravel()
        if line.shape[0] != 4:
            continue
        x1, y1, x2, y2 = line

        line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_len < 10:
            continue

        vx = (x2 - x1) / line_len
        vy = (y2 - y1) / line_len

        t = max(0.0, min(1.0, ((cx - x1) * vx + (cy - y1) * vy) / (line_len**2 + 1e-6)))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        dist_to_center = np.sqrt((cx - proj_x) ** 2 + (cy - proj_y) ** 2)

        if dist_to_center < 3 or dist_to_center > 0.35 * radius:
            continue

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dir_x = mid_x - cx
        dir_y = mid_y - cy
        dir_len = np.sqrt(dir_x**2 + dir_y**2)
        if dir_len < 5:
            continue
        dir_x /= dir_len
        dir_y /= dir_len

        dot = abs(vx * dir_x + vy * dir_y)
        if abs(dot) < 0.7:
            continue

        score = line_len * (1.0 - dist_to_center / (0.35 * radius))

        d1 = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        d2 = np.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
        if d2 > d1:
            needle_dir_x, needle_dir_y = vx, vy
        else:
            needle_dir_x, needle_dir_y = -vx, -vy

        candidates.append((score, needle_dir_x, needle_dir_y, line_len))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, dx, dy, length = candidates[0]

    angle = math.atan2(dy, dx)
    shifted = (angle - spec.min_angle_rad) % (2.0 * math.pi)
    if shifted > spec.sweep_rad + math.radians(6.0):
        dx, dy = -dx, -dy
        angle = math.atan2(dy, dx)
        shifted = (angle - spec.min_angle_rad) % (2.0 * math.pi)
        if shifted > spec.sweep_rad + math.radians(6.0):
            return None

    return NeedleDetection(
        unit_dx=dx,
        unit_dy=dy,
        confidence=score / 100.0,
        peak_value=score,
        runner_up_value=0.0,
        peak_ratio=1.0,
        peak_margin=score,
    )


# =============================================================================
# Combined detector: try all approaches, pick best quality
# =============================================================================
def combined_detect(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> NeedleDetection | None:
    """Run all detection approaches and pick the best by quality."""
    detections = []

    # Approach A: Improved spoke-vote
    det_a = spoke_vote_improved(gray, cx, cy, radius, snr_threshold=2.0)
    if det_a is not None:
        quality_a = det_a.confidence * max(det_a.peak_ratio - 1.0, 0.0)
        detections.append(("spoke_v2", det_a, quality_a))

    # Approach B: Center-weighted
    det_b = center_weighted_detect(gray, cx, cy, radius)
    if det_b is not None:
        quality_b = det_b.confidence * max(det_b.peak_ratio - 1.0, 0.0)
        detections.append(("ctr_wt", det_b, quality_b))

    # Approach D: Line segment
    det_d = line_detect(gray, cx, cy, radius)
    if det_d is not None:
        quality_d = det_d.confidence
        detections.append(("line_seg", det_d, quality_d))

    if not detections:
        return None

    # Pick highest quality
    detections.sort(key=lambda x: x[2], reverse=True)
    best_name, best_det, best_quality = detections[0]

    return best_det


# =============================================================================
# Evaluation on hard case images
# =============================================================================
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


def main() -> None:
    """Evaluate the improved baseline on hard case images."""
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

        dets = [
            ("spoke_v2", spoke_vote_improved(gray, cx, radius, snr_threshold=2.0)),
            ("ctr_wt", center_weighted_detect(gray, cx, radius)),
            ("line_seg", line_detect(gray, cx, radius)),
            ("combined", combined_detect(gray, cx, radius)),
        ]
        for name, det in dets:
            if det is not None:
                angle = math.degrees(math.atan2(det.unit_dy, det.unit_dx))
                pred = angle_to_temp(math.atan2(det.unit_dy, det.unit_dx), spec)
                err = abs(pred - true_val)
                row[f"{name}_angle"] = f"{angle:.1f}"
                row[f"{name}_pred"] = f"{pred:.1f}"
                row[f"{name}_err"] = f"{err:.1f}"
                row[f"{name}_conf"] = f"{det.confidence:.2f}"
            else:
                row[f"{name}_angle"] = "NONE"
                row[f"{name}_pred"] = "NONE"
                row[f"{name}_err"] = "NONE"
                row[f"{name}_conf"] = "0.00"

        results.append(row)

        print(f"\n{fname} (true={true_val:.0f}°C, expected angle={expected_deg:.1f}°)")
        for name in ["spoke_v2", "ctr_wt", "line_seg", "combined"]:
            angle_str = row.get(f"{name}_angle", "N/A")
            pred_str = row.get(f"{name}_pred", "N/A")
            err_str = row.get(f"{name}_err", "N/A")
            conf_str = row.get(f"{name}_conf", "N/A")
            if pred_str != "NONE":
                print(
                    f"  {name:12s}: angle={angle_str:8s} pred={pred_str:8s} err={err_str:8s} conf={conf_str}"
                )
            else:
                print(f"  {name:12s}: NO DETECTION")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'File':25s} {'True':>6s} {'SpokeV2':>10s} {'CtrWt':>10s} {'LineSeg':>10s} {'Combined':>10s}"
    )
    print("-" * 80)
    for r in results:
        fname_short = (
            r["fname"]
            .replace("capture_", "")
            .replace(".png", "")
            .replace("_preview", "")
        )
        true = r["true"]
        sv2 = r.get("spoke_v2_err", "N/A")
        ctr = r.get("ctr_wt_err", "N/A")
        seg = r.get("line_seg_err", "N/A")
        comb = r.get("combined_err", "N/A")
        print(
            f"{fname_short:25s} {true:>6.0f} {sv2:>10s} {ctr:>10s} {seg:>10s} {comb:>10s}"
        )

    print("\n" + "=" * 80)
    print("MAE BY APPROACH")
    print("=" * 80)
    for name in ["spoke_v2", "ctr_wt", "line_seg", "combined"]:
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
                f"{name:12s}: MAE={mae:6.2f}°C  max={max_err:6.2f}°C  "
                f"over5={over5}/{len(errors)}"
            )


if __name__ == "__main__":
    main()
