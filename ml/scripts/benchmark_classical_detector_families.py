"""Benchmark candidate pure-classical detector families on the hard cases.

This script compares a few classical needle readers on the current hard-case
focus manifests so we can keep the baseline focused on the strongest family
instead of chasing brittle alternatives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import math
import sys
from typing import Callable

import cv2
import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.baseline_classical_cv import (  # noqa: E402
    NeedleDetection,
    _angle_in_sweep,
    _detect_needle_unit_vector_polar,
    _point_to_segment_distance,
    _sample_line_darkness,
    detect_needle_unit_vector,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs  # noqa: E402
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry  # noqa: E402


REPO_ROOT: Path = PROJECT_ROOT.parent
DEFAULT_MANIFESTS: tuple[Path, Path] = (
    PROJECT_ROOT / "data" / "hard_cases.csv",
    PROJECT_ROOT / "data" / "hard_cases_remaining_focus.csv",
)
DEFAULT_OUTPUT: Path = PROJECT_ROOT / "artifacts" / "baseline" / "detector_family_sweep" / "benchmark.json"


@dataclass(frozen=True)
class DetectorSummary:
    """Aggregate metrics for one candidate detector family."""

    name: str
    attempted: int
    successful: int
    failed: int
    mae: float
    rmse: float
    cases_over_5c: int


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sweep."""
    parser = argparse.ArgumentParser(
        description="Benchmark pure-classical detector families on hard-case manifests."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help="Optional manifest override. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the JSON summary artifact.",
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default="littlegood_home_temp_gauge_c",
    )
    return parser.parse_args()


def _json_safe_float(value: float) -> float | None:
    """Convert NaN/inf values into JSON-friendly nulls."""
    if math.isfinite(value):
        return float(value)
    return None


def _load_rows(manifests: list[Path]) -> list[tuple[Path, float, str]]:
    """Load the image/value pairs from one or more manifest CSV files."""
    rows: list[tuple[Path, float, str]] = []
    for manifest_path in manifests:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    (REPO_ROOT / row["image_path"], float(row["value"]), manifest_path.name)
                )
    return rows


def _estimate_geometry(image_bgr: np.ndarray) -> tuple[tuple[float, float], float]:
    """Estimate a classical center/radius seed for the detector families."""
    estimated = estimate_dial_geometry(image_bgr)
    if estimated is not None:
        return estimated

    height, width = image_bgr.shape[:2]
    return (0.5 * float(width), 0.5 * float(height)), 0.45 * float(min(height, width))


def _is_in_subdial_mask(center_x: float, center_y: float, x: float, y: float, radius_px: float) -> bool:
    """Match the firmware subdial mask so the sweep ignores the same clutter."""
    dx = abs(x - center_x)
    dy = y - center_y
    return (
        dx < (0.35 * radius_px)
        and (y > (center_y + (0.10 * radius_px)))
        and (y < (center_y + (0.58 * radius_px)))
        and (dy > (0.10 * radius_px))
    )


def _gradient_polar_detector(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec,
) -> NeedleDetection | None:
    """Run the current gradient-polar spoke voter used by the Python baseline."""
    return detect_needle_unit_vector(
        image_bgr,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        gauge_spec=gauge_spec,
    )


def _ray_score_detector(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec,
) -> NeedleDetection | None:
    """Score candidate angles by sampling the shaft as a dark radial ray."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    center_x, center_y = center_xy
    height, width = gray.shape[:2]
    angle_bins = 360
    max_radius = min(float(min(height, width)) / 2.0, dial_radius_px * 0.98)
    if max_radius <= 1.0:
        return None

    start_radius = max(1.0, max_radius * 0.20)
    end_radius = max(start_radius + 1.0, max_radius * 0.78)
    sample_radii = np.linspace(start_radius, end_radius, 32, dtype=np.float32)
    angle_scores = np.zeros(angle_bins, dtype=np.float32)
    sweep_start = gauge_spec.min_angle_rad
    sweep_rad = gauge_spec.sweep_rad

    for bin_index in range(angle_bins):
        fraction = float(bin_index) / float(angle_bins)
        angle_rad = sweep_start + fraction * sweep_rad
        unit_dx = math.cos(angle_rad)
        unit_dy = math.sin(angle_rad)
        perp_dx = -unit_dy
        perp_dy = unit_dx

        score = 0.0
        valid_samples = 0
        for sample_index, radius in enumerate(sample_radii):
            sample_progress = float(sample_index) / float(len(sample_radii) - 1)
            weight = 1.0 - (0.50 * sample_progress)
            sample_x = center_x + (unit_dx * float(radius))
            sample_y = center_y + (unit_dy * float(radius))
            ix = int(round(sample_x))
            iy = int(round(sample_y))
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                continue
            if _is_in_subdial_mask(center_x, center_y, float(ix), float(iy), max_radius):
                continue

            line_luma = float(blurred[iy, ix])
            if line_luma > 220.0:
                continue

            background_sum = 0.0
            background_count = 0
            for offset in (2.0, 4.0):
                for direction in (-1.0, 1.0):
                    bg_x = sample_x + direction * offset * perp_dx
                    bg_y = sample_y + direction * offset * perp_dy
                    bg_ix = int(round(bg_x))
                    bg_iy = int(round(bg_y))
                    if bg_ix < 0 or bg_iy < 0 or bg_ix >= width or bg_iy >= height:
                        continue
                    if _is_in_subdial_mask(center_x, center_y, float(bg_ix), float(bg_iy), max_radius):
                        continue
                    bg_luma = float(blurred[bg_iy, bg_ix])
                    if bg_luma > 220.0:
                        continue
                    background_sum += bg_luma
                    background_count += 1

            if background_count == 0:
                continue

            local_background = background_sum / float(background_count)
            local_contrast = local_background - line_luma
            if local_contrast <= 0.0:
                continue

            score += local_contrast * weight
            valid_samples += 1

        if valid_samples > 0:
            angle_scores[bin_index] = score / float(valid_samples)

    if np.all(angle_scores <= 0.0):
        return None

    # Mild smoothing so a real spoke can win over single-bin noise.
    padded = np.concatenate(([angle_scores[-1]], angle_scores, [angle_scores[0]]))
    smoothed = np.convolve(padded, np.array([1.0, 1.0, 1.0], dtype=np.float32), mode="valid") / 3.0

    valid_mask = np.array(
        [
            _angle_in_sweep(
                sweep_start + (float(idx) / float(angle_bins)) * sweep_rad,
                gauge_spec,
                margin_rad=math.radians(12.0),
            )
            for idx in range(angle_bins)
        ],
        dtype=bool,
    )
    valid_scores = np.where(valid_mask, smoothed, 0.0)
    best_bin = int(np.argmax(valid_scores))
    best_score = float(valid_scores[best_bin])
    if best_score <= 0.0:
        return None

    mean_score = float(np.mean(valid_scores)) + 1e-6
    confidence = best_score / mean_score
    if confidence < 1.2:
        return None

    best_fraction = float(best_bin) / float(angle_bins - 1)
    best_angle = sweep_start + (best_fraction * sweep_rad)
    return NeedleDetection(
        unit_dx=math.cos(best_angle),
        unit_dy=math.sin(best_angle),
        confidence=float(confidence),
        peak_value=float(best_score),
        runner_up_value=0.0,
        peak_ratio=1.0,
        peak_margin=float(best_score),
    )


def _hough_lines_detector(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec,
) -> NeedleDetection | None:
    """Use probabilistic Hough line segments as a classical needle candidate."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120, apertureSize=3, L2gradient=True)

    min_line_length = max(24, int(round(0.30 * dial_radius_px)))
    max_line_gap = max(6, int(round(0.08 * dial_radius_px)))
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

    center_x, center_y = center_xy
    best_score = float("-inf")
    second_score = float("-inf")
    best_vec: tuple[float, float] | None = None

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        seg_dx = x2 - x1
        seg_dy = y2 - y1
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len < min_line_length:
            continue

        center_dist = _point_to_segment_distance(center_x, center_y, x1, y1, x2, y2)
        if center_dist > 0.22 * dial_radius_px:
            continue

        d1 = math.hypot(x1 - center_x, y1 - center_y)
        d2 = math.hypot(x2 - center_x, y2 - center_y)
        near = min(d1, d2)
        far = max(d1, d2)
        if near > 0.34 * dial_radius_px or far < 0.55 * dial_radius_px:
            continue

        far_x, far_y = (x1, y1) if d1 > d2 else (x2, y2)
        angle_rad = math.atan2(far_y - center_y, far_x - center_x)
        if not _angle_in_sweep(angle_rad, gauge_spec, margin_rad=math.radians(14.0)):
            continue

        contrast_mean, dark_fraction = _sample_line_darkness(
            gray,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        if contrast_mean <= 0.0 and dark_fraction <= 0.0:
            continue

        radiality = 1.0 - min(center_dist / max(0.22 * dial_radius_px, 1.0), 1.0)
        reach = min(far / max(0.95 * dial_radius_px, 1.0), 1.0)
        shaft_balance = 1.0 - min(abs(near - 0.10 * dial_radius_px) / max(0.20 * dial_radius_px, 1.0), 1.0)
        darkness = max(0.0, contrast_mean) + 0.5 * max(0.0, dark_fraction)
        score = (
            seg_len
            * darkness
            * (0.35 + 0.65 * radiality)
            * (0.35 + 0.65 * reach)
            * (0.35 + 0.65 * shaft_balance)
        )

        if score > best_score:
            second_score = best_score
            best_score = score
            unit_len = max(far, 1e-6)
            best_vec = ((far_x - center_x) / unit_len, (far_y - center_y) / unit_len)
        elif score > second_score:
            second_score = score

    if best_vec is None:
        return None

    confidence = best_score / max(second_score, 1e-6) if second_score > 0.0 else best_score
    if confidence < 1.08 or best_score < 2.0:
        return None

    return NeedleDetection(
        unit_dx=best_vec[0],
        unit_dy=best_vec[1],
        confidence=float(confidence),
        peak_value=float(best_score),
        runner_up_value=float(second_score if second_score > 0.0 else 0.0),
        peak_ratio=float(best_score / max(second_score, 1e-6) if second_score > 0.0 else 1.0),
        peak_margin=float(best_score - second_score if second_score > 0.0 else best_score),
    )


def _dark_polar_detector(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec,
) -> NeedleDetection | None:
    """Detect the needle as a dark vertical band in polar coordinates."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    angle_bins = 720
    radius_bins = max(64, int(round(dial_radius_px)))
    max_radius = min(float(min(blurred.shape[:2])) / 2.0, dial_radius_px * 0.98)
    if max_radius <= 1.0:
        return None

    polar = cv2.warpPolar(
        blurred,
        (angle_bins, radius_bins),
        center_xy,
        max_radius,
        cv2.WARP_POLAR_LINEAR,
    )
    if polar.size == 0:
        return None

    start_row = max(1, int(0.22 * radius_bins))
    end_row = max(start_row + 1, int(0.95 * radius_bins))
    radial_slice = polar[start_row:end_row, :].astype(np.float32)
    if radial_slice.size == 0:
        return None

    column_means = np.mean(radial_slice, axis=0)
    column_q20 = np.percentile(radial_slice, 20.0, axis=0)
    angular_profile = 0.55 * column_means + 0.45 * column_q20
    smoothed = cv2.GaussianBlur(angular_profile[np.newaxis, :], (1, 31), 0).ravel()
    local_baseline = cv2.blur(smoothed[np.newaxis, :], (1, 61), borderType=cv2.BORDER_REFLECT).ravel()
    contrast_profile = local_baseline - smoothed

    candidate_indices = np.arange(angle_bins)
    candidate_contrasts = contrast_profile[candidate_indices]
    best_offset = int(np.argmax(candidate_contrasts))
    best_index = int(candidate_indices[best_offset])
    best_contrast = float(candidate_contrasts[best_offset])
    noise = float(np.std(candidate_contrasts)) + 1e-6
    contrast_score = best_contrast / noise

    if best_contrast <= 0.0 or contrast_score < 0.05:
        return None

    angle_rad = (2.0 * math.pi * best_index) / float(angle_bins)
    if not _angle_in_sweep(angle_rad, gauge_spec, margin_rad=math.radians(12.0)):
        return None

    unit_dx = math.cos(angle_rad)
    unit_dy = math.sin(angle_rad)
    return NeedleDetection(
        unit_dx=unit_dx,
        unit_dy=unit_dy,
        confidence=float(contrast_score),
        peak_value=float(best_contrast),
        runner_up_value=0.0,
        peak_ratio=1.0,
        peak_margin=float(best_contrast),
    )


def _evaluate_detector(
    name: str,
    detector: Callable[[np.ndarray, tuple[float, float], float, GaugeSpec], NeedleDetection | None],
    rows: list[tuple[Path, float, str]],
    spec: GaugeSpec,
) -> DetectorSummary:
    """Evaluate one detector family on the chosen hard-case manifests."""
    errors: list[float] = []
    attempted = 0
    successful = 0
    failed = 0

    for image_path, true_value, _manifest_name in rows:
        attempted += 1
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            continue

        center_xy, dial_radius_px = _estimate_geometry(image)
        detection = detector(
            image,
            center_xy=center_xy,
            dial_radius_px=dial_radius_px,
            gauge_spec=spec,
        )
        if detection is None:
            failed += 1
            continue

        successful += 1
        predicted_value = needle_vector_to_value(detection.unit_dx, detection.unit_dy, spec)
        errors.append(abs(predicted_value - true_value))

    if errors:
        error_array = np.asarray(errors, dtype=np.float32)
        mae = float(np.mean(error_array))
        rmse = float(np.sqrt(np.mean(np.square(error_array))))
    else:
        mae = float("nan")
        rmse = float("nan")

    return DetectorSummary(
        name=name,
        attempted=attempted,
        successful=successful,
        failed=failed,
        mae=mae,
        rmse=rmse,
        cases_over_5c=int(sum(error > 5.0 for error in errors)),
    )


def main() -> None:
    """Run the sweep, write a JSON artifact, and print the ranking."""
    args = _parse_args()
    manifests = args.manifest if args.manifest is not None else list(DEFAULT_MANIFESTS)
    rows = _load_rows(manifests)

    specs = load_gauge_specs()
    if args.gauge_id not in specs:
        raise ValueError(f"Unknown gauge_id '{args.gauge_id}'. Available: {list(specs)}")
    spec = specs[args.gauge_id]

    summaries = [
        _evaluate_detector("gradient_polar", _gradient_polar_detector, rows, spec),
        _evaluate_detector("ray_score", _ray_score_detector, rows, spec),
        _evaluate_detector("hough_lines", _hough_lines_detector, rows, spec),
        _evaluate_detector("dark_polar", _dark_polar_detector, rows, spec),
    ]
    summaries.sort(key=lambda item: (item.failed, item.mae, item.cases_over_5c))

    payload = {
        "gauge_id": args.gauge_id,
        "manifests": [manifest.name for manifest in manifests],
        "detectors": [
            {
                **asdict(summary),
                "mae": _json_safe_float(summary.mae),
                "rmse": _json_safe_float(summary.rmse),
            }
            for summary in summaries
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print("=== Detector Family Sweep ===")
    for summary in summaries:
        print(
            f"{summary.name}: mae={summary.mae:.4f} rmse={summary.rmse:.4f} "
            f"successful={summary.successful}/{summary.attempted} "
            f"failed={summary.failed} cases_over_5c={summary.cases_over_5c}"
        )
    print(f"[SWEEP] Wrote {args.output}")


if __name__ == "__main__":
    main()
