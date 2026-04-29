"""Probe whether a polarity-agnostic polar detector helps the hard board cases."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
ML_SRC = REPO_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from embedded_gauge_reading_tinyml.baseline_classical_cv import (  # noqa: E402
    NeedleDetection,
    _runner_up_peak_after_suppression,
    _angle_in_sweep,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs  # noqa: E402
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry  # noqa: E402


def detect_polar_signed(
    image_bgr: np.ndarray,
    *,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    gauge_spec: GaugeSpec,
    polarity: str,
) -> NeedleDetection | None:
    """Detect a needle as either a dark or bright radial stripe in polar space."""
    if dial_radius_px <= 1.0:
        return None

    center_x, center_y = center_xy
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
        (center_x, center_y),
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
    column_q80 = np.percentile(radial_slice, 80.0, axis=0)
    if polarity == "dark":
        angular_profile = 0.55 * column_means + 0.45 * column_q20
    elif polarity == "bright":
        angular_profile = 0.55 * column_means + 0.45 * column_q80
    else:
        dark_profile = 0.55 * column_means + 0.45 * column_q20
        bright_profile = 0.55 * column_means + 0.45 * column_q80
        # Keep the stronger contrast direction at each angle bin.
        angular_profile = np.where(
            np.abs(bright_profile - np.mean(bright_profile))
            > np.abs(dark_profile - np.mean(dark_profile)),
            bright_profile,
            dark_profile,
        )

    smoothed = cv2.GaussianBlur(angular_profile[np.newaxis, :], (1, 31), 0).ravel()
    local_baseline = cv2.blur(
        smoothed[np.newaxis, :],
        (1, 61),
        borderType=cv2.BORDER_REFLECT,
    ).ravel()

    if polarity == "bright":
        contrast_profile = smoothed - local_baseline
    elif polarity == "dark":
        contrast_profile = local_baseline - smoothed
    else:
        dark_contrast = local_baseline - smoothed
        bright_contrast = smoothed - local_baseline
        # Use the stronger signed deviation so the detector can adapt to either polarity.
        contrast_profile = np.where(
            np.abs(bright_contrast) > np.abs(dark_contrast),
            bright_contrast,
            dark_contrast,
        )

    start_angle_rad, sweep_rad = gauge_spec.min_angle_rad, gauge_spec.sweep_rad
    start_index = int(round((start_angle_rad % (2.0 * math.pi)) / (2.0 * math.pi) * angle_bins))
    sweep_bins = max(1, int(round((sweep_rad / (2.0 * math.pi)) * angle_bins)))
    candidate_indices = (start_index + np.arange(sweep_bins + 1)) % angle_bins

    candidate_contrasts = contrast_profile[candidate_indices]
    best_offset = int(np.argmax(candidate_contrasts))
    best_index = int(candidate_indices[best_offset])
    best_contrast = float(candidate_contrasts[best_offset])
    noise = float(np.std(candidate_contrasts)) + 1e-6
    contrast_score = abs(best_contrast) / noise
    if contrast_score < 0.45:
        return None

    runner_up = _runner_up_peak_after_suppression(
        np.abs(candidate_contrasts),
        best_index=best_offset,
        suppression_bins=15,
    )
    peak_ratio = abs(best_contrast) / max(runner_up, 1e-6)

    angle_rad = (2.0 * math.pi * best_index) / float(angle_bins)
    if not _angle_in_sweep(angle_rad, gauge_spec, margin_rad=math.radians(6.0)):
        return None

    return NeedleDetection(
        unit_dx=float(math.cos(angle_rad)),
        unit_dy=float(math.sin(angle_rad)),
        confidence=float(contrast_score),
        peak_value=float(abs(best_contrast)),
        runner_up_value=float(runner_up),
        peak_ratio=float(peak_ratio),
        peak_margin=float(abs(best_contrast) - runner_up),
    )


def main() -> None:
    """Run the signed polar detector on the board30-style samples."""
    spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    samples: list[tuple[str, float]] = [
        ("captured_images/today_converted/capture_2026-04-09_06-41-57.png", 30.0),
        ("captured_images/today_converted/capture_2026-04-09_06-50-28.png", 30.0),
        ("captured_images/today_converted/capture_2026-04-09_06-51-13.png", 30.0),
        ("captured_images/capture_p35c_preview.png", 35.0),
        ("captured_images/capture_2026-04-03_08-20-49.png", 45.0),
    ]

    for rel_path, true_value in samples:
        img_path = REPO_ROOT / rel_path
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"missing: {rel_path}")
            continue

        height, width = image_bgr.shape[:2]
        estimated = estimate_dial_geometry(image_bgr)
        if estimated is None:
            center_xy = (0.5 * float(width), 0.5 * float(height))
            dial_radius_px = 0.45 * float(min(height, width))
        else:
            center_xy, dial_radius_px = estimated

        print(f"\n{rel_path} true={true_value:.1f} center={center_xy} radius={dial_radius_px:.1f}")
        for polarity in ("dark", "bright", "either"):
            detection = detect_polar_signed(
                image_bgr,
                center_xy=center_xy,
                dial_radius_px=dial_radius_px,
                gauge_spec=spec,
                polarity=polarity,
            )
            if detection is None:
                print(f"  {polarity:6s}: NONE")
                continue
            predicted = needle_vector_to_value(
                detection.unit_dx,
                detection.unit_dy,
                spec,
            )
            angle_deg = math.degrees(math.atan2(detection.unit_dy, detection.unit_dx))
            print(
                f"  {polarity:6s}: pred={predicted:7.2f} err={abs(predicted - true_value):6.2f} "
                f"angle={angle_deg:7.2f} conf={detection.confidence:7.2f} "
                f"ratio={detection.peak_ratio:6.2f}"
            )


if __name__ == "__main__":
    main()
