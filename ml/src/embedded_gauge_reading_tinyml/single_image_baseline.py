"""Single-image classical baseline runner.

This module runs the classical polar spoke-voting baseline on one camera frame
and optionally writes a small annotated preview plus JSON summary for
inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Final, Literal

import cv2
import numpy as np

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    GeometryCandidate,
    NeedleDetection,
    board_prior_geometry_candidate,
    board_prior_geometry_candidates,
    _detect_needle_unit_vector_center_weighted,
    _detect_needle_unit_vector_shaft_scan,
    _detect_needle_unit_vector_spoke_improved,
    detect_needle_unit_vector,
    needle_detection_quality,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs

ML_ROOT: Path = Path(__file__).resolve().parents[2]
"""Project root resolved from the package location."""

DEFAULT_ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "single_image_baseline"
"""Default folder for single-image baseline artifacts."""

HOUGH_GEOMETRY_MAX_CENTER_OFFSET_RATIO: Final[float] = 0.30
"""Reject Hough circles that drift too far from the image center."""

HOUGH_GEOMETRY_MAX_RADIUS_RATIO: Final[float] = 0.60
"""Reject Hough circles that are too large to be the main dial."""

HOUGH_GEOMETRY_RADIUS_SCALE: Final[float] = 0.75
"""Use a slightly smaller effective radius than the raw Hough circle."""

LOCAL_CENTER_OFFSETS_PX: Final[tuple[float, ...]] = (-16.0, 0.0, 16.0)
"""Small center shifts that help the spoke voter recover near-miss crops."""

CENTER_RADIUS_SCALES: Final[tuple[float, ...]] = (0.35, 0.45, 0.55)
"""Fallback radius scales that span the common board-capture zoom range."""

HOUGH_FINE_CENTER_OFFSETS_PX: Final[tuple[float, ...]] = tuple(
    float(offset) for offset in range(-40, 41, 4)
)
"""Dense Hough-centered offsets used when we need a stronger classical sweep."""

HOUGH_FINE_RADIUS_PX: Final[tuple[float, ...]] = tuple(
    float(radius_px) / 2.0 for radius_px in range(80, 241, 10)
)
"""Absolute radius hypotheses that cover the hard-case camera zoom spread."""

GeometrySearchMode = Literal["hough_first", "auto_sweep"]
"""Supported geometry-selection strategies for the single-image runner."""


@dataclass(frozen=True)
class SingleImageBaselineConfig:
    """Configuration for one single-image classical baseline run."""

    image_path: Path
    gauge_id: str = "littlegood_home_temp_gauge_c"
    center_x: float | None = None
    center_y: float | None = None
    dial_radius_px: float | None = None
    geometry_search_mode: GeometrySearchMode = "hough_first"
    geometry_confidence_threshold: float = 4.0
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    run_name: str = ""


@dataclass(frozen=True)
class SingleImageBaselineResult:
    """Structured output for a single-image classical baseline run."""

    image_path: Path
    gauge_spec: GaugeSpec
    center_xy: tuple[float, float]
    dial_radius_px: float
    detection: NeedleDetection | None
    predicted_value: float | None
    annotated_image_path: Path | None


def _timestamp_run_name() -> str:
    """Build a stable timestamp-based directory name for a fresh run."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _estimate_dial_geometry(
    image_bgr: np.ndarray,
) -> tuple[tuple[float, float], float] | None:
    """Estimate the dial center and radius with a coarse Hough circle search."""
    gray: np.ndarray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred: np.ndarray = cv2.GaussianBlur(gray, (9, 9), 2.0)

    height, width = gray.shape[:2]
    min_radius: int = max(8, int(min(height, width) * 0.18))
    # Allow up to 65% of the shorter dimension so close-up frames where the
    # dial fills most of the image are not rejected by a too-tight upper bound.
    max_radius: int = max(min_radius + 1, int(min(height, width) * 0.65))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, min(height, width) // 4),
        param1=120,
        param2=28,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    candidates: np.ndarray = np.squeeze(circles, axis=0)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        return None

    image_cx: float = 0.5 * width
    image_cy: float = 0.5 * height

    # Prefer circles near the image center and with a large radius.
    #
    # The Hough pass can occasionally lock onto the outer bezel or a strong
    # reflection, so we reject circles that are implausibly far from the frame
    # center or clearly larger than the main gauge face.
    best_score: float = float("-inf")
    best_circle: tuple[float, float, float] | None = None
    max_center_offset: float = HOUGH_GEOMETRY_MAX_CENTER_OFFSET_RATIO * float(
        min(height, width)
    )
    max_radius: float = HOUGH_GEOMETRY_MAX_RADIUS_RATIO * float(min(height, width))
    for cx_f, cy_f, radius_f in candidates:
        cx: float = float(cx_f)
        cy: float = float(cy_f)
        radius: float = float(radius_f)
        center_dist: float = math.hypot(cx - image_cx, cy - image_cy)
        if (center_dist > max_center_offset) or (radius > max_radius):
            continue
        score: float = radius - 0.25 * center_dist
        if score > best_score:
            best_score = score
            best_circle = (cx, cy, radius)

    if best_circle is None:
        return None

    return (best_circle[0], best_circle[1]), best_circle[
        2
    ] * HOUGH_GEOMETRY_RADIUS_SCALE


def _auto_geometry_candidates(image_bgr: np.ndarray) -> list[GeometryCandidate]:
    """Build the candidate geometries used when the image provides no hints."""
    height, width = image_bgr.shape[:2]
    image_center: tuple[float, float] = (0.5 * float(width), 0.5 * float(height))
    min_dim: float = float(min(height, width))
    candidates: list[GeometryCandidate] = []

    estimated = _estimate_dial_geometry(image_bgr)
    if estimated is not None:
        (center_x, center_y), dial_radius_px = estimated
        candidates.append(
            GeometryCandidate(
                label="hough",
                center_xy=(center_x, center_y),
                dial_radius_px=dial_radius_px,
            )
        )
        for dx in LOCAL_CENTER_OFFSETS_PX:
            for dy in LOCAL_CENTER_OFFSETS_PX:
                if dx == 0.0 and dy == 0.0:
                    continue
                candidates.append(
                    GeometryCandidate(
                        label=f"hough_{int(dx):+d}_{int(dy):+d}",
                        center_xy=(center_x + dx, center_y + dy),
                        dial_radius_px=dial_radius_px,
                    )
                )

        # Add a denser local grid around the Hough seed for the hard cases.
        #
        # The stronger geometry search is intentionally more expensive, but it is
        # still classical and it gives the selector access to the near-correct
        # crops that the tiny local sweep often misses.
        for dx in HOUGH_FINE_CENTER_OFFSETS_PX:
            for dy in HOUGH_FINE_CENTER_OFFSETS_PX:
                if dx == 0.0 and dy == 0.0:
                    continue
                for radius_px in HOUGH_FINE_RADIUS_PX:
                    candidates.append(
                        GeometryCandidate(
                            label=f"hough_grid_{int(dx):+d}_{int(dy):+d}_{radius_px:.1f}",
                            center_xy=(center_x + dx, center_y + dy),
                            dial_radius_px=radius_px,
                        )
                    )

    # Add the fixed board prior neighborhood so the sweep can prefer the
    # slightly off-center inner dial even when Hough is missing or noisy.
    candidates.extend(board_prior_geometry_candidates(image_bgr))

    # Always keep a small family of image-center fallbacks around the board's
    # most common zoom levels.
    for radius_scale in sorted({*CENTER_RADIUS_SCALES}):
        candidates.append(
            GeometryCandidate(
                label=f"image_center_{radius_scale:.2f}",
                center_xy=image_center,
                dial_radius_px=radius_scale * min_dim,
            )
        )

    return candidates


def _detect_hough_first_geometry(
    image_bgr: np.ndarray,
    *,
    gauge_spec: GaugeSpec,
    estimated_geometry: tuple[tuple[float, float], float] | None,
    fallback_center_xy: tuple[float, float],
    fallback_radius_px: float,
    confidence_threshold: float,
) -> tuple[GeometryCandidate, tuple[float, float], float, NeedleDetection | None]:
    """Prefer the Hough seed and only fall back to center geometry when needed.

    The auto-sweep path turned out to be brittle on hard captures because a
    wrong offset candidate could override a good Hough seed. The default path
    stays conservative: use the Hough geometry when it is plausible, and only
    try the center fallback when the Hough detection is missing or weak.
    """
    board_prior_candidate = board_prior_geometry_candidate(image_bgr)

    def _board_prior_detection() -> NeedleDetection | None:
        """Run the board-specific shaft scan with a generic fallback."""
        board_scan_detection = _detect_needle_unit_vector_shaft_scan(
            image_bgr,
            center_xy=board_prior_candidate.center_xy,
            dial_radius_px=board_prior_candidate.dial_radius_px,
            gauge_spec=gauge_spec,
        )
        if board_scan_detection is not None and board_scan_detection.peak_ratio >= 1.10:
            return board_scan_detection

        generic_detection = detect_needle_unit_vector(
            image_bgr,
            center_xy=board_prior_candidate.center_xy,
            dial_radius_px=board_prior_candidate.dial_radius_px,
            gauge_spec=gauge_spec,
        )
        if generic_detection is None:
            return board_scan_detection
        if board_scan_detection is None:
            return generic_detection
        return (
            generic_detection
            if needle_detection_quality(generic_detection)
            > needle_detection_quality(board_scan_detection)
            else board_scan_detection
        )

    estimated = estimated_geometry
    if estimated is None:
        estimated = _estimate_dial_geometry(image_bgr)
    if estimated is None:
        board_prior_detection = _board_prior_detection()
        center_candidate = GeometryCandidate(
            label="image_center",
            center_xy=fallback_center_xy,
            dial_radius_px=fallback_radius_px,
        )
        center_detection = detect_needle_unit_vector(
            image_bgr,
            center_xy=fallback_center_xy,
            dial_radius_px=fallback_radius_px,
            gauge_spec=gauge_spec,
        )
        if board_prior_detection is not None:
            return (
                board_prior_candidate,
                board_prior_candidate.center_xy,
                board_prior_candidate.dial_radius_px,
                board_prior_detection,
            )
        if center_detection is not None:
            return center_candidate, fallback_center_xy, fallback_radius_px, center_detection
        return center_candidate, fallback_center_xy, fallback_radius_px, None

    hough_center_xy, hough_radius_px = estimated
    hough_radius_ratio = hough_radius_px / max(board_prior_candidate.dial_radius_px, 1e-6)
    center_offset_px = math.hypot(
        hough_center_xy[0] - fallback_center_xy[0],
        hough_center_xy[1] - fallback_center_xy[1],
    )
    center_offset_ratio = center_offset_px / max(float(min(image_bgr.shape[:2])), 1.0)

    hough_candidate = GeometryCandidate(
        label="hough",
        center_xy=hough_center_xy,
        dial_radius_px=hough_radius_px,
    )
    hough_detection = detect_needle_unit_vector(
        image_bgr,
        center_xy=hough_center_xy,
        dial_radius_px=hough_radius_px,
        gauge_spec=gauge_spec,
    )

    if (
        center_offset_ratio > HOUGH_GEOMETRY_MAX_CENTER_OFFSET_RATIO
        or hough_radius_ratio < 0.80
        or hough_radius_ratio > 1.30
    ):
        board_prior_detection = _board_prior_detection()
        center_candidate = GeometryCandidate(
            label="image_center",
            center_xy=fallback_center_xy,
            dial_radius_px=fallback_radius_px,
        )
        center_detection = detect_needle_unit_vector(
            image_bgr,
            center_xy=fallback_center_xy,
            dial_radius_px=fallback_radius_px,
            gauge_spec=gauge_spec,
        )
        if board_prior_detection is not None:
            return (
                board_prior_candidate,
                board_prior_candidate.center_xy,
                board_prior_candidate.dial_radius_px,
                board_prior_detection,
            )
        if center_detection is not None:
            return center_candidate, fallback_center_xy, fallback_radius_px, center_detection
        return center_candidate, fallback_center_xy, fallback_radius_px, None

    board_prior_detection = _board_prior_detection()
    if board_prior_detection is not None and board_prior_detection.peak_ratio >= 1.10:
        return (
            board_prior_candidate,
            board_prior_candidate.center_xy,
            board_prior_candidate.dial_radius_px,
            board_prior_detection,
        )

    if hough_detection is not None and hough_detection.confidence >= confidence_threshold:
        return hough_candidate, hough_center_xy, hough_radius_px, hough_detection

    center_candidate = GeometryCandidate(
        label="image_center",
        center_xy=fallback_center_xy,
        dial_radius_px=fallback_radius_px,
    )
    center_detection = detect_needle_unit_vector(
        image_bgr,
        center_xy=fallback_center_xy,
        dial_radius_px=fallback_radius_px,
        gauge_spec=gauge_spec,
    )

    if board_prior_detection is not None:
        return (
            board_prior_candidate,
            board_prior_candidate.center_xy,
            board_prior_candidate.dial_radius_px,
            board_prior_detection,
        )
    if center_detection is not None:
        return center_candidate, fallback_center_xy, fallback_radius_px, center_detection
    return center_candidate, fallback_center_xy, fallback_radius_px, None


def estimate_dial_geometry(
    image_bgr: np.ndarray,
) -> tuple[tuple[float, float], float] | None:
    """Public wrapper for the dial geometry estimator used by the baseline.

    The manifest evaluator reuses this helper so the same geometry logic drives
    both the one-off preview mode and the batch benchmark path.
    """
    return _estimate_dial_geometry(image_bgr)


def _draw_annotation(
    image_bgr: np.ndarray,
    *,
    gauge_spec: GaugeSpec,
    center_xy: tuple[float, float],
    dial_radius_px: float,
    detection: NeedleDetection | None,
    output_path: Path,
) -> None:
    """Save a simple annotated preview for human inspection.

    The preview now includes the calibrated sweep start/end rays so we can
    visually confirm the angle-to-value mapping used by the classical baseline.
    """
    annotated: np.ndarray = image_bgr.copy()
    center_x, center_y = center_xy
    center_i: tuple[int, int] = (int(round(center_x)), int(round(center_y)))
    sweep_len: float = dial_radius_px * 0.95

    def point_on_circle(angle_rad: float) -> tuple[int, int]:
        """Project an angle onto the dial circle for annotation."""
        x: int = int(round(center_x + math.cos(angle_rad) * sweep_len))
        y: int = int(round(center_y + math.sin(angle_rad) * sweep_len))
        return (x, y)

    cv2.circle(
        annotated,
        center_i,
        int(round(dial_radius_px)),
        (0, 255, 0),
        2,
    )
    cv2.circle(
        annotated,
        center_i,
        3,
        (0, 0, 255),
        -1,
    )

    # Draw the calibrated sweep span so the preview shows the value geometry.
    min_angle_deg: float = math.degrees(gauge_spec.min_angle_rad) % 360.0
    max_angle_deg: float = (min_angle_deg + math.degrees(gauge_spec.sweep_rad)) % 360.0
    min_angle_rad: float = gauge_spec.min_angle_rad
    max_angle_rad: float = gauge_spec.min_angle_rad + gauge_spec.sweep_rad
    min_tip: tuple[int, int] = point_on_circle(min_angle_rad)
    max_tip: tuple[int, int] = point_on_circle(max_angle_rad)
    cv2.line(annotated, center_i, min_tip, (0, 165, 255), 2)
    cv2.line(annotated, center_i, max_tip, (255, 0, 255), 2)
    cv2.putText(
        annotated,
        f"min={min_angle_deg:.0f} deg",
        (min_tip[0] + 6, min_tip[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"max={max_angle_deg:.0f} deg",
        (max_tip[0] + 6, max_tip[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"sweep={math.degrees(gauge_spec.sweep_rad):.0f} deg",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if detection is not None:
        needle_len: float = dial_radius_px * 0.9
        tip_x: int = int(round(center_x + detection.unit_dx * needle_len))
        tip_y: int = int(round(center_y + detection.unit_dy * needle_len))
        cv2.line(
            annotated,
            (int(round(center_x)), int(round(center_y))),
            (tip_x, tip_y),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"conf={detection.confidence:.1f}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), annotated)


def run_single_image_baseline(
    config: SingleImageBaselineConfig,
) -> SingleImageBaselineResult:
    """Run the classical baseline against one image and save artifacts."""
    run_name: str = config.run_name or _timestamp_run_name()
    run_dir: Path = config.artifacts_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    specs: dict[str, GaugeSpec] = load_gauge_specs()
    if config.gauge_id not in specs:
        raise ValueError(
            f"Unknown gauge_id '{config.gauge_id}'. Available: {list(specs)}"
        )
    spec: GaugeSpec = specs[config.gauge_id]

    image_bgr: np.ndarray | None = cv2.imread(str(config.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image '{config.image_path}'.")

    selected_candidate: GeometryCandidate | None = None
    height, width = image_bgr.shape[:2]
    fallback_center_xy: tuple[float, float] = (
        0.5 * float(width),
        0.5 * float(height),
    )
    fallback_radius_px: float = 0.35 * float(min(height, width))
    estimated_geometry: tuple[tuple[float, float], float] | None = _estimate_dial_geometry(
        image_bgr
    )

    if config.center_x is not None and config.center_y is not None:
        center_xy: tuple[float, float] = (config.center_x, config.center_y)
    elif estimated_geometry is not None:
        center_xy = estimated_geometry[0]
    else:
        center_xy = fallback_center_xy

    if config.dial_radius_px is not None:
        dial_radius_px: float = config.dial_radius_px
    elif estimated_geometry is not None:
        dial_radius_px = estimated_geometry[1]
    else:
        dial_radius_px = fallback_radius_px

    detection: NeedleDetection | None
    if (
        config.center_x is None
        and config.center_y is None
        and config.dial_radius_px is None
    ):
        if config.geometry_search_mode == "auto_sweep":
            # Keep the experimental sweep behind an explicit flag; the default
            # path is Hough-first because the consensus sweep can override a good
            # seed with a worse offset candidate on hard frames.
            auto_candidates = _auto_geometry_candidates(image_bgr)
            selection = select_best_geometry_detection(
                image_bgr,
                candidates=auto_candidates,
                gauge_spec=spec,
                detectors=(
                    _detect_needle_unit_vector_spoke_improved,
                    _detect_needle_unit_vector_center_weighted,
                ),
            )
            if selection is not None:
                selected_candidate = selection.candidate
                center_xy = selected_candidate.center_xy
                dial_radius_px = selected_candidate.dial_radius_px
                detection = selection.detection
            else:
                selected_candidate = GeometryCandidate(
                    label="image_center",
                    center_xy=fallback_center_xy,
                    dial_radius_px=fallback_radius_px,
                )
                center_xy = fallback_center_xy
                dial_radius_px = fallback_radius_px
                detection = detect_needle_unit_vector(
                    image_bgr,
                    center_xy=center_xy,
                    dial_radius_px=dial_radius_px,
                    gauge_spec=spec,
                )
        else:
            (
                selected_candidate,
                center_xy,
                dial_radius_px,
                detection,
            ) = _detect_hough_first_geometry(
                image_bgr,
                gauge_spec=spec,
                estimated_geometry=estimated_geometry,
                fallback_center_xy=fallback_center_xy,
                fallback_radius_px=fallback_radius_px,
                confidence_threshold=config.geometry_confidence_threshold,
            )
    else:
        detection = detect_needle_unit_vector(
            image_bgr,
            center_xy=center_xy,
            dial_radius_px=dial_radius_px,
            gauge_spec=spec,
        )
        if config.center_x is not None and config.center_y is not None:
            selected_candidate = GeometryCandidate(
                label="manual",
                center_xy=center_xy,
                dial_radius_px=dial_radius_px,
            )
    # Filter weak detections: require minimum confidence and peak sharpness.
    # Based on offline testing, low confidence (<5) often correlates with large errors.
    # Peak ratio > 1.2 indicates a clear dominant spoke rather than a near-tie.
    MIN_CONFIDENCE: float = 5.0
    MIN_PEAK_RATIO: float = 1.2
    if selected_candidate is not None and selected_candidate.label.startswith("board_prior"):
        # The board-prior fallback deliberately favors the dark-shaft scan on
        # ideal captures, so it needs a much softer acceptance gate than the
        # generic Hough-first path.
        MIN_CONFIDENCE = 0.25
        MIN_PEAK_RATIO = 1.05

    valid_detection: bool = False
    if detection is not None:
        # Check confidence threshold
        if detection.confidence >= MIN_CONFIDENCE:
            # Check peak sharpness (peak ratio indicates how much stronger the best peak is vs runner-up)
            if detection.peak_ratio >= MIN_PEAK_RATIO:
                valid_detection = True
            else:
                # Peak is not sharp enough - may be a near-tie
                pass
        else:
            # Confidence too low - likely inaccurate
            pass

    predicted_value: float | None = None
    if valid_detection:
        predicted_value = needle_vector_to_value(
            detection.unit_dx,
            detection.unit_dy,
            spec,
        )

    annotated_path: Path = run_dir / f"{config.image_path.stem}_annotated.png"
    _draw_annotation(
        image_bgr,
        gauge_spec=spec,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        detection=detection,
        output_path=annotated_path,
    )

    result = SingleImageBaselineResult(
        image_path=config.image_path,
        gauge_spec=spec,
        center_xy=center_xy,
        dial_radius_px=dial_radius_px,
        detection=detection,
        predicted_value=predicted_value,
        annotated_image_path=annotated_path,
    )

    summary_path: Path = run_dir / "summary.json"
    summary_payload: dict[str, Any] = {
        "config": {
            "image_path": str(config.image_path),
            "gauge_id": config.gauge_id,
            "center_x": config.center_x,
            "center_y": config.center_y,
            "dial_radius_px": config.dial_radius_px,
            "geometry_search_mode": config.geometry_search_mode,
            "geometry_confidence_threshold": config.geometry_confidence_threshold,
            "artifacts_dir": str(config.artifacts_dir),
            "run_name": config.run_name,
        },
        "gauge_spec": asdict(spec),
        "center_xy": [center_xy[0], center_xy[1]],
        "dial_radius_px": dial_radius_px,
        "detection": (
            None
            if detection is None
            else {
                "unit_dx": detection.unit_dx,
                "unit_dy": detection.unit_dy,
                "confidence": detection.confidence,
                "peak_value": detection.peak_value,
                "runner_up_value": detection.runner_up_value,
                "peak_ratio": detection.peak_ratio,
                "peak_margin": detection.peak_margin,
            }
        ),
        "selected_candidate": (
            None
            if selected_candidate is None
            else {
                "label": selected_candidate.label,
                "center_xy": [
                    selected_candidate.center_xy[0],
                    selected_candidate.center_xy[1],
                ],
                "dial_radius_px": selected_candidate.dial_radius_px,
            }
        ),
        "predicted_value": predicted_value,
        "annotated_image_path": str(annotated_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return result
