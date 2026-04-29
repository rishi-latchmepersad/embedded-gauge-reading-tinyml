"""Shared manifest-evaluation helpers for the classical CV baseline.

This module centralizes the image loading, geometry selection, and prediction
loop so both the one-off manifest evaluator and the strategy sweep can compare
the same detector variants without duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Literal

import cv2
import numpy as np

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    ClassicalBaselineResult,
    ClassicalPrediction,
    GeometryCandidate,
    GeometrySelection,
    NeedleDetection,
    board_prior_geometry_candidate,
    board_prior_geometry_candidates,
    detect_needle_unit_vector,
    needle_vector_to_value,
    select_best_geometry_detection,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry

GeometryMode = Literal["hough_only", "hough_then_center", "center_only"]


@dataclass(frozen=True)
class GeometryEvaluationConfig:
    """Configuration for one classical baseline geometry strategy."""

    mode: GeometryMode = "hough_then_center"
    confidence_threshold: float = 4.0
    center_radius_scale: float = 0.45


LOCAL_CENTER_OFFSETS_PX: tuple[float, ...] = (-16.0, 0.0, 16.0)
"""Small center shifts that help the spoke voter recover near-miss crops."""

CENTER_RADIUS_SCALES: tuple[float, ...] = (0.35, 0.45, 0.55)
"""Fallback radius scales that span the typical board-capture zoom range."""


@dataclass(frozen=True)
class ManifestEvaluationResult:
    """Result bundle for evaluating one manifest with one geometry strategy."""

    manifest_path: Path
    config: GeometryEvaluationConfig
    attempted_samples: int
    result: ClassicalBaselineResult


def _resolve_image_path(raw_path: str, *, repo_root: Path) -> Path:
    """Resolve a manifest image path relative to the repository root."""
    path: Path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_image(path: Path) -> np.ndarray | None:
    """Load one BGR image for classical CV inference."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _center_fallback_candidates(
    image_bgr: np.ndarray,
    *,
    radius_scale: float,
) -> list[GeometryCandidate]:
    """Build the image-center fallback candidates used by the real-world sweep."""
    height, width = image_bgr.shape[:2]
    image_center: tuple[float, float] = (0.5 * float(width), 0.5 * float(height))
    min_dim: float = float(min(height, width))
    candidate_scales: tuple[float, ...] = tuple(
        sorted({radius_scale, *CENTER_RADIUS_SCALES})
    )
    return [
        GeometryCandidate(
            label=f"image_center_{scale:.2f}",
            center_xy=image_center,
            dial_radius_px=scale * min_dim,
        )
        for scale in candidate_scales
    ]


def _board_prior_candidates(image_bgr: np.ndarray) -> list[GeometryCandidate]:
    """Build the fixed board prior candidate used in the manifest sweep."""
    return board_prior_geometry_candidates(image_bgr)


def _hough_local_candidates(estimated: tuple[tuple[float, float], float]) -> list[GeometryCandidate]:
    """Generate a small local neighborhood around the Hough circle seed."""
    (center_x, center_y), dial_radius_px = estimated
    candidates: list[GeometryCandidate] = [
        GeometryCandidate(
            label="hough",
            center_xy=(center_x, center_y),
            dial_radius_px=dial_radius_px,
        )
    ]
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
    return candidates


def _search_geometry_candidates(
    image_bgr: np.ndarray,
    *,
    config: GeometryEvaluationConfig,
) -> list[GeometryCandidate]:
    """Build the candidate geometry set for one manifest row."""
    estimated = estimate_dial_geometry(image_bgr)
    if config.mode == "center_only":
        return _center_fallback_candidates(image_bgr, radius_scale=config.center_radius_scale)

    if estimated is None:
        if config.mode == "hough_only":
            return []
        candidates: list[GeometryCandidate] = []
        candidates.extend(_board_prior_candidates(image_bgr))
        candidates.extend(
            candidate
            for candidate in _center_fallback_candidates(
                image_bgr,
                radius_scale=config.center_radius_scale,
            )
            if candidate not in candidates
        )
        return candidates

    candidates: list[GeometryCandidate] = []
    if config.mode != "center_only":
        candidates.extend(_hough_local_candidates(estimated))

    if config.mode != "hough_only":
        candidates.extend(_board_prior_candidates(image_bgr))
        candidates.extend(
            candidate
            for candidate in _center_fallback_candidates(
                image_bgr,
                radius_scale=config.center_radius_scale,
            )
            if candidate not in candidates
        )

    return candidates


def _detect_with_geometry_mode(
    image_bgr: np.ndarray,
    *,
    spec: GaugeSpec,
    config: GeometryEvaluationConfig,
) -> NeedleDetection | None:
    """Detect the needle using one of the supported geometry strategies."""
    height, width = image_bgr.shape[:2]
    image_center: tuple[float, float] = (0.5 * float(width), 0.5 * float(height))
    min_dim: float = float(min(height, width))
    fallback_radius_scale: float = min(config.center_radius_scale, 0.35)
    fallback_radius_px: float = fallback_radius_scale * min_dim

    if config.mode == "center_only":
        return detect_needle_unit_vector(
            image_bgr,
            center_xy=image_center,
            dial_radius_px=fallback_radius_px,
            gauge_spec=spec,
        )

    estimated = estimate_dial_geometry(image_bgr)
    if estimated is None:
        if config.mode == "hough_only":
            return None
        selection = select_best_geometry_detection(
            image_bgr,
            candidates=_search_geometry_candidates(image_bgr, config=config),
            gauge_spec=spec,
        )
        if selection is not None:
            return selection.detection
        return detect_needle_unit_vector(
            image_bgr,
            center_xy=image_center,
            dial_radius_px=fallback_radius_px,
            gauge_spec=spec,
        )

    hough_center, hough_radius_px = estimated
    center_offset_px: float = math.hypot(
        hough_center[0] - image_center[0],
        hough_center[1] - image_center[1],
    )
    center_offset_ratio: float = center_offset_px / max(min_dim, 1.0)
    if config.mode == "hough_only":
        if center_offset_ratio > 0.30:
            return None
        return detect_needle_unit_vector(
            image_bgr,
            center_xy=hough_center,
            dial_radius_px=hough_radius_px,
            gauge_spec=spec,
        )

    selection = select_best_geometry_detection(
        image_bgr,
        candidates=_search_geometry_candidates(image_bgr, config=config),
        gauge_spec=spec,
    )
    if selection is not None:
        return selection.detection

    return detect_needle_unit_vector(
        image_bgr,
        center_xy=image_center,
        dial_radius_px=fallback_radius_px,
        gauge_spec=spec,
    )


def evaluate_manifest(
    manifest_path: Path,
    spec: GaugeSpec,
    *,
    config: GeometryEvaluationConfig,
    repo_root: Path,
    max_samples: int | None = None,
) -> ManifestEvaluationResult:
    """Evaluate one CSV manifest with a specific geometry strategy."""
    predictions: list[ClassicalPrediction] = []
    attempted: int = 0

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        required_columns = {"image_path", "value"}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError("Manifest must include image_path and value columns.")

        for row in reader:
            if max_samples is not None and attempted >= max_samples:
                break
            attempted += 1

            image_path = _resolve_image_path(row["image_path"], repo_root=repo_root)
            true_value: float = float(row["value"])
            print(
                f"[BASELINE] Predicting {image_path.name} "
                f"({config.mode}, t={config.confidence_threshold:.1f})...",
                flush=True,
            )

            image_bgr = _load_image(image_path)
            if image_bgr is None:
                print(f"[BASELINE] Skipping unreadable image: {image_path}", flush=True)
                continue

            detection = _detect_with_geometry_mode(
                image_bgr,
                spec=spec,
                config=config,
            )
            if detection is None:
                print(f"[BASELINE] No needle detected for {image_path.name}.", flush=True)
                continue

            predicted_value: float = needle_vector_to_value(
                detection.unit_dx,
                detection.unit_dy,
                spec,
            )
            abs_error: float = abs(predicted_value - true_value)
            predictions.append(
                ClassicalPrediction(
                    image_path=image_path.as_posix(),
                    true_value=true_value,
                    predicted_value=predicted_value,
                    abs_error=abs_error,
                    confidence=detection.confidence,
                )
            )

    successful: int = len(predictions)
    failed: int = attempted - successful
    if successful == 0:
        result = ClassicalBaselineResult(
            attempted_samples=attempted,
            successful_samples=0,
            failed_samples=failed,
            mae=float("nan"),
            rmse=float("nan"),
            predictions=[],
        )
        return ManifestEvaluationResult(
            manifest_path=manifest_path,
            config=config,
            attempted_samples=attempted,
            result=result,
        )

    errors: np.ndarray = np.array([prediction.abs_error for prediction in predictions], dtype=np.float32)
    result = ClassicalBaselineResult(
        attempted_samples=attempted,
        successful_samples=successful,
        failed_samples=failed,
        mae=float(np.mean(errors)),
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
        predictions=predictions,
    )
    return ManifestEvaluationResult(
        manifest_path=manifest_path,
        config=config,
        attempted_samples=attempted,
        result=result,
    )
