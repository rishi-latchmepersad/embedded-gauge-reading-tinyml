"""Shared manifest-evaluation helpers for the classical CV baseline.

This module centralizes the image loading, geometry selection, and prediction
loop so both the one-off manifest evaluator and the strategy sweep can compare
the same detector variants without duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Literal

import cv2
import numpy as np

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    ClassicalBaselineResult,
    ClassicalPrediction,
    GeometryCandidate,
    NeedleDetection,
    detect_needle_unit_vector,
    detect_needle_unit_vector_with_geometry_fallback,
    needle_vector_to_value,
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


def _detect_with_geometry_mode(
    image_bgr: np.ndarray,
    *,
    spec: GaugeSpec,
    config: GeometryEvaluationConfig,
) -> NeedleDetection | None:
    """Detect the needle using one of the supported geometry strategies."""
    height, width = image_bgr.shape[:2]
    center_fallback: tuple[float, float] = (0.5 * float(width), 0.5 * float(height))
    radius_fallback: float = config.center_radius_scale * float(min(height, width))

    if config.mode == "center_only":
        return detect_needle_unit_vector(
            image_bgr,
            center_xy=center_fallback,
            dial_radius_px=radius_fallback,
            gauge_spec=spec,
        )

    estimated = estimate_dial_geometry(image_bgr)
    if estimated is None:
        if config.mode == "hough_only":
            return None
        return detect_needle_unit_vector(
            image_bgr,
            center_xy=center_fallback,
            dial_radius_px=radius_fallback,
            gauge_spec=spec,
        )

    primary = GeometryCandidate(
        label="hough",
        center_xy=estimated[0],
        dial_radius_px=estimated[1],
    )
    secondary = GeometryCandidate(
        label="image_center",
        center_xy=center_fallback,
        dial_radius_px=radius_fallback,
    )

    if config.mode == "hough_only":
        return detect_needle_unit_vector(
            image_bgr,
            center_xy=primary.center_xy,
            dial_radius_px=primary.dial_radius_px,
            gauge_spec=spec,
        )

    return detect_needle_unit_vector_with_geometry_fallback(
        image_bgr,
        primary=primary,
        secondary=secondary,
        gauge_spec=spec,
        confidence_threshold=config.confidence_threshold,
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
