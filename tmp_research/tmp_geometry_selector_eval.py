"""Explore non-ML geometry-selection heuristics on the hard-case manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Callable

import cv2
import numpy as np

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    GeometryCandidate,
    _sample_line_darkness,
    detect_needle_unit_vector,
    needle_detection_quality,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry


REPO_ROOT: Path = Path(__file__).resolve().parent
ML_ROOT: Path = REPO_ROOT / "ml"
MANIFEST: Path = ML_ROOT / "data" / "hard_cases.csv"

CENTER_OFFSETS: tuple[float, ...] = (-32.0, -16.0, 0.0, 16.0, 32.0)
HOUGH_RADIUS_FACTORS: tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20)
CENTER_RADIUS_FACTORS: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50)


@dataclass(frozen=True)
class CandidateRecord:
    """One geometry candidate plus the detector evidence it produced."""

    family: str
    candidate: GeometryCandidate
    predicted_value: float
    quality: float
    line_contrast: float
    dark_fraction: float


def _candidate_distance(
    left: CandidateRecord,
    right: CandidateRecord,
    *,
    center_scale_px: float,
    radius_scale_px: float,
) -> float:
    """Return a normalized distance between two geometry candidates."""
    left_center_x, left_center_y = left.candidate.center_xy
    right_center_x, right_center_y = right.candidate.center_xy
    center_dist = math.hypot(left_center_x - right_center_x, left_center_y - right_center_y)
    radius_dist = abs(left.candidate.dial_radius_px - right.candidate.dial_radius_px)
    return math.sqrt(
        (center_dist / max(center_scale_px, 1e-6)) ** 2
        + (radius_dist / max(radius_scale_px, 1e-6)) ** 2
    )


def _build_candidates(image_bgr: np.ndarray) -> list[tuple[str, GeometryCandidate]]:
    """Build a broad but still manageable classical geometry candidate set."""
    height, width = image_bgr.shape[:2]
    image_center: tuple[float, float] = (0.5 * float(width), 0.5 * float(height))
    min_dim: float = float(min(height, width))

    candidates: list[tuple[str, GeometryCandidate]] = []
    estimated = estimate_dial_geometry(image_bgr)
    if estimated is not None:
        (center_x, center_y), radius_px = estimated
        for dx in CENTER_OFFSETS:
            for dy in CENTER_OFFSETS:
                for scale in HOUGH_RADIUS_FACTORS:
                    candidates.append(
                        (
                            "hough",
                            GeometryCandidate(
                                label=f"hough_{int(dx):+d}_{int(dy):+d}_{scale:.2f}",
                                center_xy=(center_x + dx, center_y + dy),
                                dial_radius_px=radius_px * scale,
                            ),
                        )
                    )

    for dx in CENTER_OFFSETS:
        for dy in CENTER_OFFSETS:
            for scale in CENTER_RADIUS_FACTORS:
                candidates.append(
                    (
                        "center",
                        GeometryCandidate(
                            label=f"center_{int(dx):+d}_{int(dy):+d}_{scale:.2f}",
                            center_xy=(image_center[0] + dx, image_center[1] + dy),
                            dial_radius_px=scale * min_dim,
                        ),
                    )
                )

    return candidates


def _ray_darkness(
    gray_image: np.ndarray,
    *,
    center_xy: tuple[float, float],
    unit_dx: float,
    unit_dy: float,
    dial_radius_px: float,
) -> tuple[float, float]:
    """Score how needle-like the detected ray looks in grayscale space."""
    center_x, center_y = center_xy
    x1 = center_x + unit_dx * (0.18 * dial_radius_px)
    y1 = center_y + unit_dy * (0.18 * dial_radius_px)
    x2 = center_x + unit_dx * (0.88 * dial_radius_px)
    y2 = center_y + unit_dy * (0.88 * dial_radius_px)
    contrast_mean, dark_fraction = _sample_line_darkness(
        gray_image,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
    return float(contrast_mean), float(dark_fraction)


def _evaluate_image(image_path: Path, true_value: float) -> list[CandidateRecord]:
    """Run the detector over all geometry candidates for one image."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return []

    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    records: list[CandidateRecord] = []

    for family, candidate in _build_candidates(image_bgr):
        detection = detect_needle_unit_vector(
            image_bgr,
            center_xy=candidate.center_xy,
            dial_radius_px=candidate.dial_radius_px,
            gauge_spec=spec,
        )
        if detection is None:
            continue

        predicted_value = float(
            needle_vector_to_value(detection.unit_dx, detection.unit_dy, spec)
        )
        quality = float(needle_detection_quality(detection))
        line_contrast, dark_fraction = _ray_darkness(
            gray_image,
            center_xy=candidate.center_xy,
            unit_dx=detection.unit_dx,
            unit_dy=detection.unit_dy,
            dial_radius_px=candidate.dial_radius_px,
        )
        records.append(
            CandidateRecord(
                family=family,
                candidate=candidate,
                predicted_value=predicted_value,
                quality=quality,
                line_contrast=line_contrast,
                dark_fraction=dark_fraction,
            )
        )

    return records


def _score_quality_only(record: CandidateRecord) -> float:
    """Baseline score: prefer the raw detector quality."""
    return record.quality


def _mean_value(records: list[CandidateRecord]) -> float:
    """Return the simple mean of all candidate temperatures."""
    values = np.asarray([record.predicted_value for record in records], dtype=np.float32)
    return float(np.mean(values))


def _median_value(records: list[CandidateRecord]) -> float:
    """Return the simple median of all candidate temperatures."""
    values = np.asarray([record.predicted_value for record in records], dtype=np.float32)
    return float(np.median(values))


def _weighted_mean_value(records: list[CandidateRecord]) -> float:
    """Return the quality-weighted mean temperature."""
    values = np.asarray([record.predicted_value for record in records], dtype=np.float32)
    weights = np.asarray([max(record.quality, 1e-6) for record in records], dtype=np.float32)
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_median_value(records: list[CandidateRecord]) -> float:
    """Return the quality-weighted median temperature."""
    pairs = sorted(
        ((record.predicted_value, max(record.quality, 1e-6)) for record in records),
        key=lambda item: item[0],
    )
    total_weight = sum(weight for _value, weight in pairs)
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= 0.5 * total_weight:
            return float(value)
    return float(pairs[-1][0])


def _trimmed_mean_value(records: list[CandidateRecord], *, trim_fraction: float = 0.25) -> float:
    """Return a mean after trimming the weakest candidates by quality."""
    ordered = sorted(records, key=lambda record: record.quality, reverse=True)
    keep_count = max(1, int(round(len(ordered) * (1.0 - trim_fraction))))
    values = np.asarray([record.predicted_value for record in ordered[:keep_count]], dtype=np.float32)
    return float(np.mean(values))


def _density_mode_value(records: list[CandidateRecord], *, sigma_c: float = 4.0) -> float:
    """Return the candidate temperature with the strongest KDE support."""
    values = np.asarray([record.predicted_value for record in records], dtype=np.float32)
    weights = np.asarray([max(record.quality, 1e-6) for record in records], dtype=np.float32)
    scores = []
    for index, value in enumerate(values):
        kernel = np.exp(-0.5 * np.square((values - value) / sigma_c))
        scores.append(float(weights[index] * np.sum(weights * kernel)))
    best_index = int(np.argmax(np.asarray(scores, dtype=np.float32)))
    return float(values[best_index])


def _stability_value(
    records: list[CandidateRecord],
    *,
    center_scale_px: float,
    radius_scale_px: float,
    neighborhood_threshold: float = 1.0,
    min_neighbors: int = 4,
) -> float:
    """Return the candidate with the smoothest local temperature neighborhood."""
    scores: list[float] = []
    for index, record in enumerate(records):
        neighbors: list[CandidateRecord] = [
            other
            for other in records
            if _candidate_distance(
                record,
                other,
                center_scale_px=center_scale_px,
                radius_scale_px=radius_scale_px,
            )
            <= neighborhood_threshold
        ]
        if len(neighbors) < min_neighbors:
            scores.append(float("-inf"))
            continue
        neighbor_values = np.asarray([neighbor.predicted_value for neighbor in neighbors], dtype=np.float32)
        local_std = float(np.std(neighbor_values))
        local_mean_abs = float(np.mean(np.abs(neighbor_values - record.predicted_value)))
        stability = 1.0 / (1.0 + local_std + 0.5 * local_mean_abs)
        score = record.quality * stability
        scores.append(score)
    best_index = int(np.argmax(np.asarray(scores, dtype=np.float32)))
    return float(records[best_index].predicted_value)


def main() -> None:
    """Run a small selection heuristic sweep on the hard-case manifest."""
    load_gauge_specs()["littlegood_home_temp_gauge_c"]
    rows: list[tuple[float, list[CandidateRecord]]] = []

    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = REPO_ROOT / image_path
            true_value = float(row["value"])
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            records = _evaluate_image(image_path, true_value)
            if records:
                rows.append((true_value, records))

    strategies: list[tuple[str, Callable[[list[CandidateRecord]], float]]] = [
        ("quality", lambda records: records[int(np.argmax([record.quality for record in records]))].predicted_value),
        ("mean_all", _mean_value),
        ("median_all", _median_value),
        ("weighted_mean", _weighted_mean_value),
        ("weighted_median", _weighted_median_value),
        ("trimmed_mean", lambda records: _trimmed_mean_value(records, trim_fraction=0.25)),
        ("density_mode", lambda records: _density_mode_value(records, sigma_c=4.0)),
        (
            "stability",
            lambda records: _stability_value(
                records,
                center_scale_px=24.0,
                radius_scale_px=12.0,
                neighborhood_threshold=1.0,
                min_neighbors=4,
            ),
        ),
        (
            "stability_wide",
            lambda records: _stability_value(
                records,
                center_scale_px=32.0,
                radius_scale_px=16.0,
                neighborhood_threshold=1.15,
                min_neighbors=4,
            ),
        ),
    ]

    for name, fn in strategies:
        errors: list[float] = []
        for true_value, records in rows:
            prediction = fn(records)
            errors.append(abs(prediction - true_value))
        error_array = np.asarray(errors, dtype=np.float32)
        print(
            f"{name:14s} "
            f"mae={float(np.mean(error_array)):.4f} "
            f"rmse={float(np.sqrt(np.mean(np.square(error_array)))):.4f} "
            f"over5={int(np.sum(error_array > 5.0))}/{len(error_array)}"
        )


if __name__ == "__main__":
    main()
