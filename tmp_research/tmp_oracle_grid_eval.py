"""Evaluate selector heuristics on the exact 5x5x5 oracle geometry grid."""

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


REPO_ROOT: Path = Path(__file__).resolve().parent
MANIFEST: Path = REPO_ROOT / "ml" / "data" / "hard_cases.csv"

CENTER_OFFSETS: tuple[float, ...] = (-32.0, -16.0, 0.0, 16.0, 32.0)
RADIUS_SCALES: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55)


@dataclass(frozen=True)
class CandidateRecord:
    """One geometry candidate and the classical detector output it produced."""

    candidate: GeometryCandidate
    grid_index: tuple[int, int, int]
    predicted_value: float
    quality: float
    line_contrast: float
    dark_fraction: float
    rim_score: float


def _build_candidates(image_bgr: np.ndarray) -> list[tuple[GeometryCandidate, tuple[int, int, int]]]:
    """Build the exact oracle grid around the image center."""
    height, width = image_bgr.shape[:2]
    image_center: tuple[float, float] = (0.5 * float(width), 0.5 * float(height))
    min_dim: float = float(min(height, width))
    candidates: list[tuple[GeometryCandidate, tuple[int, int, int]]] = []
    for dx_index, dx in enumerate(CENTER_OFFSETS):
        for dy_index, dy in enumerate(CENTER_OFFSETS):
            for scale_index, scale in enumerate(RADIUS_SCALES):
                candidates.append(
                    (
                        GeometryCandidate(
                            label=f"grid_{int(dx):+d}_{int(dy):+d}_{scale:.2f}",
                            center_xy=(image_center[0] + dx, image_center[1] + dy),
                            dial_radius_px=scale * min_dim,
                        ),
                        (dx_index, dy_index, scale_index),
                    )
                )
    return candidates


def _evaluate_image(image_path: Path, gauge_id: str) -> list[CandidateRecord]:
    """Run the detector on every candidate geometry for one image."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return []

    spec = load_gauge_specs()[gauge_id]
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    grad_mag_mean = float(np.mean(grad_mag)) + 1e-6
    grad_x_n = grad_x / np.where(grad_mag > 1e-6, grad_mag, 1.0)
    grad_y_n = grad_y / np.where(grad_mag > 1e-6, grad_mag, 1.0)
    records: list[CandidateRecord] = []
    for candidate, grid_index in _build_candidates(image_bgr):
        detection = detect_needle_unit_vector(
            image_bgr,
            center_xy=candidate.center_xy,
            dial_radius_px=candidate.dial_radius_px,
            gauge_spec=spec,
        )
        if detection is None:
            continue
        records.append(
            CandidateRecord(
                candidate=candidate,
                grid_index=grid_index,
                predicted_value=float(
                    needle_vector_to_value(detection.unit_dx, detection.unit_dy, spec)
                ),
                quality=float(needle_detection_quality(detection)),
                line_contrast=float(
                    _sample_line_darkness(
                        gray_image,
                        x1=candidate.center_xy[0]
                        + detection.unit_dx * (0.18 * candidate.dial_radius_px),
                        y1=candidate.center_xy[1]
                        + detection.unit_dy * (0.18 * candidate.dial_radius_px),
                        x2=candidate.center_xy[0]
                        + detection.unit_dx * (0.88 * candidate.dial_radius_px),
                        y2=candidate.center_xy[1]
                        + detection.unit_dy * (0.88 * candidate.dial_radius_px),
                    )[0]
                ),
                dark_fraction=float(
                    _sample_line_darkness(
                        gray_image,
                        x1=candidate.center_xy[0]
                        + detection.unit_dx * (0.18 * candidate.dial_radius_px),
                        y1=candidate.center_xy[1]
                        + detection.unit_dy * (0.18 * candidate.dial_radius_px),
                        x2=candidate.center_xy[0]
                        + detection.unit_dx * (0.88 * candidate.dial_radius_px),
                        y2=candidate.center_xy[1]
                        + detection.unit_dy * (0.88 * candidate.dial_radius_px),
                    )[1]
                ),
                rim_score=float(
                    _rim_score(
                        grad_mag=grad_mag,
                        grad_mag_mean=grad_mag_mean,
                        grad_x_n=grad_x_n,
                        grad_y_n=grad_y_n,
                        candidate=candidate,
                    )
                ),
            )
        )
    return records


def _rim_score(
    *,
    grad_mag: np.ndarray,
    grad_mag_mean: float,
    grad_x_n: np.ndarray,
    grad_y_n: np.ndarray,
    candidate: GeometryCandidate,
) -> float:
    """Score how strongly the candidate geometry matches a circular dial rim."""
    center_x, center_y = candidate.center_xy
    radius_px = candidate.dial_radius_px
    if radius_px <= 1.0:
        return 0.0

    height, width = grad_mag.shape[:2]
    angle_samples = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False, dtype=np.float32)
    radial_offsets = (-4.0, -2.0, 0.0, 2.0, 4.0)
    score_sum = 0.0
    sample_count = 0
    for angle_rad in angle_samples:
        radial_x = math.cos(float(angle_rad))
        radial_y = math.sin(float(angle_rad))
        best_sample = 0.0
        for offset_px in radial_offsets:
            sample_x = center_x + radial_x * (radius_px + offset_px)
            sample_y = center_y + radial_y * (radius_px + offset_px)
            ix = int(round(sample_x))
            iy = int(round(sample_y))
            if ix < 0 or iy < 0 or ix >= width or iy >= height:
                continue
            alignment = abs(float(grad_x_n[iy, ix]) * radial_x + float(grad_y_n[iy, ix]) * radial_y)
            sample_score = float(grad_mag[iy, ix]) * alignment
            if sample_score > best_sample:
                best_sample = sample_score
        score_sum += best_sample
        sample_count += 1

    if sample_count == 0:
        return 0.0
    return (score_sum / float(sample_count)) / grad_mag_mean


def _mean_all(records: list[CandidateRecord]) -> float:
    """Return the plain mean of candidate values."""
    return float(np.mean([record.predicted_value for record in records]))


def _median_all(records: list[CandidateRecord]) -> float:
    """Return the plain median of candidate values."""
    return float(np.median([record.predicted_value for record in records]))


def _weighted_mean(records: list[CandidateRecord]) -> float:
    """Return a quality-weighted mean of candidate values."""
    values = np.asarray([record.predicted_value for record in records], dtype=np.float32)
    weights = np.asarray([max(record.quality, 1e-6) for record in records], dtype=np.float32)
    return float(np.sum(values * weights) / np.sum(weights))


def _trimmed_mean(records: list[CandidateRecord], *, trim_fraction: float = 0.25) -> float:
    """Return a mean after dropping the weakest candidates by quality."""
    ordered = sorted(records, key=lambda record: record.quality, reverse=True)
    keep_count = max(1, int(round(len(ordered) * (1.0 - trim_fraction))))
    values = np.asarray([record.predicted_value for record in ordered[:keep_count]], dtype=np.float32)
    return float(np.mean(values))


def _density_mode(records: list[CandidateRecord], *, sigma_c: float = 4.0) -> float:
    """Return the value with the strongest quality-weighted KDE support."""
    values = np.asarray([record.predicted_value for record in records], dtype=np.float32)
    weights = np.asarray([max(record.quality, 1e-6) for record in records], dtype=np.float32)
    scores = []
    for index, value in enumerate(values):
        kernel = np.exp(-0.5 * np.square((values - value) / sigma_c))
        scores.append(float(weights[index] * np.sum(weights * kernel)))
    best_index = int(np.argmax(np.asarray(scores, dtype=np.float32)))
    return float(values[best_index])


def _rim_mode(records: list[CandidateRecord]) -> float:
    """Return the candidate temperature with the strongest rim score."""
    best_record = max(records, key=lambda record: record.rim_score)
    return float(best_record.predicted_value)


def _rim_quality_mode(records: list[CandidateRecord]) -> float:
    """Return the candidate temperature with the best rim-weighted quality."""
    best_record = max(
        records,
        key=lambda record: record.rim_score * max(record.quality, 1e-6),
    )
    return float(best_record.predicted_value)


def _topk_rim_records(records: list[CandidateRecord], *, top_k: int) -> list[CandidateRecord]:
    """Return the top-k candidates ranked by rim score."""
    ordered = sorted(records, key=lambda record: record.rim_score, reverse=True)
    return ordered[: max(1, min(top_k, len(ordered)))]


def _topk_rim_mean(records: list[CandidateRecord], *, top_k: int) -> float:
    """Return the mean temperature among the best rim candidates."""
    selected = _topk_rim_records(records, top_k=top_k)
    return float(np.mean([record.predicted_value for record in selected]))


def _topk_rim_median(records: list[CandidateRecord], *, top_k: int) -> float:
    """Return the median temperature among the best rim candidates."""
    selected = _topk_rim_records(records, top_k=top_k)
    return float(np.median([record.predicted_value for record in selected]))


def _topk_rim_weighted_mean(records: list[CandidateRecord], *, top_k: int) -> float:
    """Return the quality-weighted mean temperature among the best rim candidates."""
    selected = _topk_rim_records(records, top_k=top_k)
    values = np.asarray([record.predicted_value for record in selected], dtype=np.float32)
    weights = np.asarray([max(record.quality, 1e-6) for record in selected], dtype=np.float32)
    return float(np.sum(values * weights) / np.sum(weights))


def _stability_mode(records: list[CandidateRecord], *, ray_weight: float = 0.0) -> float:
    """Return the candidate sitting inside the smoothest 3x3x3 neighborhood."""
    values = np.asarray([record.predicted_value for record in records], dtype=np.float32)
    scores: list[float] = []
    for index, record in enumerate(records):
        neighbors: list[CandidateRecord] = [
            other
            for other in records
            if max(
                abs(record.grid_index[0] - other.grid_index[0]),
                abs(record.grid_index[1] - other.grid_index[1]),
                abs(record.grid_index[2] - other.grid_index[2]),
            )
            <= 1
        ]
        if len(neighbors) < 4:
            scores.append(float("-inf"))
            continue
        neighbor_values = np.asarray([neighbor.predicted_value for neighbor in neighbors], dtype=np.float32)
        local_std = float(np.std(neighbor_values))
        local_mean_abs = float(np.mean(np.abs(neighbor_values - record.predicted_value)))
        ray_bonus = max(record.line_contrast, 0.0) + 0.75 * record.dark_fraction
        score = record.quality / (1.0 + local_std + 0.5 * local_mean_abs)
        if ray_weight > 0.0:
            score *= 1.0 + ray_weight * ray_bonus
        scores.append(score)
    best_index = int(np.argmax(np.asarray(scores, dtype=np.float32)))
    return float(values[best_index])


def _top_quality(records: list[CandidateRecord]) -> float:
    """Return the highest-quality candidate temperature."""
    return float(max(records, key=lambda record: record.quality).predicted_value)


def _evaluate_manifest(
    manifest_path: Path,
    *,
    gauge_id: str,
    selector: Callable[[list[CandidateRecord]], float],
) -> tuple[int, int, float]:
    """Evaluate one selector on a manifest and return success/MAE stats."""
    errors: list[float] = []
    attempted = 0
    successful = 0

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            attempted += 1
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = REPO_ROOT / image_path
            true_value = float(row["value"])
            records = _evaluate_image(image_path, gauge_id)
            if not records:
                continue
            prediction = selector(records)
            successful += 1
            errors.append(abs(prediction - true_value))

    mae = float(np.mean(np.asarray(errors, dtype=np.float32))) if errors else float("nan")
    return attempted, successful, mae


def main() -> None:
    """Run the exact-grid selector sweep on the hard-case manifest."""
    gauge_id = "littlegood_home_temp_gauge_c"
    selectors: list[tuple[str, Callable[[list[CandidateRecord]], float]]] = [
        ("quality", _top_quality),
        ("rim", _rim_mode),
        ("rim_quality", _rim_quality_mode),
        ("rim_top3_mean", lambda records: _topk_rim_mean(records, top_k=3)),
        ("rim_top5_mean", lambda records: _topk_rim_mean(records, top_k=5)),
        ("rim_top7_mean", lambda records: _topk_rim_mean(records, top_k=7)),
        ("rim_top5_median", lambda records: _topk_rim_median(records, top_k=5)),
        ("rim_top7_median", lambda records: _topk_rim_median(records, top_k=7)),
        ("rim_top5_weighted", lambda records: _topk_rim_weighted_mean(records, top_k=5)),
        ("mean_all", _mean_all),
        ("median_all", _median_all),
        ("weighted_mean", _weighted_mean),
        ("trimmed_mean", lambda records: _trimmed_mean(records, trim_fraction=0.25)),
        ("density_mode", lambda records: _density_mode(records, sigma_c=4.0)),
        ("density_mode_3", lambda records: _density_mode(records, sigma_c=3.0)),
        ("stability", _stability_mode),
        ("stability_ray", lambda records: _stability_mode(records, ray_weight=2.0)),
    ]

    for name, selector in selectors:
        attempted, successful, mae = _evaluate_manifest(
            MANIFEST,
            gauge_id=gauge_id,
            selector=selector,
        )
        print(
            f"{name:14s} attempted={attempted:3d} successful={successful:3d} "
            f"mae={mae:.4f}"
        )


if __name__ == "__main__":
    main()
