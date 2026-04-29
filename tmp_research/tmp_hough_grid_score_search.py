"""Search simple non-ML scoring rules on the Hough-centered fine grid."""

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
    detect_needle_unit_vector,
    needle_detection_quality,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry


REPO_ROOT: Path = Path(__file__).resolve().parent
MANIFEST: Path = REPO_ROOT / "ml" / "data" / "hard_cases.csv"

CENTER_OFFSETS: tuple[int, ...] = tuple(range(-40, 41, 4))
ABS_RADII: tuple[float, ...] = tuple(float(x) / 2.0 for x in range(80, 241, 10))


@dataclass(frozen=True)
class CandidateRecord:
    """One geometry candidate and the detector evidence it produced."""

    candidate: GeometryCandidate
    predicted_value: float
    quality: float


def _build_candidates(image_bgr: np.ndarray) -> list[GeometryCandidate]:
    """Build the Hough-centered fine grid used by the oracle-style sweep."""
    estimated = estimate_dial_geometry(image_bgr)
    if estimated is None:
        return []

    (center_x, center_y), _radius_px = estimated
    candidates: list[GeometryCandidate] = []
    for dx in CENTER_OFFSETS:
        for dy in CENTER_OFFSETS:
            for radius_px in ABS_RADII:
                candidates.append(
                    GeometryCandidate(
                        label=f"hough_{dx:+d}_{dy:+d}_{radius_px:.1f}",
                        center_xy=(center_x + float(dx), center_y + float(dy)),
                        dial_radius_px=radius_px,
                    )
                )
    return candidates


def _evaluate_image(image_path: Path, gauge_id: str) -> list[CandidateRecord]:
    """Run the classical detector over the Hough-centered candidate grid."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return []

    from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs

    spec = load_gauge_specs()[gauge_id]
    records: list[CandidateRecord] = []
    for candidate in _build_candidates(image_bgr):
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
                predicted_value=float(
                    needle_vector_to_value(detection.unit_dx, detection.unit_dy, spec)
                ),
                quality=float(needle_detection_quality(detection)),
            )
        )
    return records


def _score_quality(record: CandidateRecord) -> float:
    """Use the current raw quality score."""
    return record.quality


def _score_log_quality(record: CandidateRecord) -> float:
    """Compress the runaway score with a single log."""
    return math.log1p(max(record.quality, 0.0))


def _score_loglog_quality(record: CandidateRecord) -> float:
    """Compress the runaway score with a double log."""
    return math.log1p(math.log1p(max(record.quality, 0.0)))


def _score_loglog_radius(record: CandidateRecord, *, radius_scale: float) -> float:
    """Compress the score and penalize very large radii."""
    return math.log1p(math.log1p(max(record.quality, 0.0))) / (1.0 + record.candidate.dial_radius_px / radius_scale)


def _score_log_quality_radius(record: CandidateRecord, *, radius_scale: float) -> float:
    """Compress once and penalize very large radii."""
    return math.log1p(max(record.quality, 0.0)) / (1.0 + record.candidate.dial_radius_px / radius_scale)


def _evaluate_manifest(
    manifest_path: Path,
    *,
    gauge_id: str,
    selector: Callable[[list[CandidateRecord]], float],
) -> tuple[int, int, float]:
    """Evaluate one selector on one manifest."""
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
    """Search a handful of score transforms on the hard-case manifest."""
    gauge_id = "littlegood_home_temp_gauge_c"
    selectors: list[tuple[str, Callable[[list[CandidateRecord]], float]]] = [
        ("quality", lambda records: max(records, key=_score_quality).predicted_value),
        ("log_quality", lambda records: max(records, key=_score_log_quality).predicted_value),
        ("loglog_quality", lambda records: max(records, key=_score_loglog_quality).predicted_value),
        ("log_quality_r50", lambda records: max(records, key=lambda record: _score_log_quality_radius(record, radius_scale=50.0)).predicted_value),
        ("log_quality_r80", lambda records: max(records, key=lambda record: _score_log_quality_radius(record, radius_scale=80.0)).predicted_value),
        ("loglog_r50", lambda records: max(records, key=lambda record: _score_loglog_radius(record, radius_scale=50.0)).predicted_value),
        ("loglog_r80", lambda records: max(records, key=lambda record: _score_loglog_radius(record, radius_scale=80.0)).predicted_value),
    ]

    for name, selector in selectors:
        attempted, successful, mae = _evaluate_manifest(
            MANIFEST,
            gauge_id=gauge_id,
            selector=selector,
        )
        print(
            f"{name:16s} attempted={attempted:3d} successful={successful:3d} mae={mae:.4f}"
        )


if __name__ == "__main__":
    main()
