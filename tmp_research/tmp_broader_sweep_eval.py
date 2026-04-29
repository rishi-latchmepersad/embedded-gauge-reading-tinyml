"""Evaluate a broader geometry candidate sweep on the hard-case manifests."""

from __future__ import annotations

import csv
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
    GeometryCandidate,
    needle_vector_to_value,
    select_best_geometry_detection,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs  # noqa: E402
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry  # noqa: E402


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest path relative to the repository root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _broader_candidates(
    image_bgr: np.ndarray,
) -> list[GeometryCandidate]:
    """Generate a wider geometry family than the current default sweep."""
    height, width = image_bgr.shape[:2]
    image_center = (0.5 * float(width), 0.5 * float(height))
    min_dim = float(min(height, width))
    candidates: list[GeometryCandidate] = []

    estimated = estimate_dial_geometry(image_bgr)
    if estimated is not None:
        (center_x, center_y), dial_radius_px = estimated
        for dx in (-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0):
            for dy in (-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0):
                candidates.append(
                    GeometryCandidate(
                        label=f"hough_{int(dx):+d}_{int(dy):+d}",
                        center_xy=(center_x + dx, center_y + dy),
                        dial_radius_px=dial_radius_px,
                    )
                )

    # Keep a much wider image-center fallback family for the dark board crops.
    for center_dx in (-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0):
        for center_dy in (-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0):
            center_xy = (image_center[0] + center_dx, image_center[1] + center_dy)
            for radius_scale in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
                candidates.append(
                    GeometryCandidate(
                        label=f"image_{int(center_dx):+d}_{int(center_dy):+d}_{radius_scale:.2f}",
                        center_xy=center_xy,
                        dial_radius_px=radius_scale * min_dim,
                    )
                )

    return candidates


def main() -> None:
    """Evaluate the broader geometry search on the hard manifests."""
    spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    manifests = [
        REPO_ROOT / "ml" / "data" / "hard_cases.csv",
        REPO_ROOT / "ml" / "data" / "hard_cases_plus_board30_valid_with_new5.csv",
        REPO_ROOT / "ml" / "data" / "hard_cases_plus_board30_valid_with_new5_closeup14c.csv",
    ]

    for manifest_path in manifests:
        errors: list[float] = []
        attempted = 0
        successful = 0
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                attempted += 1
                image_path = _resolve_image_path(row["image_path"])
                true_value = float(row["value"])
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    continue

                selection = select_best_geometry_detection(
                    image_bgr,
                    candidates=_broader_candidates(image_bgr),
                    gauge_spec=spec,
                )
                if selection is None:
                    continue

                successful += 1
                predicted_value = needle_vector_to_value(
                    selection.detection.unit_dx,
                    selection.detection.unit_dy,
                    spec,
                )
                errors.append(abs(predicted_value - true_value))

        arr = np.asarray(errors, dtype=np.float32) if errors else np.asarray([], dtype=np.float32)
        mae = float(np.mean(arr)) if arr.size else float("nan")
        rmse = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else float("nan")
        print(
            f"{manifest_path.name}: attempted={attempted} successful={successful} "
            f"failed={attempted - successful} mae={mae:.4f} rmse={rmse:.4f}"
        )


if __name__ == "__main__":
    main()
