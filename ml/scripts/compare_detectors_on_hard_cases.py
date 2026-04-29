"""Compare spoke-vote vs polar detectors on hard case manifests.

Evaluates both detectors on all hard case CSVs and produces a side-by-side
comparison with per-sample predictions and aggregate metrics.

Usage:
    poetry run python -u scripts/compare_detectors_on_hard_cases.py
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    ClassicalPrediction,
    detect_needle_unit_vector,
    _detect_needle_unit_vector_polar,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry

REPO_ROOT = PROJECT_ROOT.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "detector_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]

MANIFESTS = [
    ("hard_cases.csv", PROJECT_ROOT / "data" / "hard_cases.csv"),
    (
        "hard_cases_remaining_focus.csv",
        PROJECT_ROOT / "data" / "hard_cases_remaining_focus.csv",
    ),
]


@dataclass
class DetectorResult:
    """Result from one detector on one image."""

    image_name: str
    true_value: float
    spoke_angle: float | None
    spoke_pred: float | None
    spoke_conf: float | None
    spoke_ratio: float | None
    polar_angle: float | None
    polar_pred: float | None
    polar_conf: float | None
    polar_ratio: float | None


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest image path relative to the repository root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def evaluate_manifest(manifest_path: Path, manifest_name: str) -> list[DetectorResult]:
    """Evaluate both detectors on every image in a manifest."""
    results: list[DetectorResult] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = _resolve_image_path(row["image_path"])
            true_value = float(row["value"])

            if not image_path.exists():
                print(f"  SKIP {image_path.name} (not found)")
                continue

            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  SKIP {image_path.name} (unreadable)")
                continue

            h, w = img.shape[:2]
            estimated = estimate_dial_geometry(img)
            if estimated is not None:
                (cx, cy), radius = estimated
            else:
                cx, cy = 0.5 * w, 0.5 * h
                radius = 0.45 * min(h, w)

            # Spoke-vote detector
            spoke_det = detect_needle_unit_vector(
                img, center_xy=(cx, cy), dial_radius_px=radius, gauge_spec=spec
            )
            if spoke_det is not None:
                spoke_angle = math.degrees(
                    math.atan2(spoke_det.unit_dy, spoke_det.unit_dx)
                )
                spoke_pred = needle_vector_to_value(
                    spoke_det.unit_dx, spoke_det.unit_dy, spec
                )
                spoke_conf = spoke_det.confidence
                spoke_ratio = spoke_det.peak_ratio
            else:
                spoke_angle = spoke_pred = spoke_conf = spoke_ratio = None

            # Polar detector
            polar_det = _detect_needle_unit_vector_polar(
                img,
                center_xy=(cx, cy),
                dial_radius_px=radius,
                angle_bounds_rad=(spec.min_angle_rad, spec.sweep_rad),
            )
            if polar_det is not None:
                polar_angle = math.degrees(
                    math.atan2(polar_det.unit_dy, polar_det.unit_dx)
                )
                polar_pred = needle_vector_to_value(
                    polar_det.unit_dx, polar_det.unit_dy, spec
                )
                polar_conf = polar_det.confidence
                polar_ratio = polar_det.peak_ratio
            else:
                polar_angle = polar_pred = polar_conf = polar_ratio = None

            results.append(
                DetectorResult(
                    image_name=image_path.name,
                    true_value=true_value,
                    spoke_angle=spoke_angle,
                    spoke_pred=spoke_pred,
                    spoke_conf=spoke_conf,
                    spoke_ratio=spoke_ratio,
                    polar_angle=polar_angle,
                    polar_pred=polar_pred,
                    polar_conf=polar_conf,
                    polar_ratio=polar_ratio,
                )
            )

            # Print progress
            spoke_str = f"{spoke_pred:7.2f}°C" if spoke_pred is not None else "  NONE  "
            polar_str = f"{polar_pred:7.2f}°C" if polar_pred is not None else "  NONE  "
            print(
                f"  {image_path.name:45s} true={true_value:6.1f}  "
                f"spoke={spoke_str}  polar={polar_str}"
            )

    return results


def print_summary(results: list[DetectorResult], label: str) -> None:
    """Print aggregate metrics for a set of results."""
    spoke_errors = [
        abs(r.spoke_pred - r.true_value) for r in results if r.spoke_pred is not None
    ]
    polar_errors = [
        abs(r.polar_pred - r.true_value) for r in results if r.polar_pred is not None
    ]

    spoke_mae = np.mean(spoke_errors) if spoke_errors else float("nan")
    spoke_max = max(spoke_errors) if spoke_errors else float("nan")
    polar_mae = np.mean(polar_errors) if polar_errors else float("nan")
    polar_max = max(polar_errors) if polar_errors else float("nan")

    spoke_over_5 = sum(1 for e in spoke_errors if e > 5.0)
    polar_over_5 = sum(1 for e in polar_errors if e > 5.0)

    print(f"\n  {'':>20s}  {'Spoke-vote':>14s}  {'Polar':>14s}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*14}")
    print(f"  {'Attempted':>20s}  {len(results):>14d}  {len(results):>14d}")
    print(f"  {'Successful':>20s}  {len(spoke_errors):>14d}  {len(polar_errors):>14d}")
    print(
        f"  {'Failed':>20s}  {len(results) - len(spoke_errors):>14d}  {len(results) - len(polar_errors):>14d}"
    )
    print(f"  {'MAE (°C)':>20s}  {spoke_mae:>14.4f}  {polar_mae:>14.4f}")
    print(f"  {'Max error (°C)':>20s}  {spoke_max:>14.4f}  {polar_max:>14.4f}")
    print(f"  {'Cases >5°C':>20s}  {spoke_over_5:>14d}  {polar_over_5:>14d}")

    # Find worst cases for each detector
    if spoke_errors:
        worst_spoke = max(
            results,
            key=lambda r: (
                abs(r.spoke_pred - r.true_value) if r.spoke_pred is not None else -1
            ),
        )
        print(
            f"\n  Worst spoke-vote: {worst_spoke.image_name} "
            f"true={worst_spoke.true_value:.1f} pred={worst_spoke.spoke_pred:.2f} "
            f"err={abs(worst_spoke.spoke_pred - worst_spoke.true_value):.2f}"
        )
    if polar_errors:
        worst_polar = max(
            results,
            key=lambda r: (
                abs(r.polar_pred - r.true_value) if r.polar_pred is not None else -1
            ),
        )
        print(
            f"  Worst polar:     {worst_polar.image_name} "
            f"true={worst_polar.true_value:.1f} pred={worst_polar.polar_pred:.2f} "
            f"err={abs(worst_polar.polar_pred - worst_polar.true_value):.2f}"
        )


def write_detailed_csv(results: list[DetectorResult], path: Path) -> None:
    """Write per-sample results to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "true_value",
                "spoke_angle",
                "spoke_pred",
                "spoke_conf",
                "spoke_ratio",
                "spoke_error",
                "polar_angle",
                "polar_pred",
                "polar_conf",
                "polar_ratio",
                "polar_error",
            ]
        )
        for r in results:
            spoke_err = (
                abs(r.spoke_pred - r.true_value) if r.spoke_pred is not None else ""
            )
            polar_err = (
                abs(r.polar_pred - r.true_value) if r.polar_pred is not None else ""
            )
            writer.writerow(
                [
                    r.image_name,
                    f"{r.true_value:.1f}",
                    f"{r.spoke_angle:.2f}" if r.spoke_angle is not None else "NONE",
                    f"{r.spoke_pred:.4f}" if r.spoke_pred is not None else "NONE",
                    f"{r.spoke_conf:.4f}" if r.spoke_conf is not None else "NONE",
                    f"{r.spoke_ratio:.4f}" if r.spoke_ratio is not None else "NONE",
                    f"{spoke_err:.4f}" if spoke_err != "" else "NONE",
                    f"{r.polar_angle:.2f}" if r.polar_angle is not None else "NONE",
                    f"{r.polar_pred:.4f}" if r.polar_pred is not None else "NONE",
                    f"{r.polar_conf:.4f}" if r.polar_conf is not None else "NONE",
                    f"{r.polar_ratio:.4f}" if r.polar_ratio is not None else "NONE",
                    f"{polar_err:.4f}" if polar_err != "" else "NONE",
                ]
            )


def main() -> None:
    """Evaluate both detectors on all hard case manifests."""
    print("=" * 72)
    print("DETECTOR COMPARISON: Spoke-vote vs Polar")
    print("=" * 72)
    print(f"Gauge: {spec}")
    print()

    for manifest_name, manifest_path in MANIFESTS:
        print(f"\n{'=' * 72}")
        print(f"MANIFEST: {manifest_name}")
        print(f"{'=' * 72}")
        results = evaluate_manifest(manifest_path, manifest_name)
        print_summary(results, manifest_name)

        # Write detailed CSV
        csv_path = OUT_DIR / f"{manifest_name.replace('.csv', '')}_comparison.csv"
        write_detailed_csv(results, csv_path)
        print(f"\n  Detailed results: {csv_path}")

    # Combined analysis
    print(f"\n{'=' * 72}")
    print("COMBINED ANALYSIS")
    print(f"{'=' * 72}")
    all_results: list[DetectorResult] = []
    for _, manifest_path in MANIFESTS:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = _resolve_image_path(row["image_path"])
                if image_path.exists():
                    all_results.append(
                        DetectorResult(
                            image_name=image_path.name,
                            true_value=float(row["value"]),
                            spoke_angle=None,
                            spoke_pred=None,
                            spoke_conf=None,
                            spoke_ratio=None,
                            polar_angle=None,
                            polar_pred=None,
                            polar_conf=None,
                            polar_ratio=None,
                        )
                    )

    # Re-evaluate for combined (we already have results from above)
    # Just print combined summary from the per-manifest results
    print("\nSee per-manifest summaries above for combined picture.")

    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
