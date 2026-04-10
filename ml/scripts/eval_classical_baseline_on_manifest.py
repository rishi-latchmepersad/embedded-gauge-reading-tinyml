"""Evaluate the classical gauge baseline on a CSV manifest.

This script reuses the single-image geometry estimator and needle detector so we
can benchmark the classical angle pipeline on arbitrary labelled manifests such
as the hard-case CSVs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

# Make the package importable when this script is run from the ml/ directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.baseline_classical_cv import (  # noqa: E402
    ClassicalPrediction,
    detect_needle_unit_vector,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a manifest benchmark run."""
    parser = argparse.ArgumentParser(
        description="Evaluate the classical gauge baseline on a manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--gauge-id",
        type=str,
        default="littlegood_home_temp_gauge_c",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for faster smoke runs.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Optional path for per-sample predictions.",
    )
    return parser.parse_args()


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest path relative to the repository root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT.parent / path


def _load_image(path: Path) -> np.ndarray | None:
    """Load one BGR image for classical CV inference."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _write_predictions_csv(predictions_path: Path, rows: list[ClassicalPrediction]) -> None:
    """Write per-sample predictions to CSV for later inspection."""
    fieldnames: list[str] = [
        "image_path",
        "true_value",
        "predicted_value",
        "abs_error",
        "confidence",
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in rows:
            writer.writerow(
                {
                    "image_path": prediction.image_path,
                    "true_value": prediction.true_value,
                    "predicted_value": prediction.predicted_value,
                    "abs_error": prediction.abs_error,
                    "confidence": prediction.confidence,
                }
            )


def main() -> None:
    """Run the classical baseline on every row in the manifest and summarize errors."""
    args = _parse_args()
    specs: dict[str, GaugeSpec] = load_gauge_specs()
    if args.gauge_id not in specs:
        raise ValueError(f"Unknown gauge_id '{args.gauge_id}'. Available: {list(specs)}")
    spec: GaugeSpec = specs[args.gauge_id]

    predictions: list[ClassicalPrediction] = []
    attempted: int = 0

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {args.manifest}")
        required_columns = {"image_path", "value"}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError("Manifest must include image_path and value columns.")

        for row in reader:
            if args.max_samples is not None and attempted >= args.max_samples:
                break
            attempted += 1

            image_path = _resolve_image_path(row["image_path"])
            true_value = float(row["value"])
            print(f"[BASELINE] Predicting {image_path.name}...", flush=True)

            image_bgr = _load_image(image_path)
            if image_bgr is None:
                print(f"[BASELINE] Skipping unreadable image: {image_path}", flush=True)
                continue

            estimated = estimate_dial_geometry(image_bgr)
            if estimated is None:
                height, width = image_bgr.shape[:2]
                center_xy = (0.5 * float(width), 0.5 * float(height))
                dial_radius_px = 0.45 * float(min(height, width))
            else:
                center_xy, dial_radius_px = estimated

            detection = detect_needle_unit_vector(
                image_bgr,
                center_xy=center_xy,
                dial_radius_px=dial_radius_px,
                gauge_spec=spec,
            )
            if detection is None:
                print(f"[BASELINE] No needle detected for {image_path.name}.", flush=True)
                continue

            predicted_value = needle_vector_to_value(
                detection.unit_dx,
                detection.unit_dy,
                spec,
            )
            abs_error = abs(predicted_value - true_value)
            predictions.append(
                ClassicalPrediction(
                    image_path=image_path.as_posix(),
                    true_value=true_value,
                    predicted_value=predicted_value,
                    abs_error=abs_error,
                    confidence=detection.confidence,
                )
            )
            print(
                f"{image_path.name}: true={true_value:.4f} pred={predicted_value:.4f} "
                f"abs_err={abs_error:.4f}",
                flush=True,
            )

    if not predictions:
        raise ValueError("The classical baseline did not produce any predictions.")

    errors = np.array([prediction.abs_error for prediction in predictions], dtype=np.float32)
    mean_abs_err = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    max_abs_err = float(np.max(errors))
    cases_over_5c = int(np.sum(errors > 5.0))
    worst = max(predictions, key=lambda item: item.abs_error)

    print(f"attempted={attempted}")
    print(f"successful={len(predictions)}")
    print(f"mean_abs_err={mean_abs_err:.4f}")
    print(f"rmse={rmse:.4f}")
    print(f"max_abs_err={max_abs_err:.4f}")
    print(
        f"worst={worst.image_path} true={worst.true_value:.4f} "
        f"pred={worst.predicted_value:.4f} abs_err={worst.abs_error:.4f}"
    )
    print(f"cases_over_5c={cases_over_5c}")

    if args.predictions_csv is not None:
        args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_predictions_csv(args.predictions_csv, predictions)
        print(f"[BASELINE] Wrote predictions to {args.predictions_csv}", flush=True)


if __name__ == "__main__":
    main()
