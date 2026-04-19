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

# Make the package importable when this script is run from the ml/ directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.baseline_classical_cv import ClassicalPrediction  # noqa: E402
from embedded_gauge_reading_tinyml.baseline_report import (  # noqa: E402
    write_failure_report,
)
from embedded_gauge_reading_tinyml.baseline_manifest_eval import (  # noqa: E402
    GeometryEvaluationConfig,
    ManifestEvaluationResult,
    evaluate_manifest,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs


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
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Optional output directory for ranked failure-analysis artifacts.",
    )
    parser.add_argument(
        "--geometry-confidence-threshold",
        type=float,
        default=4.0,
        help="Prefer the Hough geometry unless its detection confidence is below this value.",
    )
    parser.add_argument(
        "--geometry-mode",
        type=str,
        default="hough_then_center",
        choices=("hough_only", "hough_then_center", "center_only"),
        help="Geometry strategy to use for this manifest run.",
    )
    parser.add_argument(
        "--center-radius-scale",
        type=float,
        default=0.45,
        help="Radius scale for the image-center fallback geometry.",
    )
    return parser.parse_args()


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

    manifest_result: ManifestEvaluationResult = evaluate_manifest(
        args.manifest,
        spec,
        config=GeometryEvaluationConfig(
            mode=args.geometry_mode,
            confidence_threshold=args.geometry_confidence_threshold,
            center_radius_scale=args.center_radius_scale,
        ),
        repo_root=PROJECT_ROOT.parent,
        max_samples=args.max_samples,
    )
    predictions: list[ClassicalPrediction] = manifest_result.result.predictions

    if not predictions:
        raise ValueError("The classical baseline did not produce any predictions.")

    mean_abs_err: float = manifest_result.result.mae
    rmse: float = manifest_result.result.rmse
    max_abs_err = max(prediction.abs_error for prediction in predictions)
    cases_over_5c = sum(prediction.abs_error > 5.0 for prediction in predictions)
    worst = max(predictions, key=lambda item: item.abs_error)

    print(f"attempted={manifest_result.attempted_samples}")
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

    if args.report_dir is not None:
        report = write_failure_report(
            args.report_dir,
            predictions,
            attempted_samples=manifest_result.attempted_samples,
            top_n=10,
            value_bucket_size=10.0,
        )
        print(
            f"[BASELINE] Wrote failure report to {args.report_dir} "
            f"(mae={report.mae:.4f}, rmse={report.rmse:.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
