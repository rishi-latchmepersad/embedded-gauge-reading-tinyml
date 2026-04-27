"""Compare board replay and classical CV on the hard-case manifest.

This script evaluates the same labeled captures with:
- the pure classical baseline,
- the raw scalar CNN from the board replay,
- the calibrated scalar CNN from the board replay, and
- the full board-pipeline replay result after burst smoothing.

The goal is to keep all four numbers on the same manifest so we can see where
the board path diverges from the offline tests.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import sys
import time
from typing import Final

import numpy as np

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_MANIFEST: Final[Path] = PROJECT_ROOT / "data" / "hard_cases_remaining_focus.csv"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "artifacts" / "board_pipeline_hardcase_compare"
DEFAULT_OBB_MODEL: Final[Path] = (
    PROJECT_ROOT / "artifacts" / "deployment" / "prod_model_v0.3_obb_int8" / "model_int8.tflite"
)
DEFAULT_RECTIFIER_MODEL: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "mobilenetv2_rectifier_hardcase_finetune_v3_int8"
    / "model_int8.tflite"
)
DEFAULT_SCALAR_MODEL: Final[Path] = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "scalar_full_finetune_from_best_piecewise_calibrated_int8"
    / "model_int8.tflite"
)


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One manifest row with classical and board-pipeline predictions."""

    image_path: str
    true_value: float
    classical_pred: float | None
    classical_abs_err: float | None
    classical_confidence: float | None
    board_raw_pred: float | None
    board_raw_abs_err: float | None
    board_calibrated_pred: float | None
    board_calibrated_abs_err: float | None
    board_reported_pred: float | None
    board_reported_abs_err: float | None
    board_selected_stage: str | None
    board_selected_crop_x0: float | None
    board_selected_crop_y0: float | None
    board_selected_crop_x1: float | None
    board_selected_crop_y1: float | None


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the hard-case comparison run."""
    parser = argparse.ArgumentParser(
        description="Compare the hard-case manifest across board and classical modes."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV manifest with image_path,value rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the comparison artifacts should be written.",
    )
    parser.add_argument(
        "--obb-model",
        type=Path,
        default=DEFAULT_OBB_MODEL,
        help="Path to the deployed OBB localizer.",
    )
    parser.add_argument(
        "--rectifier-model",
        type=Path,
        default=DEFAULT_RECTIFIER_MODEL,
        help="Path to the deployed rectifier fallback.",
    )
    parser.add_argument(
        "--scalar-model",
        type=Path,
        default=DEFAULT_SCALAR_MODEL,
        help="Path to the deployed scalar CNN.",
    )
    parser.add_argument(
        "--classical-geometry-mode",
        type=str,
        default="hough_then_center",
        choices=("hough_only", "hough_then_center", "center_only"),
        help="Geometry mode used by the pure classical baseline.",
    )
    parser.add_argument(
        "--classical-confidence-threshold",
        type=float,
        default=4.0,
        help="Geometry fallback confidence threshold for the classical baseline.",
    )
    parser.add_argument(
        "--classical-center-radius-scale",
        type=float,
        default=0.45,
        help="Fallback radius scale for the classical baseline center geometry.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=3,
        help="Size of the firmware-style burst history used by board replay.",
    )
    parser.add_argument(
        "--history-reset-delta",
        type=float,
        default=12.0,
        help="Reset the burst history when values jump by more than this amount.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for faster smoke runs.",
    )
    parser.add_argument(
        "--trace-stages",
        action="store_true",
        help="Print stage-by-stage progress for the board-pipeline replay.",
    )
    return parser.parse_args()


def _resolve_manifest_path(raw_path: str) -> Path:
    """Resolve a manifest image path relative to the repository root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_manifest(manifest_path: Path, *, max_samples: int | None = None) -> list[tuple[Path, float]]:
    """Load the hard-case manifest as resolved image/value pairs."""
    items: list[tuple[Path, float]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if max_samples is not None and len(items) >= max_samples:
                break
            items.append((_resolve_manifest_path(row["image_path"]), float(row["value"])))
    return items


def _metrics(values: list[float]) -> dict[str, float | int]:
    """Summarize one list of absolute errors."""
    if not values:
        return {"successful": 0, "mae": float("nan"), "rmse": float("nan"), "max_abs_err": float("nan"), "cases_over_5c": 0}
    errors = np.asarray(values, dtype=np.float32)
    return {
        "successful": int(errors.size),
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "max_abs_err": float(np.max(errors)),
        "cases_over_5c": int(np.sum(errors > 5.0)),
    }


def _write_csv(path: Path, rows: list[ComparisonRow]) -> None:
    """Write the per-sample comparison table to CSV."""
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: ("" if value is None else value) for key, value in asdict(row).items()}
            writer.writerow(payload)


def main() -> None:
    """Run the hard-case comparison and write summary artifacts."""
    args = _parse_args()
    items = _load_manifest(args.manifest, max_samples=args.max_samples)
    if not items:
        raise FileNotFoundError(f"No rows loaded from {args.manifest}")

    print("[COMPARE] Importing board pipeline module...", flush=True)
    import_start = time.perf_counter()
    from embedded_gauge_reading_tinyml.board_pipeline import (  # noqa: E402
        InferenceBurstHistory,
        load_model_session,
        predict_board_pipeline_on_capture,
    )
    from embedded_gauge_reading_tinyml.baseline_manifest_eval import (  # noqa: E402
        GeometryEvaluationConfig,
        evaluate_manifest,
    )
    from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs  # noqa: E402

    import_elapsed = time.perf_counter() - import_start
    print(f"[COMPARE] Imported board modules in {import_elapsed:.3f}s", flush=True)

    specs = load_gauge_specs()
    gauge_id = "littlegood_home_temp_gauge_c"
    if gauge_id not in specs:
        raise ValueError(f"Unknown gauge_id '{gauge_id}'. Available: {list(specs)}")
    gauge_spec = specs[gauge_id]

    print("[COMPARE] Evaluating classical baseline on manifest...", flush=True)
    classical_result = evaluate_manifest(
        args.manifest,
        gauge_spec,
        config=GeometryEvaluationConfig(
            mode=args.classical_geometry_mode,
            confidence_threshold=args.classical_confidence_threshold,
            center_radius_scale=args.classical_center_radius_scale,
        ),
        repo_root=REPO_ROOT,
        max_samples=args.max_samples,
    )
    classical_by_path = {
        prediction.image_path: prediction for prediction in classical_result.result.predictions
    }

    print("[COMPARE] Loading board models...", flush=True)
    obb_session = load_model_session(args.obb_model, "auto")
    rectifier_session = load_model_session(args.rectifier_model, "auto")
    scalar_session = load_model_session(args.scalar_model, "auto")
    history = InferenceBurstHistory(
        size=args.history_size,
        reset_delta_c=args.history_reset_delta,
    )
    progress_callback = (
        (lambda message: print(f"[COMPARE] {message}", flush=True))
        if args.trace_stages
        else None
    )

    rows: list[ComparisonRow] = []
    raw_errors: list[float] = []
    calibrated_errors: list[float] = []
    reported_errors: list[float] = []
    classical_errors: list[float] = []

    for image_path, true_value in items:
        print(f"[COMPARE] Replaying {image_path.name}...", flush=True)
        classical_key = image_path.as_posix()
        classical_prediction = classical_by_path.get(classical_key)
        board_result = predict_board_pipeline_on_capture(
            image_path,
            obb_session=obb_session,
            rectifier_session=rectifier_session,
            scalar_session=scalar_session,
            history=history,
            progress=progress_callback,
        )

        raw_abs_err = abs(board_result.raw_prediction - true_value)
        calibrated_abs_err = abs(board_result.calibrated_prediction - true_value)
        reported_abs_err = abs(board_result.reported_prediction - true_value)
        raw_errors.append(raw_abs_err)
        calibrated_errors.append(calibrated_abs_err)
        reported_errors.append(reported_abs_err)

        classical_pred = None
        classical_abs_err = None
        classical_confidence = None
        if classical_prediction is not None:
            classical_pred = classical_prediction.predicted_value
            classical_abs_err = classical_prediction.abs_error
            classical_confidence = classical_prediction.confidence
            classical_errors.append(classical_abs_err)

        rows.append(
            ComparisonRow(
                image_path=classical_key,
                true_value=true_value,
                classical_pred=classical_pred,
                classical_abs_err=classical_abs_err,
                classical_confidence=classical_confidence,
                board_raw_pred=board_result.raw_prediction,
                board_raw_abs_err=raw_abs_err,
                board_calibrated_pred=board_result.calibrated_prediction,
                board_calibrated_abs_err=calibrated_abs_err,
                board_reported_pred=board_result.reported_prediction,
                board_reported_abs_err=reported_abs_err,
                board_selected_stage=board_result.selected_stage,
                board_selected_crop_x0=board_result.selected_crop_box_xyxy[0],
                board_selected_crop_y0=board_result.selected_crop_box_xyxy[1],
                board_selected_crop_x1=board_result.selected_crop_box_xyxy[2],
                board_selected_crop_y1=board_result.selected_crop_box_xyxy[3],
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "comparison.csv"
    json_path = args.output_dir / "summary.json"
    _write_csv(csv_path, rows)

    summary = {
        "manifest": str(args.manifest),
        "items": len(rows),
        "classical": _metrics(classical_errors),
        "board_raw": _metrics(raw_errors),
        "board_calibrated": _metrics(calibrated_errors),
        "board_reported": _metrics(reported_errors),
        "classical_geometry_mode": args.classical_geometry_mode,
        "classical_confidence_threshold": args.classical_confidence_threshold,
        "classical_center_radius_scale": args.classical_center_radius_scale,
        "history_size": args.history_size,
        "history_reset_delta": args.history_reset_delta,
        "obb_model": str(args.obb_model),
        "rectifier_model": str(args.rectifier_model),
        "scalar_model": str(args.scalar_model),
        "csv_path": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[COMPARE] Wrote {csv_path}", flush=True)
    print(f"[COMPARE] Wrote {json_path}", flush=True)
    print(
        "[COMPARE] "
        f"classical_mae={summary['classical']['mae']:.4f} "
        f"raw_mae={summary['board_raw']['mae']:.4f} "
        f"calibrated_mae={summary['board_calibrated']['mae']:.4f} "
        f"reported_mae={summary['board_reported']['mae']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
