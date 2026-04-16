"""Evaluate a keypoint-gated reader cascade on a labeled manifest.

This benchmark bridges the research idea and the board path:
- a localization model proposes keypoints and a confidence score,
- a scalar reader converts the selected crop into temperature, and
- a second pass is only attempted when the localization confidence is low.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import keras
import numpy as np

# Make the package importable when this script is run from the `ml/` directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import (  # noqa: E402
    estimate_board_crop_from_rgb,
    load_rgb_image,
)
from embedded_gauge_reading_tinyml.geometry_cascade import (  # noqa: E402
    CascadeResult,
    run_geometry_cascade,
)
from embedded_gauge_reading_tinyml.models import (  # noqa: E402
    GaugeValueFromKeypoints,
    SpatialSoftArgmax2D,
)


@dataclass(frozen=True)
class EvalItem:
    """One labeled image and its target scalar value."""

    image_path: Path
    value: float


@dataclass(frozen=True)
class EvalRow:
    """Per-sample results for the cascade benchmark."""

    image_path: Path
    true_value: float
    first_value: float
    second_value: float | None
    final_value: float
    first_abs_err: float
    final_abs_err: float
    first_confidence: float
    second_confidence: float | None
    used_second_pass: bool
    crop_x_min: int
    crop_y_min: int
    crop_width: int
    crop_height: int


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the cascade evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a keypoint-gated reader cascade on a manifest."
    )
    parser.add_argument(
        "--localizer-model",
        type=Path,
        required=True,
        help="Path to the Keras model that predicts keypoints and confidence.",
    )
    parser.add_argument(
        "--reader-model",
        type=Path,
        default=None,
        help=(
            "Optional scalar reader model. If omitted, the localizer model's "
            "gauge_value output is used for the final prediction."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV file with image_path,value rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "cascade_eval",
        help="Directory where the report CSV and summary JSON should be written.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square input size used by both cascade models.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
        help="Confidence threshold below which the cascade attempts a second crop.",
    )
    parser.add_argument(
        "--recrop-scale",
        type=float,
        default=0.75,
        help="Scale factor used when building the tighter second-pass crop.",
    )
    parser.add_argument(
        "--min-recrop-size",
        type=float,
        default=64.0,
        help="Minimum edge length for the second-pass crop.",
    )
    parser.add_argument(
        "--legacy-mobilenetv2-preprocess",
        action="store_true",
        help=(
            "Load older MobileNetV2 models that used a non-serializable preprocess "
            "Lambda."
        ),
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_manifest(manifest_path: Path) -> list[EvalItem]:
    """Load the image/value pairs from a CSV manifest."""
    items: list[EvalItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        required_columns = {"image_path", "value"}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError("Manifest must include image_path and value columns.")

        for row in reader:
            items.append(
                EvalItem(
                    image_path=_resolve_path(Path(row["image_path"])),
                    value=float(row["value"]),
                )
            )
    return items


def _load_model(model_path: Path, *, legacy_preprocess: bool) -> keras.Model:
    """Load a saved Keras model with the serializable geometry layers available."""
    print(f"[CASCADE] Loading model from {model_path}...", flush=True)
    custom_objects: dict[str, object] = {
        "preprocess_input": keras.applications.mobilenet_v2.preprocess_input,
        "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
        "GaugeValueFromKeypoints": GaugeValueFromKeypoints,
    }
    if legacy_preprocess:
        print("[CASCADE] Legacy MobileNetV2 preprocess support enabled.", flush=True)
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )
    print(f"[CASCADE] Model loaded: {model.name}", flush=True)
    return model


def _summarize_row(row: EvalRow) -> str:
    """Format one row for the per-sample console table."""
    second_value = "n/a" if row.second_value is None else f"{row.second_value:.4f}"
    second_confidence = (
        "n/a" if row.second_confidence is None else f"{row.second_confidence:.3f}"
    )
    return (
        f"{row.image_path.name}: true={row.true_value:.4f} "
        f"first={row.first_value:.4f} final={row.final_value:.4f} "
        f"abs_err={row.final_abs_err:.4f} first_conf={row.first_confidence:.3f} "
        f"second={second_value} second_conf={second_confidence} "
        f"used_second={row.used_second_pass}"
    )


def _run_cascade_on_manifest(
    *,
    localizer_model: keras.Model,
    reader_model: keras.Model,
    items: list[EvalItem],
    image_size: int,
    confidence_threshold: float,
    recrop_scale: float,
    min_recrop_size: float,
    output_dir: Path,
) -> list[EvalRow]:
    """Run the cascade over every manifest item and write a machine-readable log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[EvalRow] = []
    report_path = output_dir / "rows.csv"

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_path",
                "true_value",
                "first_value",
                "second_value",
                "final_value",
                "first_abs_err",
                "final_abs_err",
                "first_confidence",
                "second_confidence",
                "used_second_pass",
                "crop_x_min",
                "crop_y_min",
                "crop_width",
                "crop_height",
            ]
        )

        for item in items:
            print(f"[CASCADE] Predicting {item.image_path.name}...", flush=True)
            source_image = load_rgb_image(item.image_path)
            board_estimate = estimate_board_crop_from_rgb(source_image)
            if board_estimate is None:
                print(
                    f"[CASCADE] Skipping {item.image_path.name}: board crop heuristic failed.",
                    flush=True,
                )
                continue

            base_crop_box_xyxy = (
                float(board_estimate.crop_box.x_min),
                float(board_estimate.crop_box.y_min),
                float(board_estimate.crop_box.x_max),
                float(board_estimate.crop_box.y_max),
            )
            cascade: CascadeResult = run_geometry_cascade(
                model=localizer_model,
                reader_model=reader_model,
                source_image=source_image,
                base_crop_box_xyxy=base_crop_box_xyxy,
                image_height=source_image.shape[0],
                image_width=source_image.shape[1],
                input_size=image_size,
                confidence_threshold=confidence_threshold,
                recrop_scale=recrop_scale,
                min_recrop_size=min_recrop_size,
            )
            first_pass = cascade.first_pass
            second_pass = cascade.second_pass
            final_abs_err = abs(cascade.final_value - item.value)
            first_abs_err = abs(first_pass.value - item.value)
            row = EvalRow(
                image_path=item.image_path,
                true_value=item.value,
                first_value=first_pass.value,
                second_value=None if second_pass is None else second_pass.value,
                final_value=cascade.final_value,
                first_abs_err=first_abs_err,
                final_abs_err=final_abs_err,
                first_confidence=first_pass.confidence,
                second_confidence=None if second_pass is None else second_pass.confidence,
                used_second_pass=cascade.used_second_pass,
                crop_x_min=board_estimate.crop_box.x_min,
                crop_y_min=board_estimate.crop_box.y_min,
                crop_width=board_estimate.crop_box.width,
                crop_height=board_estimate.crop_box.height,
            )
            rows.append(row)
            writer.writerow(
                [
                    row.image_path.as_posix(),
                    row.true_value,
                    row.first_value,
                    row.second_value if row.second_value is not None else "",
                    row.final_value,
                    row.first_abs_err,
                    row.final_abs_err,
                    row.first_confidence,
                    row.second_confidence if row.second_confidence is not None else "",
                    int(row.used_second_pass),
                    row.crop_x_min,
                    row.crop_y_min,
                    row.crop_width,
                    row.crop_height,
                ]
            )
            print(f"[CASCADE] {_summarize_row(row)}", flush=True)

    return rows


def main() -> None:
    """Evaluate the cascade benchmark and print a summary table."""
    args = _parse_args()
    items = _load_manifest(args.manifest)
    localizer_model = _load_model(
        args.localizer_model,
        legacy_preprocess=args.legacy_mobilenetv2_preprocess,
    )
    reader_model = (
        localizer_model
        if args.reader_model is None
        else _load_model(
            args.reader_model,
            legacy_preprocess=args.legacy_mobilenetv2_preprocess,
        )
    )

    rows = _run_cascade_on_manifest(
        localizer_model=localizer_model,
        reader_model=reader_model,
        items=items,
        image_size=args.image_size,
        confidence_threshold=args.confidence_threshold,
        recrop_scale=args.recrop_scale,
        min_recrop_size=args.min_recrop_size,
        output_dir=args.output_dir,
    )

    if not rows:
        print("[CASCADE] No samples were scored.", flush=True)
        return

    final_abs_errors = np.asarray([row.final_abs_err for row in rows], dtype=np.float32)
    first_abs_errors = np.asarray([row.first_abs_err for row in rows], dtype=np.float32)
    second_pass_count = sum(row.used_second_pass for row in rows)
    worst = max(rows, key=lambda row: row.final_abs_err)
    summary = {
        "samples": len(rows),
        "skipped": len(items) - len(rows),
        "mean_first_pass_abs_err": float(np.mean(first_abs_errors)),
        "mean_final_abs_err": float(np.mean(final_abs_errors)),
        "max_final_abs_err": float(np.max(final_abs_errors)),
        "cases_over_5c": int(np.sum(final_abs_errors > 5.0)),
        "second_pass_count": second_pass_count,
        "worst_image": worst.image_path.as_posix(),
        "worst_true": worst.true_value,
        "worst_pred": worst.final_value,
        "worst_abs_err": worst.final_abs_err,
        "localizer_model": str(args.localizer_model),
        "reader_model": (
            str(args.reader_model) if args.reader_model is not None else str(args.localizer_model)
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[CASCADE] samples={summary['samples']} skipped={summary['skipped']}", flush=True)
    print(
        f"[CASCADE] mean_first_pass_abs_err={summary['mean_first_pass_abs_err']:.4f}",
        flush=True,
    )
    print(
        f"[CASCADE] mean_final_abs_err={summary['mean_final_abs_err']:.4f}",
        flush=True,
    )
    print(f"[CASCADE] max_final_abs_err={summary['max_final_abs_err']:.4f}", flush=True)
    print(f"[CASCADE] cases_over_5c={summary['cases_over_5c']}", flush=True)
    print(f"[CASCADE] second_pass_count={summary['second_pass_count']}", flush=True)
    print(
        f"[CASCADE] worst={summary['worst_image']} true={summary['worst_true']:.4f} "
        f"pred={summary['worst_pred']:.4f} abs_err={summary['worst_abs_err']:.4f}",
        flush=True,
    )
    print(f"[CASCADE] report_csv={args.output_dir / 'rows.csv'}", flush=True)
    print(f"[CASCADE] summary_json={summary_path}", flush=True)


if __name__ == "__main__":
    main()
