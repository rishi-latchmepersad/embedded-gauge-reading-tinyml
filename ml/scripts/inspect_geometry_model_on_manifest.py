"""Inspect per-sample geometry predictions on a labeled manifest.

This diagnostic script loads a geometry-style Keras model, applies the same
board-style crop heuristic used by the deployment path, and prints the model's
gauge value, keypoint coordinates, and heatmap peak summary for each sample.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

# Make the package importable when this script is run from the `ml/` directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import (  # noqa: E402
    estimate_board_crop_from_rgb,
    load_rgb_image,
    resize_with_pad_rgb,
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
class GeometryRow:
    """One per-sample diagnostic row."""

    image_path: Path
    true_value: float
    prediction: float
    abs_error: float
    center_x: float | None
    center_y: float | None
    tip_x: float | None
    tip_y: float | None
    separation: float | None
    raw_angle_deg: float | None
    heatmap_summary: str
    crop_x_min: int
    crop_y_min: int
    crop_width: int
    crop_height: int


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inspection run."""
    parser = argparse.ArgumentParser(
        description="Inspect geometry-model predictions on a labeled manifest."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square model input size.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on the number of manifest rows to inspect.",
    )
    parser.add_argument(
        "--legacy-mobilenetv2-preprocess",
        action="store_true",
        help="Load older MobileNetV2 models that used a non-serializable preprocess Lambda.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    """Resolve a repo-relative path against the repository root."""
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_manifest(manifest_path: Path, *, max_samples: int) -> list[EvalItem]:
    """Load the labeled image/value pairs from a CSV manifest."""
    items: list[EvalItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        required_columns = {"image_path", "value"}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError("Manifest must include image_path and value columns.")

        for index, row in enumerate(reader):
            if max_samples > 0 and index >= max_samples:
                break
            items.append(
                EvalItem(
                    image_path=_resolve_path(Path(row["image_path"])),
                    value=float(row["value"]),
                )
            )
    return items


def _load_model(model_path: Path, *, legacy_preprocess: bool) -> tf.keras.Model:
    """Load a saved Keras model with the geometry custom layers available."""
    print(f"[INSPECT] Loading model from {model_path}...", flush=True)
    custom_objects: dict[str, Any] = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
        "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
        "GaugeValueFromKeypoints": GaugeValueFromKeypoints,
    }
    if legacy_preprocess:
        print("[INSPECT] Legacy MobileNetV2 preprocess support enabled.", flush=True)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )
    print(f"[INSPECT] Model loaded: {model.name}", flush=True)
    print(f"[INSPECT] Output names: {model.output_names}", flush=True)
    return model


def _predict_item(
    model: tf.keras.Model,
    image_path: Path,
    *,
    image_size: int,
) -> GeometryRow:
    """Run one board-style crop through the model and summarize its outputs."""
    source_image = load_rgb_image(image_path)
    board_estimate = estimate_board_crop_from_rgb(source_image)
    if board_estimate is None:
        raise ValueError(f"Board crop heuristic failed for {image_path}.")

    board_crop = resize_with_pad_rgb(
        source_image,
        (
            float(board_estimate.crop_box.x_min),
            float(board_estimate.crop_box.y_min),
            float(board_estimate.crop_box.x_max),
            float(board_estimate.crop_box.y_max),
        ),
        image_size=image_size,
    )
    batch = np.expand_dims(board_crop.astype(np.float32) / 255.0, axis=0)
    prediction = model.predict(batch, verbose=0)

    output_names = list(model.output_names)
    values: dict[str, np.ndarray] = {}
    if isinstance(prediction, dict):
        for name in output_names:
            if name in prediction:
                values[name] = np.asarray(prediction[name])
    elif isinstance(prediction, (list, tuple)):
        for name, array in zip(output_names, prediction, strict=True):
            values[name] = np.asarray(array)
    else:
        values[output_names[0]] = np.asarray(prediction)

    gauge_value = float(np.asarray(values.get("gauge_value", 0.0)).reshape(-1)[0])
    keypoint_coords = values.get("keypoint_coords")
    heatmaps = values.get("keypoint_heatmaps")

    center_x: float | None = None
    center_y: float | None = None
    tip_x: float | None = None
    tip_y: float | None = None
    separation: float | None = None
    raw_angle_deg: float | None = None
    if keypoint_coords is not None:
        coords = np.asarray(keypoint_coords, dtype=np.float32).reshape(-1, 2)
        if coords.shape[0] >= 2:
            center_x = float(coords[0, 0])
            center_y = float(coords[0, 1])
            tip_x = float(coords[1, 0])
            tip_y = float(coords[1, 1])
            dx = tip_x - center_x
            dy = tip_y - center_y
            separation = float(math.hypot(dx, dy))
            raw_angle_deg = float(math.degrees(math.atan2(dy, dx)))

    heatmap_summary = "n/a"
    if heatmaps is not None:
        heatmap_array = np.asarray(heatmaps, dtype=np.float32)
        if heatmap_array.ndim == 4:
            heatmap_array = heatmap_array[0]
        peaks: list[str] = []
        for channel in range(heatmap_array.shape[-1]):
            channel_map = heatmap_array[..., channel]
            peak_index = int(np.argmax(channel_map))
            peak_y, peak_x = np.unravel_index(peak_index, channel_map.shape)
            peak_value = float(channel_map[peak_y, peak_x])
            peaks.append(f"k{channel}:{peak_value:.3f}@({peak_x},{peak_y})")
        heatmap_summary = "; ".join(peaks)

    return GeometryRow(
        image_path=image_path,
        true_value=0.0,  # filled by caller
        prediction=gauge_value,
        abs_error=0.0,  # filled by caller
        center_x=center_x,
        center_y=center_y,
        tip_x=tip_x,
        tip_y=tip_y,
        separation=separation,
        raw_angle_deg=raw_angle_deg,
        heatmap_summary=heatmap_summary,
        crop_x_min=int(board_estimate.crop_box.x_min),
        crop_y_min=int(board_estimate.crop_box.y_min),
        crop_width=int(board_estimate.crop_box.width),
        crop_height=int(board_estimate.crop_box.height),
    )


def _format_row(row: GeometryRow) -> str:
    """Format a row for compact console inspection."""
    coords_text = "n/a"
    if row.center_x is not None and row.tip_x is not None:
        coords_text = (
            f"center=({row.center_x:.2f},{row.center_y:.2f}) "
            f"tip=({row.tip_x:.2f},{row.tip_y:.2f}) "
            f"sep={row.separation:.2f} angle={row.raw_angle_deg:.1f}deg"
        )
    return (
        f"{row.image_path.name}: true={row.true_value:.4f} pred={row.prediction:.4f} "
        f"abs_err={row.abs_error:.4f} {coords_text} heatmaps=[{row.heatmap_summary}] "
        f"crop=({row.crop_x_min},{row.crop_y_min},{row.crop_width},{row.crop_height})"
    )


def main() -> None:
    """Inspect geometry outputs for each sample in the manifest."""
    args = _parse_args()
    model = _load_model(args.model, legacy_preprocess=args.legacy_mobilenetv2_preprocess)
    items = _load_manifest(args.manifest, max_samples=args.max_samples)

    rows: list[GeometryRow] = []
    abs_errors: list[float] = []
    skipped = 0

    for item in items:
        print(f"[INSPECT] Predicting {item.image_path.name}...", flush=True)
        try:
            row = _predict_item(model, item.image_path, image_size=args.image_size)
        except ValueError:
            print(
                f"[INSPECT] Skipping {item.image_path.name}: board crop heuristic failed.",
                flush=True,
            )
            skipped += 1
            continue

        row = GeometryRow(
            image_path=row.image_path,
            true_value=item.value,
            prediction=row.prediction,
            abs_error=abs(row.prediction - item.value),
            center_x=row.center_x,
            center_y=row.center_y,
            tip_x=row.tip_x,
            tip_y=row.tip_y,
            separation=row.separation,
            raw_angle_deg=row.raw_angle_deg,
            heatmap_summary=row.heatmap_summary,
            crop_x_min=row.crop_x_min,
            crop_y_min=row.crop_y_min,
            crop_width=row.crop_width,
            crop_height=row.crop_height,
        )
        rows.append(row)
        abs_errors.append(row.abs_error)
        print(f"[INSPECT] { _format_row(row) }", flush=True)

    if abs_errors:
        print(f"[INSPECT] samples={len(abs_errors)} skipped={skipped}", flush=True)
        print(f"[INSPECT] mean_abs_err={float(np.mean(abs_errors)):.4f}", flush=True)
        print(f"[INSPECT] max_abs_err={float(np.max(abs_errors)):.4f}", flush=True)
        worst = sorted(rows, key=lambda row: row.abs_error, reverse=True)[:5]
        print("[INSPECT] Worst samples:", flush=True)
        for row in worst:
            print(f"[INSPECT]   {_format_row(row)}", flush=True)
    else:
        print(f"[INSPECT] No samples were scored; skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
