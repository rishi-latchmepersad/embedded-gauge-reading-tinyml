"""Evaluate a Keras gauge model on board-style cropped samples.

This command mirrors the board-style crop + resize path used for the current
scalar deployment, but it also reports interval coverage when the model exposes
lower/upper uncertainty bounds.
"""

from __future__ import annotations

import argparse
import csv
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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the board-style evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Keras gauge model on board-style cropped samples."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--legacy-mobilenetv2-preprocess",
        action="store_true",
        help="Load older MobileNetV2 models that used a non-serializable preprocess Lambda.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square model input size.",
    )
    return parser.parse_args()


def _load_model(model_path: Path, *, legacy_preprocess: bool) -> tf.keras.Model:
    """Load a saved Keras model with optional legacy MobileNetV2 support."""
    print(f"[BOARD-KERAS] Loading model from {model_path}...", flush=True)
    custom_objects: dict[str, Any] = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
        "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
        "GaugeValueFromKeypoints": GaugeValueFromKeypoints,
    }
    if legacy_preprocess:
        print("[BOARD-KERAS] Legacy MobileNetV2 preprocess support enabled.", flush=True)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )
    print(f"[BOARD-KERAS] Model loaded: {model.name}", flush=True)
    return model


def _load_manifest(manifest_path: Path) -> list[tuple[Path, float]]:
    """Load labeled image paths and values from the CSV manifest."""
    rows: list[tuple[Path, float]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        required = {"image_path", "value"}
        if not required.issubset(reader.fieldnames):
            raise ValueError("Manifest must include image_path and value columns.")
        for row in reader:
            raw_path = Path(row["image_path"])
            image_path = raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path)
            rows.append((image_path, float(row["value"])))
    return rows


def _predict_item(
    model: tf.keras.Model,
    image_path: Path,
    *,
    image_size: int,
) -> dict[str, float]:
    """Run one board-style crop through the model and normalize all outputs."""
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
    values: dict[str, float] = {}
    if isinstance(prediction, dict):
        for name in output_names:
            if name in prediction:
                values[name] = float(np.asarray(prediction[name]).reshape(-1)[0])
    elif isinstance(prediction, (list, tuple)):
        for name, array in zip(output_names, prediction, strict=True):
            values[name] = float(np.asarray(array).reshape(-1)[0])
    else:
        values[output_names[0]] = float(np.asarray(prediction).reshape(-1)[0])

    values["_crop_x_min"] = float(board_estimate.crop_box.x_min)
    values["_crop_y_min"] = float(board_estimate.crop_box.y_min)
    values["_crop_w"] = float(board_estimate.crop_box.width)
    values["_crop_h"] = float(board_estimate.crop_box.height)
    return values


def main() -> None:
    """Run the board-style preprocessing pipeline on each labeled sample."""
    args = _parse_args()
    model = _load_model(args.model, legacy_preprocess=args.legacy_mobilenetv2_preprocess)
    items = _load_manifest(args.manifest)

    abs_errors: list[float] = []
    interval_widths: list[float] = []
    covered_count = 0
    skipped = 0

    for image_path, true_value in items:
        print(f"[BOARD-KERAS] Predicting {image_path.name}...", flush=True)
        try:
            pred = _predict_item(model, image_path, image_size=args.image_size)
        except ValueError:
            print(
                f"[BOARD-KERAS] Skipping {image_path.name}: board crop heuristic failed.",
                flush=True,
            )
            skipped += 1
            continue

        prediction = pred.get("gauge_value")
        if prediction is None:
            prediction = pred.get(model.output_names[0], 0.0)
        abs_error = abs(prediction - true_value)
        abs_errors.append(abs_error)

        lower = pred.get("gauge_value_lower")
        upper = pred.get("gauge_value_upper")
        interval_text = ""
        if lower is not None and upper is not None:
            interval_width = upper - lower
            interval_widths.append(interval_width)
            covered = lower <= true_value <= upper
            covered_count += int(covered)
            interval_text = (
                f" lower={lower:.4f} upper={upper:.4f} width={interval_width:.4f} "
                f"covered={covered}"
            )

        print(
            f"[BOARD-KERAS] {image_path.name}: true={true_value:.4f} "
            f"pred={prediction:.4f} abs_err={abs_error:.4f} "
            f"crop=({int(pred['_crop_x_min'])},{int(pred['_crop_y_min'])},"
            f"{int(pred['_crop_w'])},{int(pred['_crop_h'])})"
            f"{interval_text}",
            flush=True,
        )

    if abs_errors:
        print(f"[BOARD-KERAS] samples={len(abs_errors)} skipped={skipped}", flush=True)
        print(f"[BOARD-KERAS] mean_abs_err={float(np.mean(abs_errors)):.4f}", flush=True)
        print(f"[BOARD-KERAS] max_abs_err={float(np.max(abs_errors)):.4f}", flush=True)
        if interval_widths:
            print(
                f"[BOARD-KERAS] interval_mean_width={float(np.mean(interval_widths)):.4f}",
                flush=True,
            )
            print(
                f"[BOARD-KERAS] interval_coverage={covered_count / len(interval_widths):.4f}",
                flush=True,
            )
    else:
        print(f"[BOARD-KERAS] No samples were scored; skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
