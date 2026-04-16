"""Evaluate one scalar gauge model on every row in a CSV manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

# Make the package importable when this script is run from the ml/ directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.models import (
    GaugeValueFromKeypoints,
    SpatialSoftArgmax2D,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the manifest evaluation job."""
    parser = argparse.ArgumentParser(description="Evaluate a scalar gauge model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--legacy-mobilenetv2-preprocess",
        action="store_true",
        help="Load older MobileNetV2 models that used a non-serializable preprocess Lambda.",
    )
    return parser.parse_args()


def _load_model(model_path: Path, *, legacy_preprocess: bool) -> tf.keras.Model:
    """Load a saved Keras model with optional legacy MobileNetV2 support."""
    print(f"[EVAL] Loading model from {model_path}...", flush=True)
    custom_objects: dict[str, Any] = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
        "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
        "GaugeValueFromKeypoints": GaugeValueFromKeypoints,
    }
    if legacy_preprocess:
        print("[EVAL] Legacy MobileNetV2 preprocess support enabled.", flush=True)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )
    try:
        if len(model.outputs) > 1:
            model = tf.keras.Model(
                inputs=model.inputs,
                outputs=model.get_layer("gauge_value").output,
                name=f"{model.name}_scalar",
            )
            print("[EVAL] Using scalar gauge_value output for prediction.", flush=True)
    except ValueError:
        pass
    print("[EVAL] Model loaded.", flush=True)
    return model


def _load_image(path: Path, image_size: int = 224) -> tf.Tensor:
    """Load and normalize one RGB image for scalar regression inference."""
    image: tf.Tensor = tf.keras.utils.load_img(path, target_size=(image_size, image_size))
    image_array: np.ndarray = tf.keras.utils.img_to_array(image).astype(np.float32)
    return tf.convert_to_tensor(image_array[None, ...] / 255.0, dtype=tf.float32)


def main() -> None:
    """Run inference for every labeled image in the manifest and summarize errors."""
    args = _parse_args()
    model = _load_model(args.model, legacy_preprocess=args.legacy_mobilenetv2_preprocess)

    abs_errors: list[float] = []
    rows: list[tuple[str, float, float, float]] = []
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {args.manifest}")
        required_columns = {"image_path", "value"}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError("Manifest must include image_path and value columns.")

        for row in reader:
            raw_path = Path(row["image_path"])
            image_path = raw_path if raw_path.is_absolute() else (PROJECT_ROOT.parent / raw_path)
            value = float(row["value"])
            print(f"[EVAL] Predicting {image_path.name}...", flush=True)
            image: tf.Tensor = _load_image(image_path)
            pred_value: float = float(model.predict(image, verbose=0)[0][0])
            abs_error: float = abs(pred_value - value)
            abs_errors.append(abs_error)
            rows.append((image_path.as_posix(), value, pred_value, abs_error))
            print(
                f"{image_path.name}: true={value:.4f} pred={pred_value:.4f} "
                f"abs_err={abs_error:.4f}",
                flush=True,
            )

    if abs_errors:
        mean_abs_err: float = float(np.mean(abs_errors))
        max_abs_err: float = float(np.max(abs_errors))
        worst = max(rows, key=lambda item: item[3])
        cases_over_5c = sum(err > 5.0 for err in abs_errors)
        print(f"mean_abs_err={mean_abs_err:.4f}")
        print(f"max_abs_err={max_abs_err:.4f}")
        print(
            f"worst={worst[0]} true={worst[1]:.4f} pred={worst[2]:.4f} "
            f"abs_err={worst[3]:.4f}"
        )
        print(f"cases_over_5c={cases_over_5c}")


if __name__ == "__main__":
    main()
