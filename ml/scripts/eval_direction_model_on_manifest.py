"""Evaluate one direction gauge model on every row in a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

# Make the package importable when this script is run from the ml/ directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the manifest evaluation job."""
    parser = argparse.ArgumentParser(description="Evaluate a direction gauge model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--gauge-id",
        type=str,
        default="littlegood_home_temp_gauge_c",
        help="Gauge calibration identifier used to convert direction into value.",
    )
    return parser.parse_args()


def _load_model(model_path: Path) -> tf.keras.Model:
    """Load a saved Keras direction model."""
    print(f"[EVAL] Loading model from {model_path}...", flush=True)
    # The saved model includes a custom training loss, but evaluation only needs
    # the forward graph, so skip compile-time deserialization.
    model = tf.keras.models.load_model(model_path, compile=False)
    print("[EVAL] Model loaded.", flush=True)
    return model


def _load_image(path: Path, image_size: int = 224) -> tf.Tensor:
    """Load and normalize one RGB image for direction regression inference."""
    image: tf.Tensor = tf.keras.utils.load_img(path, target_size=(image_size, image_size))
    image_array: np.ndarray = tf.keras.utils.img_to_array(image).astype(np.float32)
    return tf.convert_to_tensor(image_array[None, ...] / 255.0, dtype=tf.float32)


def _needle_vector_to_value(
    unit_dx: float,
    unit_dy: float,
    *,
    min_angle_rad: float,
    sweep_rad: float,
    min_value: float,
    max_value: float,
) -> float:
    """Convert a unit direction vector to a calibrated gauge value."""
    raw_angle: float = math.atan2(unit_dy, unit_dx)
    shifted: float = (raw_angle - min_angle_rad) % (2.0 * math.pi)
    fraction: float = min(max(shifted / sweep_rad, 0.0), 1.0)
    return min_value + fraction * (max_value - min_value)


def main() -> None:
    """Run inference for every labeled image in the manifest and summarize errors."""
    args = _parse_args()
    model = _load_model(args.model)
    gauge_specs = load_gauge_specs()
    spec = gauge_specs[args.gauge_id]

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
            pred_vec: np.ndarray = np.asarray(model.predict(image, verbose=0)[0], dtype=np.float32)
            pred_value: float = _needle_vector_to_value(
                float(pred_vec[0]),
                float(pred_vec[1]),
                min_angle_rad=spec.min_angle_rad,
                sweep_rad=spec.sweep_rad,
                min_value=spec.min_value,
                max_value=spec.max_value,
            )
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
