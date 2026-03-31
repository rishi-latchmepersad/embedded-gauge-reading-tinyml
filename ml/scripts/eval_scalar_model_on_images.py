"""Evaluate a scalar gauge regressor on a small set of labeled images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EvalItem:
    """One image and its scalar gauge value label."""

    image_path: Path
    value: float


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the small evaluation job."""
    parser = argparse.ArgumentParser(description="Evaluate a scalar gauge model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--legacy-mobilenetv2-preprocess",
        action="store_true",
        help="Load older MobileNetV2 models that used a non-serializable preprocess Lambda.",
    )
    parser.add_argument(
        "--image",
        action="append",
        nargs=2,
        metavar=("PATH", "VALUE"),
        required=True,
        help="Image path and scalar label. Repeat for each evaluation image.",
    )
    return parser.parse_args()


def _load_model(model_path: Path, *, legacy_preprocess: bool) -> tf.keras.Model:
    """Load a saved Keras model, optionally providing the legacy preprocess symbol."""
    print(f"[EVAL] Loading model from {model_path}...", flush=True)
    if legacy_preprocess:
        custom_objects: dict[str, Any] = {
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        model = tf.keras.models.load_model(model_path)
    print("[EVAL] Model loaded.", flush=True)
    return model


def _load_image(path: Path, image_size: int = 224) -> tf.Tensor:
    """Load and normalize one RGB image for scalar regression inference."""
    image: tf.Tensor = tf.keras.utils.load_img(path, target_size=(image_size, image_size))
    image_array: np.ndarray = tf.keras.utils.img_to_array(image).astype(np.float32)
    return tf.convert_to_tensor(image_array[None, ...] / 255.0, dtype=tf.float32)


def main() -> None:
    """Run inference for each labeled image and print prediction errors."""
    args = _parse_args()
    model = _load_model(args.model, legacy_preprocess=args.legacy_mobilenetv2_preprocess)

    items: list[EvalItem] = [
        EvalItem(image_path=Path(path), value=float(value)) for path, value in args.image
    ]

    abs_errors: list[float] = []
    for item in items:
        print(f"[EVAL] Predicting {item.image_path.name}...", flush=True)
        image: tf.Tensor = _load_image(item.image_path)
        pred_value: float = float(model.predict(image, verbose=0)[0][0])
        abs_error: float = abs(pred_value - item.value)
        abs_errors.append(abs_error)
        print(
            f"{item.image_path.name}: true={item.value:.4f} pred={pred_value:.4f} "
            f"abs_err={abs_error:.4f}"
        )

    if abs_errors:
        print(f"mean_abs_err={float(np.mean(abs_errors)):.4f}")


if __name__ == "__main__":
    main()
