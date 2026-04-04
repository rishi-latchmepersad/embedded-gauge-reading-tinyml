"""Evaluate a scalar gauge TFLite model on a labeled CSV manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvalItem:
    """One labeled evaluation image."""

    image_path: Path
    value: float


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation job."""
    parser = argparse.ArgumentParser(
        description="Evaluate a scalar gauge TFLite model on a CSV manifest."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the TFLite model to evaluate.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV file with image_path,value rows.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size used by the model input.",
    )
    return parser.parse_args()


def _load_manifest(manifest_path: Path) -> list[EvalItem]:
    """Load labeled image paths from a CSV manifest."""
    items: list[EvalItem] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_image_path = Path(row["image_path"])
            image_path = raw_image_path if raw_image_path.is_absolute() else (PROJECT_ROOT / raw_image_path)
            items.append(
                EvalItem(
                    image_path=image_path.resolve(),
                    value=float(row["value"]),
                )
            )
    return items


def _load_image(image_path: Path, image_size: int) -> np.ndarray:
    """Load one image and normalize it to the model's expected float range."""
    image = tf.keras.utils.load_img(image_path, target_size=(image_size, image_size))
    image_array = tf.keras.utils.img_to_array(image).astype(np.float32)
    return np.expand_dims(image_array / 255.0, axis=0)


def _quantize_input(input_tensor: np.ndarray, input_details: dict[str, object]) -> np.ndarray:
    """Quantize a float32 input tensor to the model's requested dtype."""
    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(input_tensor / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(np.int8)


def _dequantize_output(output_tensor: np.ndarray, output_details: dict[str, object]) -> float:
    """Convert a quantized model output back to float."""
    scale = float(output_details["quantization"][0])
    zero_point = int(output_details["quantization"][1])
    return float(scale * (int(output_tensor) - zero_point))


def _evaluate_model(model_path: Path, items: Iterable[EvalItem], image_size: int) -> None:
    """Run the TFLite model on each labeled image and print the errors."""
    print(f"[EVAL] Loading TFLite model from {model_path}.", flush=True)
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[EVAL] Input details: {input_details}", flush=True)
    print(f"[EVAL] Output details: {output_details}", flush=True)

    abs_errors: list[float] = []
    for item in items:
        print(f"[EVAL] Predicting {item.image_path.name}...", flush=True)
        input_tensor = _load_image(item.image_path, image_size)
        quantized_input = _quantize_input(input_tensor, input_details[0])
        interpreter.set_tensor(input_details[0]["index"], quantized_input)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]["index"])[0][0]
        prediction = _dequantize_output(raw_output, output_details[0])
        abs_error = abs(prediction - item.value)
        abs_errors.append(abs_error)
        print(
            f"{item.image_path.name}: true={item.value:.4f} pred={prediction:.4f} "
            f"abs_err={abs_error:.4f}",
            flush=True,
        )

    if abs_errors:
        print(f"mean_abs_err={float(np.mean(abs_errors)):.4f}", flush=True)
        print(f"max_abs_err={float(np.max(abs_errors)):.4f}", flush=True)


def main() -> None:
    """Entry point for the command-line tool."""
    args = _parse_args()
    items = _load_manifest(args.manifest)
    _evaluate_model(args.model, items, args.image_size)


if __name__ == "__main__":
    main()
