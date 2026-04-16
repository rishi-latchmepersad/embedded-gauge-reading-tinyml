"""Evaluate a TFLite scalar model with a saved piecewise calibration fit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Final

import numpy as np
import tensorflow as tf

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the calibrated TFLite evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a TFLite scalar model with a saved piecewise calibration."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model_int8.tflite.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV manifest with image_path,value rows used for evaluation.",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        required=True,
        help="JSON file containing bias, weights, and knots for the piecewise fit.",
    )
    return parser.parse_args()


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest image path relative to the repo root when needed."""
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return REPO_ROOT / image_path


def _load_items(manifest_path: Path) -> tuple[list[Path], np.ndarray]:
    """Load image paths and labels from the CSV manifest."""
    image_paths: list[Path] = []
    values: list[float] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_paths.append(_resolve_image_path(row["image_path"]))
            values.append(float(row["value"]))
    return image_paths, np.asarray(values, dtype=np.float32)


def _predict_tflite(model_path: Path, image_paths: list[Path]) -> np.ndarray:
    """Run the scalar TFLite model on the supplied images."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_index = int(input_detail["index"])
    output_index = int(output_detail["index"])
    input_dtype = input_detail["dtype"]
    output_dtype = output_detail["dtype"]
    input_scale = float(input_detail["quantization"][0])
    input_zero_point = int(input_detail["quantization"][1])
    output_scale = float(output_detail["quantization"][0])
    output_zero_point = int(output_detail["quantization"][1])

    print(
        "[TFLITE] input_quant="
        f"({input_scale:.10f}, {input_zero_point}) "
        "output_quant="
        f"({output_scale:.10f}, {output_zero_point})",
        flush=True,
    )

    predictions: list[float] = []
    for image_path in image_paths:
        image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        image_array = tf.keras.utils.img_to_array(image).astype(np.float32) / 255.0
        if np.issubdtype(input_dtype, np.floating):
            interpreter.set_tensor(input_index, image_array[None, ...].astype(input_dtype))
        else:
            quantized = np.round(image_array / input_scale + input_zero_point).astype(input_dtype)
            interpreter.set_tensor(input_index, quantized[None, ...])
        interpreter.invoke()
        output_value = interpreter.get_tensor(output_index)[0][0]
        if np.issubdtype(output_dtype, np.floating):
            prediction = float(output_value)
        else:
            prediction = (float(output_value) - output_zero_point) * output_scale
        predictions.append(prediction)
        print(f"[TFLITE] {image_path.name}: pred={prediction:.4f}", flush=True)

    return np.asarray(predictions, dtype=np.float32)


def _apply_piecewise_calibration(
    predictions: np.ndarray,
    calibration_json: Path,
) -> tuple[np.ndarray, float]:
    """Apply the saved piecewise calibration to raw model predictions."""
    payload = json.loads(calibration_json.read_text(encoding="utf-8"))
    bias = float(payload["bias"])
    weights = np.asarray(payload["weights"], dtype=np.float32)
    knots = np.asarray(payload["knots"], dtype=np.float32)

    if weights.size != knots.size + 1:
        raise ValueError(
            "Calibration JSON is inconsistent: weights must be one longer than knots."
        )

    design = np.column_stack(
        [predictions] + [np.maximum(predictions - knot, 0.0) for knot in knots]
    )
    calibrated = design @ weights + bias
    return calibrated.astype(np.float32), bias


def main() -> None:
    """Evaluate raw and calibrated errors on a manifest."""
    args = _parse_args()
    image_paths, labels = _load_items(args.manifest)
    predictions = _predict_tflite(args.model, image_paths)
    calibrated_predictions, bias = _apply_piecewise_calibration(predictions, args.calibration_json)

    raw_mae = float(np.mean(np.abs(predictions - labels)))
    raw_max = float(np.max(np.abs(predictions - labels)))
    calibrated_mae = float(np.mean(np.abs(calibrated_predictions - labels)))
    calibrated_max = float(np.max(np.abs(calibrated_predictions - labels)))
    cases_over_5c = int(np.sum(np.abs(calibrated_predictions - labels) > 5.0))

    print(
        f"[CAL] raw_mae={raw_mae:.4f} raw_max={raw_max:.4f} "
        f"calibrated_mae={calibrated_mae:.4f} calibrated_max={calibrated_max:.4f} "
        f"cases_over_5c={cases_over_5c} bias={bias:.6f}",
        flush=True,
    )

    worst_indices = np.argsort(np.abs(calibrated_predictions - labels))[::-1][:10]
    for idx in worst_indices:
        print(
            "[CAL] worst: "
            f"{image_paths[idx].name} true={labels[idx]:.1f} "
            f"raw={predictions[idx]:.4f} cal={calibrated_predictions[idx]:.4f} "
            f"err={abs(calibrated_predictions[idx] - labels[idx]):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
