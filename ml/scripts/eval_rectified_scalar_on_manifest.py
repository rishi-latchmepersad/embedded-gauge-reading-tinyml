"""Evaluate a rectifier model + scalar reader chain on a labeled manifest."""

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
    load_rgb_image,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.models import (  # noqa: E402
    build_mobilenetv2_rectifier_model,
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for rectified scalar evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a rectifier + scalar model chain on a labeled manifest."
    )
    parser.add_argument(
        "--rectifier-model",
        type=Path,
        required=True,
        help="Path to the saved rectifier Keras model.",
    )
    parser.add_argument(
        "--scalar-model",
        type=Path,
        required=True,
        help="Path to the quantized scalar TFLite reader.",
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
        help="Square input size for both the rectifier and scalar reader.",
    )
    parser.add_argument(
        "--rectifier-crop-scale",
        type=float,
        default=1.25,
        help="Scale factor applied to the rectifier-predicted crop before scalar inference.",
    )
    return parser.parse_args()


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest image path relative to the repo root when needed."""
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return REPO_ROOT / image_path


def _load_manifest(manifest_path: Path) -> list[tuple[Path, float]]:
    """Load labeled image paths and scalar values from the CSV manifest."""
    items: list[tuple[Path, float]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            items.append(
                (_resolve_image_path(row["image_path"]), float(row["value"]))
            )
    return items


def _quantize_input(batch: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    """Quantize a float32 batch to the scalar reader's input tensor dtype."""
    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(batch / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(np.int8)


def _dequantize_output(output_tensor: np.ndarray, output_details: dict[str, Any]) -> float:
    """Convert the scalar reader output tensor back to a float prediction."""
    scale = float(output_details["quantization"][0])
    zero_point = int(output_details["quantization"][1])
    return float(scale * (int(output_tensor) - zero_point))


def _load_rectifier(model_path: Path) -> tf.keras.Model:
    """Load the rectifier Keras model with the MobileNetV2 custom objects."""
    print(f"[RECT-EVAL] Loading rectifier model from {model_path}...", flush=True)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
        },
        compile=False,
        safe_mode=False,
    )
    print(f"[RECT-EVAL] Rectifier loaded: {model.name}", flush=True)
    return model


def _predict_rectified_scalar(
    rectifier: tf.keras.Model,
    scalar_interpreter: tf.lite.Interpreter,
    input_details: dict[str, Any],
    output_details: dict[str, Any],
    image_path: Path,
    *,
    image_size: int,
    rectifier_crop_scale: float,
) -> dict[str, float]:
    """Predict a crop with the rectifier, then read that crop with the scalar model."""
    source_image = load_rgb_image(image_path)
    full_frame = resize_with_pad_rgb(
        source_image,
        (
            0.0,
            0.0,
            float(source_image.shape[1]),
            float(source_image.shape[0]),
        ),
        image_size=image_size,
    )
    rectifier_batch = np.expand_dims(full_frame.astype(np.float32) / 255.0, axis=0)
    rectifier_pred = rectifier.predict(rectifier_batch, verbose=0)
    if isinstance(rectifier_pred, dict):
        rectifier_box = np.asarray(rectifier_pred["rectifier_box"]).reshape(-1)
    else:
        rectifier_box = np.asarray(rectifier_pred).reshape(-1)

    center_x = float(np.clip(rectifier_box[0], 0.0, 1.0))
    center_y = float(np.clip(rectifier_box[1], 0.0, 1.0))
    box_w = float(np.clip(rectifier_box[2], 0.05, 1.0))
    box_h = float(np.clip(rectifier_box[3], 0.05, 1.0))

    canvas_w = float(image_size)
    canvas_h = float(image_size)
    scaled_box_w = min(1.0, box_w * rectifier_crop_scale)
    scaled_box_h = min(1.0, box_h * rectifier_crop_scale)
    x_min = max(0.0, (center_x - 0.5 * scaled_box_w) * canvas_w)
    y_min = max(0.0, (center_y - 0.5 * scaled_box_h) * canvas_h)
    x_max = min(canvas_w, (center_x + 0.5 * scaled_box_w) * canvas_w)
    y_max = min(canvas_h, (center_y + 0.5 * scaled_box_h) * canvas_h)

    if x_max <= x_min + 1.0:
        x_max = min(canvas_w, x_min + 1.0)
    if y_max <= y_min + 1.0:
        y_max = min(canvas_h, y_min + 1.0)

    crop = resize_with_pad_rgb(
        full_frame,
        (x_min, y_min, x_max, y_max),
        image_size=image_size,
    )
    batch = np.expand_dims(crop.astype(np.float32) / 255.0, axis=0)
    quantized_batch = _quantize_input(batch, input_details)
    scalar_interpreter.set_tensor(int(input_details["index"]), quantized_batch)
    scalar_interpreter.invoke()
    raw_output = scalar_interpreter.get_tensor(int(output_details["index"]))[0][0]
    prediction = _dequantize_output(raw_output, output_details)

    return {
        "prediction": prediction,
        "_rect_center_x": center_x,
        "_rect_center_y": center_y,
        "_rect_box_w": box_w,
        "_rect_box_h": box_h,
        "_crop_x_min": x_min,
        "_crop_y_min": y_min,
        "_crop_x_max": x_max,
        "_crop_y_max": y_max,
    }


def main() -> None:
    """Run the rectifier + scalar chain on every labeled sample."""
    args = _parse_args()
    rectifier = _load_rectifier(args.rectifier_model)
    items = _load_manifest(args.manifest)

    print(f"[RECT-EVAL] Loading scalar reader from {args.scalar_model}...", flush=True)
    scalar_interpreter = tf.lite.Interpreter(model_path=str(args.scalar_model), num_threads=1)
    scalar_interpreter.allocate_tensors()
    input_details = scalar_interpreter.get_input_details()[0]
    output_details = scalar_interpreter.get_output_details()[0]
    print(f"[RECT-EVAL] Scalar input details: {input_details}", flush=True)
    print(f"[RECT-EVAL] Scalar output details: {output_details}", flush=True)

    abs_errors: list[float] = []
    for image_path, true_value in items:
        print(f"[RECT-EVAL] Predicting {image_path.name}...", flush=True)
        pred = _predict_rectified_scalar(
            rectifier,
            scalar_interpreter,
            input_details,
            output_details,
            image_path,
            image_size=args.image_size,
            rectifier_crop_scale=args.rectifier_crop_scale,
        )
        abs_error = abs(pred["prediction"] - true_value)
        abs_errors.append(abs_error)
        print(
            f"[RECT-EVAL] {image_path.name}: true={true_value:.4f} "
            f"pred={pred['prediction']:.4f} abs_err={abs_error:.4f} "
            f"box=({pred['_crop_x_min']:.1f},{pred['_crop_y_min']:.1f},"
            f"{pred['_crop_x_max']:.1f},{pred['_crop_y_max']:.1f})",
            flush=True,
        )

    if abs_errors:
        print(f"[RECT-EVAL] samples={len(abs_errors)}", flush=True)
        print(f"[RECT-EVAL] mean_abs_err={float(np.mean(abs_errors)):.4f}", flush=True)
        print(f"[RECT-EVAL] max_abs_err={float(np.max(abs_errors)):.4f}", flush=True)


if __name__ == "__main__":
    main()
