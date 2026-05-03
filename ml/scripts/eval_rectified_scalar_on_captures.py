"""Evaluate a rectifier + scalar-reader chain on raw board captures."""

from __future__ import annotations

import argparse
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
    compare_board_capture,
    load_yuv422_capture_as_rgb,
    resize_with_pad_rgb,
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for raw board-capture rectified evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a rectifier + scalar model chain on raw board captures."
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
        "--captured-dir",
        type=Path,
        default=REPO_ROOT / "data" / "captured" / "images",
        help="Directory containing raw .yuv422 board captures.",
    )
    parser.add_argument(
        "--capture-path",
        type=Path,
        action="append",
        default=[],
        help="Optional capture path to evaluate. Repeat to score specific files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Maximum number of newest captures to score when no explicit paths are given.",
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


def _quantize_input(batch: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    """Quantize a float32 batch to the scalar reader's input tensor dtype."""
    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(batch / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(np.int8)


def _dequantize_output(
    output_tensor: np.ndarray, output_details: dict[str, Any]
) -> float:
    """Convert the scalar reader output tensor back to a float prediction."""
    scale = float(output_details["quantization"][0])
    zero_point = int(output_details["quantization"][1])
    return float(scale * (int(output_tensor) - zero_point))


def _load_rectifier(model_path: Path) -> tf.keras.Model:
    """Load the rectifier Keras model with the MobileNetV2 custom objects."""
    print(f"[LIVE] Loading rectifier model from {model_path}...", flush=True)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
        },
        compile=False,
        safe_mode=False,
    )
    print(f"[LIVE] Rectifier loaded: {model.name}", flush=True)
    return model


def _load_scalar(model_path: Path) -> tf.lite.Interpreter:
    """Load the quantized scalar reader used after rectification."""
    print(f"[LIVE] Loading scalar reader from {model_path}...", flush=True)
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    return interpreter


def _predict_capture(
    rectifier: tf.keras.Model,
    scalar_interpreter: tf.lite.Interpreter,
    input_details: dict[str, Any],
    output_details: dict[str, Any],
    capture_path: Path,
    *,
    image_size: int,
    rectifier_crop_scale: float,
) -> dict[str, float]:
    """Run the rectifier + scalar chain on one raw board capture."""
    source_image = load_yuv422_capture_as_rgb(
        capture_path,
        image_width=image_size,
        image_height=image_size,
    )
    board_report = compare_board_capture(
        capture_path,
        capture_path.parent / "_live_rectified_probe",
        image_size=image_size,
        image_width=image_size,
        image_height=image_size,
    )
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
    rectifier_box = np.asarray(
        rectifier_pred["rectifier_box"]
        if isinstance(rectifier_pred, dict)
        else rectifier_pred
    ).reshape(-1)

    center_x = float(np.clip(rectifier_box[0], 0.0, 1.0))
    center_y = float(np.clip(rectifier_box[1], 0.0, 1.0))
    box_w = min(1.0, float(np.clip(rectifier_box[2], 0.05, 1.0)) * rectifier_crop_scale)
    box_h = min(1.0, float(np.clip(rectifier_box[3], 0.05, 1.0)) * rectifier_crop_scale)

    canvas_w = float(image_size)
    canvas_h = float(image_size)
    if (box_w < 0.25) or (box_h < 0.25) or (box_w > 0.95) or (box_h > 0.95):
        # Keep live replay aligned with the board fallback when the rectifier
        # predicts a box that is too tiny or too close to full-frame.
        x_min = 0.1027 * canvas_w
        y_min = 0.2573 * canvas_h
        x_max = 0.7987 * canvas_w
        y_max = 0.8071 * canvas_h
    else:
        x_min = max(0.0, (center_x - 0.5 * box_w) * canvas_w)
        y_min = max(0.0, (center_y - 0.5 * box_h) * canvas_h)
        x_max = min(canvas_w, (center_x + 0.5 * box_w) * canvas_w)
        y_max = min(canvas_h, (center_y + 0.5 * box_h) * canvas_h)
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
        "board_mean_luma": board_report.board_crop_estimate.mean_luma,
        "board_center_luma": float(board_report.board_crop_estimate.center_luma),
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
    """Run the rectifier + scalar chain on the newest raw board captures."""
    args = _parse_args()
    capture_paths = [path.resolve() for path in args.capture_path]
    if not capture_paths:
        candidates = sorted(
            args.captured_dir.glob("*.yuv422"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        capture_paths = [
            path.resolve() for path in candidates if path.stat().st_size > 0
        ][: args.limit]

    if not capture_paths:
        raise FileNotFoundError(f"No capture files found in {args.captured_dir}.")

    rectifier = _load_rectifier(args.rectifier_model)
    scalar_interpreter = _load_scalar(args.scalar_model)
    input_details = scalar_interpreter.get_input_details()[0]
    output_details = scalar_interpreter.get_output_details()[0]

    print(f"[LIVE] captures={len(capture_paths)}", flush=True)
    for capture_path in capture_paths:
        print(f"[LIVE] Predicting {capture_path.name}...", flush=True)
        pred = _predict_capture(
            rectifier,
            scalar_interpreter,
            input_details,
            output_details,
            capture_path,
            image_size=args.image_size,
            rectifier_crop_scale=args.rectifier_crop_scale,
        )
        print(
            f"[LIVE] {capture_path.name}: pred={pred['prediction']:.4f} "
            f"board_mean_luma={pred['board_mean_luma']:.2f} "
            f"board_center_luma={pred['board_center_luma']:.1f} "
            f"crop=({pred['_crop_x_min']:.1f},{pred['_crop_y_min']:.1f},"
            f"{pred['_crop_x_max']:.1f},{pred['_crop_y_max']:.1f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
