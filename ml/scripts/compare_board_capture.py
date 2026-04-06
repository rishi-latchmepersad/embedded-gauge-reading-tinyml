"""Compare a raw board YUV422 capture against the firmware crop heuristic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

# Make the package importable when this script is run from the ml/ directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import (
    compare_board_capture,
    find_latest_board_capture,
)

DEFAULT_MODEL_PATH: Path = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "scalar_full_retrain_v9_int8"
    / "model_int8.tflite"
)


def _load_tflite_model(model_path: Path) -> tf.lite.Interpreter:
    """Load the quantized TFLite model used for the board-capture sanity check."""
    print(f"[BOARD] Loading model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    return interpreter


def _load_image(image_path: Path, image_size: int) -> np.ndarray:
    """Load the resized board crop as a normalized float32 batch."""
    image = tf.keras.utils.load_img(image_path, target_size=(image_size, image_size))
    image_array = tf.keras.utils.img_to_array(image).astype(np.float32)
    return np.expand_dims(image_array / 255.0, axis=0)


def _quantize_input(batch: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    """Quantize a float32 batch to the model's int8 input tensor."""
    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(batch / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(np.int8)


def _dequantize_output(output_tensor: np.ndarray, output_details: dict[str, Any]) -> float:
    """Convert the model's int8 output tensor back to a float gauge value."""
    scale = float(output_details["quantization"][0])
    zero_point = int(output_details["quantization"][1])
    return float(scale * (int(output_tensor) - zero_point))


def _predict_board_crop(model_path: Path, board_crop_path: Path, image_size: int) -> None:
    """Run the quantized model on the board-style crop and print the prediction."""
    interpreter = _load_tflite_model(model_path)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"[BOARD] Input details: {input_details}")
    print(f"[BOARD] Output details: {output_details}")

    batch = _load_image(board_crop_path, image_size)
    quantized_batch = _quantize_input(batch, input_details)
    interpreter.set_tensor(input_details["index"], quantized_batch)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details["index"])[0][0]
    prediction = _dequantize_output(raw_output, output_details)
    print(
        f"[BOARD] Model prediction on board crop: raw={int(raw_output)} pred={prediction:.6f}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the board capture comparison."""
    parser = argparse.ArgumentParser(
        description="Compare a raw board YUV422 capture with the firmware crop heuristic."
    )
    parser.add_argument(
        "--capture-path",
        type=Path,
        default=None,
        help="Path to a raw .yuv422 capture. Defaults to the newest file in captured_images.",
    )
    parser.add_argument(
        "--captured-dir",
        type=Path,
        default=REPO_ROOT / "captured_images",
        help="Directory that contains the board capture files.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the quantized TFLite model to run on the board crop.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "board_capture_compare",
        help="Directory where comparison artifacts should be written.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square output size for the resized crop.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Board capture width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Board capture height in pixels.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the board capture comparison and print a compact summary."""
    args = parse_args()
    capture_path = args.capture_path
    if capture_path is None:
        capture_path = find_latest_board_capture(args.captured_dir)

    capture_path = capture_path.resolve()
    print(f"[BOARD] Capture: {capture_path}")

    report = compare_board_capture(
        capture_path,
        args.output_dir,
        image_size=args.image_size,
        image_width=args.width,
        image_height=args.height,
    )

    board = report.board_crop_estimate
    box = board.crop_box
    print(
        "[BOARD] Crop: "
        f"x_min={box.x_min} y_min={box.y_min} w={box.width} h={box.height} "
        f"centroid=({box.centroid_x},{box.centroid_y}) "
        f"bright_count={box.bright_count}"
    )
    print(
        "[BOARD] Luma: "
        f"center={board.center_luma} "
        f"mean={board.mean_luma:.2f} "
        f"min={board.min_luma} "
        f"max={board.max_luma}"
    )
    print(f"[BOARD] Preview: {report.capture_preview_path}")
    print(f"[BOARD] Crop image: {report.capture_crop_path}")
    print(f"[BOARD] Comparison figure: {report.comparison_figure_path}")
    print(f"[BOARD] JSON report: {report.report_json_path}")

    if args.model.exists():
        _predict_board_crop(args.model, report.capture_crop_path, args.image_size)
    else:
        print(f"[BOARD] Model not found, skipping prediction: {args.model}")


if __name__ == "__main__":
    main()
