"""Evaluate a quantized scalar TFLite model on board-style cropped samples.

This command bridges the gap between the labeled-image offline metrics and the
live STM32 board path by:
- loading labeled RGB samples from a CSV manifest,
- estimating the same bright-region crop the firmware uses,
- resizing with pad to the model input size,
- quantizing the input tensor exactly like the board, and
- reporting raw prediction error against the labels.
"""

from __future__ import annotations

import argparse
import csv
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


@dataclass(frozen=True)
class EvalItem:
    """One labeled image and its target scalar value."""

    image_path: Path
    value: float


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the board-style evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a quantized TFLite model on board-style cropped samples."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the quantized TFLite model.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV file with image_path,value rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "board_style_eval",
        help="Directory where per-sample comparison artifacts should be written.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square model input size.",
    )
    return parser.parse_args()


def _resolve_image_path(raw_path: str) -> Path:
    """Resolve a manifest image path relative to the repo root when needed."""
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    return REPO_ROOT / image_path


def _load_manifest(manifest_path: Path) -> list[EvalItem]:
    """Load labeled image paths and values from the CSV manifest."""
    items: list[EvalItem] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            items.append(
                EvalItem(
                    image_path=_resolve_image_path(row["image_path"]),
                    value=float(row["value"]),
                )
            )
    return items


def _quantize_input(batch: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    """Quantize a float32 batch to the model's input tensor dtype."""
    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    quantized = np.round(batch / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(np.int8)


def _dequantize_output(output_tensor: np.ndarray, output_details: dict[str, Any]) -> float:
    """Convert the model output tensor back to a scalar float."""
    scale = float(output_details["quantization"][0])
    zero_point = int(output_details["quantization"][1])
    return float(scale * (int(output_tensor) - zero_point))


def _predict_board_style(
    model_path: Path,
    items: list[EvalItem],
    *,
    image_size: int,
    output_dir: Path,
) -> None:
    """Run the board-style preprocessing pipeline on each labeled sample."""
    print(f"[BOARD-EVAL] Loading model: {model_path}", flush=True)
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"[BOARD-EVAL] Input details: {input_details}", flush=True)
    print(f"[BOARD-EVAL] Output details: {output_details}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    abs_errors: list[float] = []
    skipped = 0

    for item in items:
        print(f"[BOARD-EVAL] Predicting {item.image_path.name}...", flush=True)
        source_image = load_rgb_image(item.image_path)
        board_estimate = estimate_board_crop_from_rgb(source_image)
        if board_estimate is None:
            print(
                f"[BOARD-EVAL] Skipping {item.image_path.name}: board crop heuristic failed.",
                flush=True,
            )
            skipped += 1
            continue

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
        quantized_batch = _quantize_input(batch, input_details)
        interpreter.set_tensor(int(input_details["index"]), quantized_batch)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(int(output_details["index"]))[0][0]
        prediction = _dequantize_output(raw_output, output_details)
        abs_error = abs(prediction - item.value)
        abs_errors.append(abs_error)
        print(
            f"[BOARD-EVAL] {item.image_path.name}: true={item.value:.4f} "
            f"pred={prediction:.4f} abs_err={abs_error:.4f} "
            f"crop=({board_estimate.crop_box.x_min},{board_estimate.crop_box.y_min},"
            f"{board_estimate.crop_box.width},{board_estimate.crop_box.height})",
            flush=True,
        )

    if abs_errors:
        print(f"[BOARD-EVAL] samples={len(abs_errors)} skipped={skipped}", flush=True)
        print(f"[BOARD-EVAL] mean_abs_err={float(np.mean(abs_errors)):.4f}", flush=True)
        print(f"[BOARD-EVAL] max_abs_err={float(np.max(abs_errors)):.4f}", flush=True)
    else:
        print(f"[BOARD-EVAL] No samples were scored; skipped={skipped}", flush=True)


def main() -> None:
    """Entry point for the board-style quantized evaluation."""
    args = _parse_args()
    items = _load_manifest(args.manifest)
    _predict_board_style(
        args.model,
        items,
        image_size=args.image_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
