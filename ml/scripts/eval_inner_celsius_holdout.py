#!/usr/bin/env python3
"""Evaluate the inner-Celsius-only mask on the 3 hard-case board captures.

The three captures (07-03-51, 07-05-35, 07-07-22) were saved while the
real gauge read approximately 5 °C.  This script loads each raw YUV frame,
applies the training crop + resize (matching the board pipeline), optionally
applies the inner-Celsius mask, runs the TFLite model, and prints the
predicted temperature so we can compare against the expected ~5 °C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.firmware_preprocessing import (
    load_yuv422_capture_as_rgb,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.inner_celsius_mask import apply_inner_celsius_mask
from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)
from tensorflow import keras
from embedded_gauge_reading_tinyml.geometry_heatmap_tflite_utils import (
    load_tflite_model,
    run_tflite_model,
)
from embedded_gauge_reading_tinyml.heatmap_utils import normalized_point_from_heatmap

# Crop ratios matching the firmware / training pipeline.
CROP_X_MIN = 0.1027
CROP_Y_MIN = 0.2573
CROP_X_MAX = 0.7987
CROP_Y_MAX = 0.8071

CAPTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "captured_images"
DEFAULT_INT8_TFLITE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "st_ai_output"
    / "models"
    / "tip_focus_v4_112_int8"
    / "tip_focus_v4_112_int8.tflite"
)
DEFAULT_KERAS_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "ml"
    / "artifacts"
    / "training"
    / "geometry_heatmap_v4_112_inner_celsius_mask"
    / "model_v4_112.keras"
)

HOLDOUT_CAPTURES = [
    "capture_2026-05-27_07-03-51.yuv422",
    "capture_2026-05-27_07-05-35.yuv422",
    "capture_2026-05-27_07-07-22.yuv422",
]
EXPECTED_TEMPERATURE_C = 5.0


def _load_and_preprocess(path: Path, *, apply_mask: bool) -> np.ndarray:
    """Load a board YUV422 capture and produce a [1,224,224,3] float32 tensor."""
    rgb = load_yuv422_capture_as_rgb(
        path,
        image_width=224,
        image_height=224,
    ).astype(np.float32) / 255.0

    if apply_mask:
        rgb = apply_inner_celsius_mask(rgb)

    return rgb[np.newaxis, ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate inner-Celsius holdout captures."
    )
    parser.add_argument(
        "--int8-tflite-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--keras-model-path",
        type=Path,
        default=DEFAULT_KERAS_MODEL_PATH,
    )
    parser.add_argument(
        "--inner-celsius-mask",
        action="store_true",
        help="Apply the inner-Celsius-only mask before inference.",
    )
    args = parser.parse_args()

    print(f"Holdout evaluation -- expected temperature ~{EXPECTED_TEMPERATURE_C} °C")
    print(f"  Mask: {'ON' if args.inner_celsius_mask else 'OFF'}")

    if args.int8_tflite_path is not None:
        if not args.int8_tflite_path.exists():
            print(f"TFLite model not found: {args.int8_tflite_path}")
            sys.exit(1)

        print(f"  Model: {args.int8_tflite_path.name} (TFLite)")
        bundle = load_tflite_model(args.int8_tflite_path)

        def _infer_fn(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            outputs = run_tflite_model(bundle, t)
            chm = np.squeeze(outputs[0])
            thm = np.squeeze(outputs[1])
            return chm, thm
    else:
        if not args.keras_model_path.exists():
            print(f"Keras model not found: {args.keras_model_path}")
            sys.exit(1)

        print(f"  Model: {args.keras_model_path.name} (Keras)")
        model = keras.models.load_model(args.keras_model_path)

        def _infer_fn(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            outputs = model(t, training=False)
            if isinstance(outputs, dict):
                chm = np.asarray(outputs["center_heatmap"], dtype=np.float32)
                thm = np.asarray(outputs["tip_heatmap"], dtype=np.float32)
            else:
                chm = np.asarray(outputs[0], dtype=np.float32)
                thm = np.asarray(outputs[1], dtype=np.float32)
            chm = np.squeeze(chm)  # (1,112,112,1) -> (112,112)
            thm = np.squeeze(thm)
            return chm, thm

    print()

    for capture_name in HOLDOUT_CAPTURES:
        capture_path = CAPTURES_DIR / capture_name
        if not capture_path.exists():
            print(f"  {capture_name}: NOT FOUND")
            continue

        tensor = _load_and_preprocess(capture_path, apply_mask=args.inner_celsius_mask)

        center_hm, tip_hm = _infer_fn(tensor)

        cy, cx = normalized_point_from_heatmap(center_hm, method="softargmax")
        ty, tx = normalized_point_from_heatmap(tip_hm, method="softargmax")

        # Scale normalized [0,1] coords back to 224 px
        cx = cx * 224.0
        cy = cy * 224.0
        tx = tx * 224.0
        ty = ty * 224.0

        angle = angle_degrees_from_center_to_tip(cx, cy, tx, ty)
        temp = celsius_from_inner_dial_angle_degrees(angle)

        err = abs(temp - EXPECTED_TEMPERATURE_C)
        print(f"  {capture_name}")
        print(f"    angle={angle:.1f}°  temp={temp:.1f} °C  error={err:.1f} °C")


if __name__ == "__main__":
    main()
