#!/usr/bin/env python3
"""Export the trained centre-prediction model to int8 TFLite for the STM32N6."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import keras
from collections.abc import Iterable

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# Latest training run (best model by training MAE)
MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "center_model_20260531_193518"
    / "best_model.keras"
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "deployment" / "center_model_v2_int8"
METADATA_PATH = PROJECT_ROOT / "data" / "preprocessed_crops" / "metadata.json"


def _representative_dataset() -> Iterable[list[np.ndarray]]:
    """Yield 224×224 calibration images from the training set."""
    with open(METADATA_PATH) as f:
        entries = json.load(f)
    # Use train+val images for calibration (not test)
    calib = [e for e in entries if e["split"] in ("train", "val")]
    for e in calib[:100]:  # 100 images is more than enough for calibration
        path = PROJECT_ROOT / "data" / "preprocessed_crops" / e["image_path"]
        img = tf.io.read_file(str(path))
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img = tf.cast(img, tf.float32)
        yield [np.expand_dims(img.numpy(), axis=0)]


def main() -> None:
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # Load best model
    model = keras.models.load_model(str(MODEL_PATH))
    print(f"[EXPORT] Loaded model from {MODEL_PATH}", flush=True)

    # Check output layer
    if not model.get_layer("center_xy"):
        msg = "Model missing 'center_xy' output layer"
        raise ValueError(msg)
    print(f"[EXPORT] Model input: {model.input_shape}  output: {model.output_shape}", flush=True)

    # Convert to int8
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = OUTPUT_DIR / "model_int8.tflite"
    with open(str(tflite_path), "wb") as f:
        f.write(tflite_model)
    print(f"[EXPORT] Saved int8 TFLite model: {tflite_path} ({len(tflite_model)} bytes)", flush=True)

    # Inspect quantization parameters
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    metadata = {
        "source_model_path": str(MODEL_PATH),
        "tflite_path": str(tflite_path.relative_to(PROJECT_ROOT)),
        "input_shape": [int(v) for v in in_det["shape"]],
        "output_shape": [int(v) for v in out_det["shape"]],
        "input_scale": float(in_det["quantization"][0]),
        "input_zero_point": int(in_det["quantization"][1]),
        "output_scale": float(out_det["quantization"][0]),
        "output_zero_point": int(out_det["quantization"][1]),
        "representative_examples": 100,
        "board_input_size": {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
        "deployment_kind": "center_detector",
        "alpha": 0.5,
        "head_units": 128,
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[EXPORT] Saved metadata: {meta_path}", flush=True)

    # Done — firmware integration (Cube.AI import, C codegen) is a manual
    # STM32CubeIDE step.  The .tflite is ready at the path below.
    print(f"[EXPORT] TFLite model ready at {tflite_path}", flush=True)
    print("[EXPORT] Import into STM32Cube.AI to generate C headers + NPU weight buffer.", flush=True)


if __name__ == "__main__":
    main()
