"""Verify the gauge_ellipse_v1 TFLite contract against its QAT weights."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_gauge_ellipse_v1 import _load_split, build_model


ALLOWED_OPS = {"CONV_2D", "MEAN", "FULLY_CONNECTED", "LOGISTIC", "RESHAPE"}


def _load_images(paths: np.ndarray) -> np.ndarray:
    """Decode validation PNGs with the same normalized input contract."""
    return np.stack(
        [tf.io.decode_png(tf.io.read_file(path), channels=1).numpy() for path in paths]
    ).astype(np.float32) / 255.0


def main() -> None:
    """Run parity, quantization-contract, and operator-set checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "gauge_ellipse_v1",
    )
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    weights_path = args.artifact_dir / "gauge_ellipse_v1_qat.weights.h5"
    legacy_model_path = args.artifact_dir / "gauge_ellipse_v1.keras"
    tflite_path = args.artifact_dir / "gauge_ellipse_v1_int8.tflite"
    paths, targets = _load_split(
        PROJECT_ROOT / "data" / "gauge_face_ellipse_v1_640_gray", "val"
    )
    paths, targets = paths[: args.samples], targets[: args.samples]
    images = _load_images(paths)

    # Rebuild the exact QAT graph because tfmot's generated wrapper is not
    # reliably deserializable through tf_keras 2.20's .keras loader.
    qat_model = tfmot.quantization.keras.quantize_model(build_model())
    if weights_path.is_file():
        weight_bytes = weights_path.read_bytes()
    else:
        with zipfile.ZipFile(legacy_model_path) as archive:
            weight_bytes = archive.read("model.weights.h5")
    # h5py-backed Keras loaders require a filesystem path, so use a scoped temp.
    with tempfile.TemporaryDirectory(prefix="gauge_ellipse_verify_") as temp_dir:
        weights_path = Path(temp_dir) / "model.weights.h5"
        weights_path.write_bytes(weight_bytes)
        qat_model.load_weights(weights_path)
    keras_prediction = np.asarray(qat_model.predict(images, verbose=0))

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    tflite_predictions = []
    for image in images:
        # why: the board receives signed int8 pixels, so quantize with the
        # generated affine contract instead of assuming a hard-coded zero point.
        quantized = np.clip(
            np.round(image / input_scale + input_zero), -128, 127
        ).astype(np.int8)[None]
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        tflite_predictions.append((raw - output_zero) * output_scale)
    tflite_prediction = np.concatenate(tflite_predictions, axis=0)

    ops = sorted(
        {
            detail["op_name"]
            for detail in interpreter._get_ops_details()
            if detail["op_name"] != "DELEGATE"
        }
    )
    report = {
        "samples": len(paths),
        "keras_vs_tflite_max_abs": float(
            np.max(np.abs(keras_prediction - tflite_prediction))
        ),
        "keras_vs_tflite_mean_abs": float(
            np.mean(np.abs(keras_prediction - tflite_prediction))
        ),
        "qat_mae_vs_label": float(np.mean(np.abs(keras_prediction - targets))),
        "tflite_mae_vs_label": float(np.mean(np.abs(tflite_prediction - targets))),
        "input": {
            "dtype": str(input_detail["dtype"]),
            "shape": input_detail["shape"].tolist(),
            "quantization": [float(input_scale), int(input_zero)],
        },
        "output": {
            "dtype": str(output_detail["dtype"]),
            "shape": output_detail["shape"].tolist(),
            "quantization": [float(output_scale), int(output_zero)],
        },
        "operators": ops,
        "operators_allowed": set(ops).issubset(ALLOWED_OPS),
    }
    output = args.artifact_dir / "verification.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
