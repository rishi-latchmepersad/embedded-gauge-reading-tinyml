"""Evaluate the LittleGood held-out split and TFLite parity for retrained models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

ROOT = Path(__file__).resolve().parents[1]
ELLIPSE_ARTIFACT = ROOT / "artifacts" / "gauge_ellipse_littlegood_v5"
CENTER_ARTIFACT = ROOT / "artifacts" / "gauge_center_tip_littlegood_v4"
sys.path.insert(0, str(ROOT / "scripts"))
from train_gauge_center_tip_v1 import build_model as build_center_model  # noqa: E402
from train_gauge_ellipse_v1 import _load_split, build_model as build_ellipse_model  # noqa: E402


def _tflite_predict(path: Path, inputs: np.ndarray) -> np.ndarray:
    """Run a batch through an int8 TFLite interpreter using its affine scales."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    predictions: list[np.ndarray] = []
    for sample in inputs:
        # why: input/output scales are read from the flatbuffer, not assumed.
        quantized = np.clip(np.round(sample / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        predictions.append((raw - output_zero) * output_scale)
    return np.concatenate(predictions, axis=0)


def _ellipse_report() -> dict[str, object]:
    """Measure ellipse coordinates and Keras-vs-TFLite parity on LittleGood test."""
    data = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
    paths, targets = _load_split(data, "test")
    images = np.stack([tf.io.decode_png(tf.io.read_file(p), channels=1).numpy() for p in paths]).astype(np.float32) / 255.0
    qat = tfmot.quantization.keras.quantize_model(build_ellipse_model())
    qat.load_weights(ELLIPSE_ARTIFACT / "gauge_ellipse_v1_qat.weights.h5")
    keras_predictions = np.asarray(qat.predict(images, verbose=0))
    tflite_predictions = _tflite_predict(ELLIPSE_ARTIFACT / "gauge_ellipse_v1_int8.tflite", images)
    pixel_error = np.abs(keras_predictions[:, :4] - targets[:, :4]) * 640.0
    return {
        "samples": len(paths),
        "keras_mae_normalized": float(np.mean(np.abs(keras_predictions - targets))),
        "tflite_mae_normalized": float(np.mean(np.abs(tflite_predictions - targets))),
        "tflite_center_radius_mae_px": pixel_error.mean(axis=0).round(4).tolist(),
        "keras_vs_tflite_max_abs": float(np.max(np.abs(keras_predictions - tflite_predictions))),
        "keras_vs_tflite_mean_abs": float(np.mean(np.abs(keras_predictions - tflite_predictions))),
    }


def _center_tip_report() -> dict[str, object]:
    """Measure heatmap argmax errors and TFLite parity on LittleGood test."""
    data = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
    rows = json.loads((data / "metadata.json").read_text(encoding="utf-8"))["splits"]["test"]
    inputs = np.stack([np.load(data / row["heatmap"]).astype(np.float32) for row in rows])
    # Reuse the preparation function so input channel construction matches training.
    from train_gauge_center_tip_v1 import make_input  # noqa: PLC0415
    inputs = np.stack([make_input(data / row["image"], row) for row in rows])
    qat = tfmot.quantization.keras.quantize_model(build_center_model())
    qat.load_weights(CENTER_ARTIFACT / "gauge_center_tip_v1_qat.weights.h5")
    keras_predictions = np.asarray(qat.predict(inputs, verbose=0))
    tflite_predictions = _tflite_predict(CENTER_ARTIFACT / "gauge_center_tip_v1_int8.tflite", inputs)
    target_points = np.asarray([[row["center_xy_norm"], row["tip_xy_norm"]] for row in rows], dtype=np.float32) * 160.0
    def decode(predictions: np.ndarray) -> np.ndarray:
        """Decode each heatmap to one point in 160x160 crop coordinates."""
        points = []
        for sample in predictions:
            decoded = []
            for channel in range(2):
                y, x = np.unravel_index(np.argmax(sample[:, :, channel]), sample[:, :, channel].shape)
                decoded.append([(x + 0.5) * 2.0, (y + 0.5) * 2.0])
            points.append(decoded)
        return np.asarray(points, dtype=np.float32)
    keras_points = decode(keras_predictions)
    tflite_points = decode(tflite_predictions)
    keras_error = np.linalg.norm(keras_points - target_points, axis=2)
    tflite_error = np.linalg.norm(tflite_points - target_points, axis=2)
    return {
        "samples": len(rows),
        "keras_center_tip_mae_px": np.mean(np.abs(keras_points - target_points), axis=0).round(4).tolist(),
        "tflite_center_tip_mae_px": np.mean(np.abs(tflite_points - target_points), axis=0).round(4).tolist(),
        "tflite_center_tip_mean_euclidean_px": tflite_error.mean(axis=0).round(4).tolist(),
        "tflite_center_within_8px": float(np.mean( tflite_error[:, 0] <= 8.0)),
        "tflite_tip_within_8px": float(np.mean(tflite_error[:, 1] <= 8.0)),
        "keras_vs_tflite_max_abs": float(np.max(np.abs(keras_predictions - tflite_predictions))),
        "keras_vs_tflite_mean_abs": float(np.mean(np.abs(keras_predictions - tflite_predictions))),
    }


def main() -> None:
    """Write the held-out LittleGood evaluation report."""
    report = {"ellipse": _ellipse_report(), "center_tip": _center_tip_report()}
    output = ROOT / "artifacts" / "gauge_retrained_littlegood_v5_evaluation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
