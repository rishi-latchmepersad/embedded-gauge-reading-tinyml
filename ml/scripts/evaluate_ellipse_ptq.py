"""Compare a full-integer PTQ export with the current ellipse QAT export.

This diagnostic keeps the trained FP32 weights fixed.  It answers whether the
large deployment error comes from the regression architecture or from the
TensorFlow Model Optimization fake-quant wrapper used during QAT.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "gauge_ellipse_littlegood_v3"
DATA = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"


def _load_samples() -> tuple[np.ndarray, np.ndarray]:
    """Load the held-out LittleGood grayscale images and ellipse targets."""
    from train_gauge_ellipse_v1 import _load_split

    paths, targets = _load_split(DATA, "test")
    images = np.stack(
        [tf.io.decode_png(tf.io.read_file(path), channels=1).numpy() for path in paths]
    ).astype(np.float32) / 255.0
    return images, targets


def _representative(images: np.ndarray):
    """Yield representative images in the model's float input contract."""
    for image in images[: min(256, len(images))]:
        yield [image[None]]


def _predict(path: Path, images: np.ndarray) -> np.ndarray:
    """Run an exported integer model and dequantize its output vector."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    predictions: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(
            np.round(image / input_scale + input_zero), -128, 127
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        predictions.append((raw - output_zero) * output_scale)
    return np.concatenate(predictions, axis=0)


def main() -> None:
    """Export FP32 weights with PTQ, evaluate, and write a comparison report."""
    images, targets = _load_samples()
    model = keras.models.load_model(ARTIFACTS / "gauge_ellipse_v1_fp32.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative(images)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path = ARTIFACTS / "gauge_ellipse_v1_ptq_int8.tflite"
    path.write_bytes(converter.convert())
    prediction = _predict(path, images)
    report = {
        "bytes": path.stat().st_size,
        "ptq_mae_normalized": float(np.mean(np.abs(prediction - targets))),
        "ptq_center_radius_mae_px": (
            np.mean(np.abs(prediction[:, :4] - targets[:, :4]), axis=0) * 640.0
        ).round(4).tolist(),
    }
    (ARTIFACTS / "ptq_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
