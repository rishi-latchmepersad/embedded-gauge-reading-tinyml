"""Train a CVAT-augmented crop heatmap keypoint model with int8 QAT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from train_gauge_center_tip_cvat_crop_v1 import load_cvat
from train_gauge_center_tip_v1 import build_model, configure_gpu, export_int8


ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_cvat_heatmap_littlegood_v1"
HEATMAP = 80


def make_targets(points: np.ndarray) -> np.ndarray:
    """Rasterize center and tip Gaussians in the 80-pixel decoder grid."""
    yy, xx = np.mgrid[0:HEATMAP, 0:HEATMAP]
    targets = np.zeros((len(points), HEATMAP, HEATMAP, 2), np.float32)
    for index, pair in enumerate(points):
        for channel, point in enumerate(pair):
            px, py = point * HEATMAP - 0.5
            # why: clipping keeps the few tips outside the 1.35x crop trainable
            # without inventing extra source images or oversampling them.
            px, py = np.clip((px, py), 0.5, HEATMAP - 0.5)
            targets[index, ..., channel] = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 2.2**2))
    return targets


def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Emphasize sparse keypoint peaks, especially the tip channel."""
    weights = 1.0 + 28.0 * y_true * tf.constant([1.0, 3.0], y_true.dtype)[None, None, None, :]
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))


def decode(heatmaps: np.ndarray) -> np.ndarray:
    """Decode heatmaps with a local weighted centroid around each peak."""
    result = []
    for sample in heatmaps:
        row = []
        for channel in range(2):
            heatmap = sample[..., channel]
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y0, y1, x0, x1 = max(0, y - 6), min(HEATMAP, y + 7), max(0, x - 6), min(HEATMAP, x + 7)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            weights = np.maximum(heatmap[y0:y1, x0:x1] - 0.03, 0.0) ** 2
            total = weights.sum()
            row.append(np.asarray(((xx * weights).sum() / total + 0.5, (yy * weights).sum() / total + 0.5), np.float32) / HEATMAP if total else np.asarray((x + 0.5, y + 0.5), np.float32) / HEATMAP)
        result.append(row)
    return np.asarray(result, np.float32)


def predict(path: Path, samples: np.ndarray) -> np.ndarray:
    """Run the fully integer model and dequantize its heatmaps."""
    interpreter = tf.lite.Interpreter(model_path=str(path)); interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    predictions = []
    for sample in samples:
        scale, zero = inp["quantization"]
        interpreter.set_tensor(inp["index"], np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]); interpreter.invoke()
        raw = interpreter.get_tensor(out["index"]).astype(np.float32); scale, zero = out["quantization"]
        predictions.append((raw - zero) * scale)
    return np.concatenate(predictions)


def main() -> None:
    """Train with CVAT plus natural LittleGood frequency and score untouched test."""
    configure_gpu(); tf.keras.utils.set_random_seed(42); OUT.mkdir(parents=True, exist_ok=True)
    cvat_x, cvat_y = load_cvat(); cvat_y = cvat_y.reshape(-1, 2, 2); student = {split: np.load(STUDENT / f"{split}.npz") for split in ("train", "val", "test")}
    x_train = np.concatenate((cvat_x, student["train"]["inputs"])); y_train = make_targets(np.concatenate((cvat_y, student["train"]["points"])))
    x_val, y_val = student["val"]["inputs"], make_targets(student["val"]["points"])
    model = build_model(); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss); model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, epochs=12, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=loss); qat.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, epochs=4, verbose=2)
    path = OUT / "gauge_center_tip_cvat_heatmap_v1_int8.tflite"; export_int8(qat, x_train, path)
    prediction = decode(predict(path, student["test"]["inputs"])); truth = student["test"]["points"]; errors = np.linalg.norm((prediction - truth) * 160.0, axis=2)
    report = {"cvat_samples": len(cvat_x), "littlegood_test_samples": len(truth), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "bytes": path.stat().st_size}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
