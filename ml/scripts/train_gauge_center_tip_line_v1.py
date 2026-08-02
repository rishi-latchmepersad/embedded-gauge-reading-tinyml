"""Train an int8 center-plus-needle-line heatmap model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from train_gauge_center_tip_fullframe_v1 import decode
from train_gauge_center_tip_v1 import build_model, configure_gpu, export_int8, load_arrays


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_line_littlegood_v2"
SIZE = 80


def line_targets(points: np.ndarray) -> np.ndarray:
    """Create a center Gaussian and a thick line heatmap in crop coordinates."""
    result = np.zeros((len(points), SIZE, SIZE, 2), dtype=np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for index, pair in enumerate(points):
        center, tip = pair
        center_px, tip_px = center * SIZE, tip * SIZE
        result[index, ..., 0] = np.exp(-((xx - center_px[0]) ** 2 + (yy - center_px[1]) ** 2) / (2.0 * 2.0**2))
        vector = tip_px - center_px
        length = np.linalg.norm(vector) + 1e-6
        unit = vector / length
        projection = (xx - center_px[0]) * unit[0] + (yy - center_px[1]) * unit[1]
        distance = np.abs((xx - center_px[0]) * unit[1] - (yy - center_px[1]) * unit[0])
        inside = (projection >= 0.0) & (projection <= length)
        result[index, ..., 1] = np.where(inside, np.exp(-(distance**2) / (2.0 * 2.2**2)), 0.0)
    return result


def line_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Weight positive line and center pixels to avoid an all-zero solution."""
    channel_weight = tf.constant([2.0, 1.0], dtype=y_true.dtype)[None, None, None, :]
    weights = 1.0 + 32.0 * y_true * channel_weight
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))


def decode_line(heatmaps: np.ndarray) -> np.ndarray:
    """Decode center, line direction, and line extent into center/tip points."""
    center = decode(heatmaps[..., :1].repeat(2, axis=-1))[:, 0]
    points = []
    for sample, origin in zip(heatmaps[..., 1], center):
        y, x = np.mgrid[0:SIZE, 0:SIZE]
        weights = np.maximum(sample - 0.12, 0.0) ** 2
        direction_pixels = np.stack((x - origin[0] * SIZE, y - origin[1] * SIZE), axis=-1)
        direction = (direction_pixels * weights[..., None]).sum(axis=(0, 1))
        direction /= np.linalg.norm(direction) + 1e-6
        projection = direction_pixels[..., 0] * direction[0] + direction_pixels[..., 1] * direction[1]
        extent = float(np.percentile(projection[weights > 0.12], 90)) if np.any(weights > 0.12) else 20.0
        points.append(np.stack((origin, origin + direction * max(extent, 12.0) / SIZE)))
    return np.asarray(points, dtype=np.float32)


def main() -> None:
    """Train, export, and score the line-supervised model on LittleGood test."""
    configure_gpu(); tf.keras.utils.set_random_seed(42); OUT.mkdir(parents=True, exist_ok=True)
    student = {split: np.load(STUDENT / f"{split}.npz") for split in ("train", "val", "test")}
    # why: the prepared student tensors and local point labels share the exact
    # predicted-ellipse crop transform; stale generic heatmaps do not.
    x_train = student["train"]["inputs"]
    y_train = line_targets(student["train"]["points"])
    x_val = student["val"]["inputs"]
    y_val = line_targets(student["val"]["points"])
    model = build_model(); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=line_loss)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, epochs=12, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=line_loss)
    qat.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, epochs=4, verbose=2)
    path = OUT / "gauge_center_tip_line_v1_int8.tflite"; export_int8(qat, x_train, path)
    interpreter = tf.lite.Interpreter(model_path=str(path)); interpreter.allocate_tensors(); inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    predictions = []
    for sample in student["test"]["inputs"]:
        scale, zero = inp["quantization"]; interpreter.set_tensor(inp["index"], np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]); interpreter.invoke(); raw = interpreter.get_tensor(out["index"]).astype(np.float32); scale, zero = out["quantization"]; predictions.append((raw - zero) * scale)
    result = decode_line(np.concatenate(predictions)); errors = np.linalg.norm((result - student["test"]["points"]) * 160.0, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "bytes": path.stat().st_size}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
