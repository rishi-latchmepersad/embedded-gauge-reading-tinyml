"""Train a rotation-augmented canonical-direction keypoint model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from train_gauge_center_tip_fullframe_v1 import decode, tip_weighted_loss
from train_gauge_center_tip_v1 import build_model, configure_gpu, export_int8, load_arrays


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_direction_aug_littlegood_v1"
SIZE = 80
RADIUS = 0.27


def targets(points: np.ndarray) -> np.ndarray:
    """Create center and fixed-radius direction heatmaps."""
    result = np.zeros((len(points), SIZE, SIZE, 2), np.float32); yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for index, pair in enumerate(points):
        center, tip = pair; direction = tip - center; direction /= np.linalg.norm(direction) + 1e-6
        for channel, point in enumerate((center, center + direction * RADIUS)):
            px, py = point * SIZE - 0.5; result[index, ..., channel] = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 2.2**2))
    return result


def rotate_dataset(inputs: np.ndarray, heatmaps: np.ndarray, training: bool) -> tf.data.Dataset:
    """Rotate each source crop and both heatmap channels consistently."""
    ds = tf.data.Dataset.from_tensor_slices((inputs, heatmaps))
    if training: ds = ds.shuffle(len(inputs), seed=42, reshuffle_each_iteration=True)
    def augment(image: tf.Tensor, heatmap: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply a quarter-turn augmentation without duplicating files."""
        k = tf.random.uniform((), 0, 4, dtype=tf.int32, seed=42)
        return tf.image.rot90(image, k), tf.image.rot90(heatmap, k)
    if training: ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(16).prefetch(tf.data.AUTOTUNE)


def predict(path: Path, inputs: np.ndarray) -> np.ndarray:
    """Run and dequantize the fully integer heatmap model."""
    interpreter = tf.lite.Interpreter(model_path=str(path)); interpreter.allocate_tensors(); inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]; predictions = []
    for sample in inputs:
        scale, zero = inp["quantization"]; interpreter.set_tensor(inp["index"], np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]); interpreter.invoke(); raw = interpreter.get_tensor(out["index"]).astype(np.float32); scale, zero = out["quantization"]; predictions.append((raw - zero) * scale)
    return np.concatenate(predictions)


def main() -> None:
    """Train, QAT-export, and evaluate the untouched corrected LittleGood test."""
    configure_gpu(); tf.keras.utils.set_random_seed(42); OUT.mkdir(parents=True, exist_ok=True)
    generic_train, generic_heat = load_arrays(DATA, "train"); generic_val, generic_val_heat = load_arrays(DATA, "val")
    student = {s: np.load(STUDENT / f"{s}.npz") for s in ("train", "val", "test")}
    x_train = np.concatenate((generic_train, student["train"]["inputs"])); y_train = np.concatenate((targets(decode(generic_heat)), targets(student["train"]["points"])))
    x_val = np.concatenate((generic_val, student["val"]["inputs"])); y_val = np.concatenate((targets(decode(generic_val_heat)), targets(student["val"]["points"])))
    model = build_model(); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tip_weighted_loss); model.fit(rotate_dataset(x_train, y_train, True), validation_data=rotate_dataset(x_val, y_val, False), epochs=14, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=tf.keras.optimizers.Adam(2e-4), loss=tip_weighted_loss); qat.fit(rotate_dataset(x_train, y_train, True), validation_data=rotate_dataset(x_val, y_val, False), epochs=5, verbose=2)
    path = OUT / "gauge_center_tip_direction_aug_v1_int8.tflite"; export_int8(qat, x_train, path); decoded = decode(predict(path, student["test"]["inputs"])); direction = decoded[:, 1] - decoded[:, 0]; direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6; prediction = np.stack((decoded[:, 0], decoded[:, 0] + direction * RADIUS), axis=1); errors = np.linalg.norm((prediction - student["test"]["points"]) * 160.0, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "radius": RADIUS, "bytes": path.stat().st_size}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
