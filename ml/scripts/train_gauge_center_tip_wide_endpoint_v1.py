"""Train a mixed-domain direct endpoint-heatmap model on the 1.6x crop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_fullframe_v1 import decode, tip_weighted_loss
from train_gauge_center_tip_v1 import build_model, configure_gpu, export_int8


ROOT = Path(__file__).resolve().parents[1]
GENERIC = ROOT / "tmp" / "generic_conditioned_wide_v1"
STUDENT = ROOT / "tmp" / "student_conditioned_wide_v1"
OUT = ROOT / "artifacts" / "gauge_center_tip_wide_endpoint_littlegood_v1"


def endpoint_targets(points: np.ndarray) -> np.ndarray:
    """Rasterize the actual center and tip, not a fixed-radius proxy."""
    result = np.zeros((len(points), 80, 80, 2), dtype=np.float32)
    yy, xx = np.mgrid[0:80, 0:80]
    for index, pair in enumerate(points):
        for channel, point in enumerate(pair):
            px, py = point * 80.0 - 0.5
            result[index, ..., channel] = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * 2.2**2))
    return result


def dataset(inputs: np.ndarray, targets: np.ndarray, training: bool) -> tf.data.Dataset:
    """Create one-pass photometric and quarter-turn augmented samples."""
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if training:
        ds = ds.shuffle(len(inputs), seed=42, reshuffle_each_iteration=True)

    def augment(image: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Rotate both image and endpoint targets consistently."""
        k = tf.random.uniform((), 0, 4, dtype=tf.int32, seed=42)
        return tf.image.rot90(image, k), tf.image.rot90(target, k)

    return ds.map(augment if training else lambda image, target: (image, target), num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    """Train, QAT-export, and score all 97 untouched LittleGood test frames."""
    configure_gpu()
    tf.keras.utils.set_random_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    generic_train = np.load(GENERIC / "train.npz")
    generic_val = np.load(GENERIC / "val.npz")
    student = {split: np.load(STUDENT / f"{split}.npz") for split in ("train", "val", "test")}
    train_x = np.concatenate((generic_train["inputs"], student["train"]["inputs"]))
    train_y = endpoint_targets(np.concatenate((generic_train["points"], student["train"]["points"])))
    val_x = np.concatenate((generic_val["inputs"], student["val"]["inputs"]))
    val_y = endpoint_targets(np.concatenate((generic_val["points"], student["val"]["points"])))
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=tip_weighted_loss)
    model.fit(dataset(train_x, train_y, True), validation_data=dataset(val_x, val_y, False), epochs=14, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=tip_weighted_loss)
    qat.fit(dataset(train_x, train_y, True), validation_data=dataset(val_x, val_y, False), epochs=5, verbose=2)
    path = OUT / "gauge_center_tip_wide_endpoint_v1_int8.tflite"
    export_int8(qat, train_x, path)
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    predictions = []
    for sample in student["test"]["inputs"]:
        scale, zero = input_detail["quantization"]
        interpreter.set_tensor(input_detail["index"], np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        scale, zero = output_detail["quantization"]
        predictions.append((raw - zero) * scale)
    errors = np.linalg.norm((decode(np.concatenate(predictions)) - student["test"]["points"]) * 160.0, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "bytes": path.stat().st_size}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
