"""Train the wider geometry model with loss-weighted, non-duplicated domains."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_direction_aug_v1 import targets
from train_gauge_center_tip_direction_radius_v1 import export_int8, model_with_radius, predict, radius_targets
from train_gauge_center_tip_fullframe_v1 import decode, tip_weighted_loss
from train_gauge_center_tip_v1 import configure_gpu


ROOT = Path(__file__).resolve().parents[1]
GENERIC = ROOT / "tmp" / "generic_conditioned_wide_v1"
STUDENT = ROOT / "tmp" / "student_conditioned_wide_v1"
OUT = ROOT / "artifacts" / "gauge_center_tip_wide_weighted_littlegood_v1"
STUDENT_WEIGHT = 4.0


def weighted_dataset(inputs: np.ndarray, heatmaps: np.ndarray, radii: np.ndarray, weights: np.ndarray, training: bool) -> tf.data.Dataset:
    """Return one-pass samples with explicit LittleGood loss weights."""
    ds = tf.data.Dataset.from_tensor_slices((inputs, heatmaps, radii, weights))
    if training:
        ds = ds.shuffle(len(inputs), seed=42, reshuffle_each_iteration=True)

    def augment(image: tf.Tensor, heatmap: tf.Tensor, radius: tf.Tensor, weight: tf.Tensor) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
        """Rotate geometry and carry the per-source-domain weight unchanged."""
        k = tf.random.uniform((), 0, 4, dtype=tf.int32, seed=42)
        return tf.image.rot90(image, k), (tf.image.rot90(heatmap, k), radius), (weight, weight)

    def identity(image: tf.Tensor, heatmap: tf.Tensor, radius: tf.Tensor, weight: tf.Tensor) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
        """Keep validation geometry fixed while returning both output weights."""
        return image, (heatmap, radius), (weight, weight)

    return ds.map(augment if training else identity, num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load generic wide inputs and local points."""
    arrays = np.load(GENERIC / f"{split}.npz")
    return arrays["inputs"], arrays["points"]


def per_sample_heatmap_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Return one weighted heatmap loss per sample for Keras weighting."""
    channel_weight = tf.constant([1.0, 8.0], dtype=y_true.dtype)[None, None, None, :]
    weights = 1.0 + 48.0 * y_true * channel_weight
    return tf.reduce_mean(weights * tf.square(y_pred - y_true), axis=(1, 2, 3))


def per_sample_radius_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Return one robust scalar-radius loss per sample."""
    return tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred, delta=0.03), axis=-1)


def main() -> None:
    """Train, QAT-export, and score the untouched LittleGood test set."""
    configure_gpu()
    tf.keras.utils.set_random_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    gx, gp = load("train")
    gv, gvp = load("val")
    student = {split: np.load(STUDENT / f"{split}.npz") for split in ("train", "val", "test")}
    train_x = np.concatenate((gx, student["train"]["inputs"]))
    train_points = np.concatenate((gp, student["train"]["points"]))
    train_weights = np.concatenate((np.ones(len(gx), np.float32), np.full(len(student["train"]["inputs"]), STUDENT_WEIGHT, np.float32)))
    val_x = np.concatenate((gv, student["val"]["inputs"]))
    val_points = np.concatenate((gvp, student["val"]["points"]))
    val_weights = np.concatenate((np.ones(len(gv), np.float32), np.full(len(student["val"]["inputs"]), STUDENT_WEIGHT, np.float32)))
    train_h, val_h = targets(train_points), targets(val_points)
    train_r, val_r = radius_targets(train_points), radius_targets(val_points)
    model = model_with_radius()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=[per_sample_heatmap_loss, per_sample_radius_loss], loss_weights=[1.0, 4.0])
    model.fit(weighted_dataset(train_x, train_h, train_r, train_weights, True), validation_data=weighted_dataset(val_x, val_h, val_r, val_weights, False), epochs=14, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=[per_sample_heatmap_loss, per_sample_radius_loss], loss_weights=[1.0, 4.0])
    qat.fit(weighted_dataset(train_x, train_h, train_r, train_weights, True), validation_data=weighted_dataset(val_x, val_h, val_r, val_weights, False), epochs=5, verbose=2)
    path = OUT / "gauge_center_tip_wide_weighted_v1_int8.tflite"
    contract = export_int8(qat, train_x, path)
    heat, radius = predict(path, student["test"]["inputs"])
    decoded = decode(heat)
    direction = decoded[:, 1] - decoded[:, 0]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
    prediction = np.stack((decoded[:, 0], decoded[:, 0] + direction * radius * 0.5), axis=1)
    errors = np.linalg.norm((prediction - student["test"]["points"]) * 160.0, axis=2)
    report = {"samples": len(errors), "student_weight": STUDENT_WEIGHT, "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
