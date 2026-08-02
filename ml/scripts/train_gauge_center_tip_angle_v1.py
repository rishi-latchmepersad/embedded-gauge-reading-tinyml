"""Train an int8 center, direction, and radius regression head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_v1 import configure_gpu


ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_angle_littlegood_v2"
SIZE = 160


def build_model() -> keras.Model:
    """Build a small shared encoder with explicit geometric outputs."""
    layers = keras.layers
    inputs = keras.Input((SIZE, SIZE, 2), name="ellipse_conditioned_input")
    x = inputs
    for stage, (filters, repeats) in enumerate(((16, 2), (24, 2), (40, 2), (64, 2))):
        for repeat in range(repeats):
            x = layers.Conv2D(filters, 3, strides=2 if repeat == 0 else 1, padding="same", use_bias=False, name=f"stage{stage}_conv{repeat}")(x)
            x = layers.BatchNormalization(name=f"stage{stage}_bn{repeat}")(x)
            x = layers.ReLU(6.0, name=f"stage{stage}_relu{repeat}")(x)
    x = layers.Conv2D(64, 10, padding="valid", use_bias=True, name="spatial_collapse")(x)
    x = layers.ReLU(6.0, name="spatial_collapse_relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    center_radius = layers.Dense(3, activation="sigmoid", name="center_radius")(x)
    direction = layers.Dense(2, activation="tanh", name="direction")(x)
    return keras.Model(inputs, [center_radius, direction])


def targets(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert local center/tip points into center/radius and unit direction."""
    center, tip = points[:, 0], points[:, 1]
    vector = tip - center
    radius = np.linalg.norm(vector, axis=1, keepdims=True)
    direction = vector / (radius + 1e-6)
    return np.concatenate((center, radius), axis=1), direction


def main() -> None:
    """Train, QAT-export, and evaluate explicit geometric predictions."""
    configure_gpu(); tf.keras.utils.set_random_seed(42)
    train = np.load(STUDENT / "train.npz"); val = np.load(STUDENT / "val.npz"); test = np.load(STUDENT / "test.npz")
    train_geometry, train_direction = targets(train["points"]); val_geometry, val_direction = targets(val["points"]); test_geometry, test_direction = targets(test["points"])
    def augment(image: tf.Tensor, geometry: tf.Tensor, direction: tf.Tensor) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Rotate one source frame and its geometric targets together."""
        quarter_turn = tf.random.uniform((), 0, 4, dtype=tf.int32, seed=42)
        image = tf.image.rot90(image, k=quarter_turn)
        center = geometry[:2]
        center = tf.switch_case(quarter_turn, branch_fns=(
            lambda: center,
            lambda: tf.stack((center[1], 1.0 - center[0])),
            lambda: 1.0 - center,
            lambda: tf.stack((1.0 - center[1], center[0])),
        ))
        direction = tf.switch_case(quarter_turn, branch_fns=(
            lambda: direction,
            lambda: tf.stack((direction[1], -direction[0])),
            lambda: -direction,
            lambda: tf.stack((-direction[1], direction[0])),
        ))
        return image, (tf.concat((center, geometry[2:]), axis=0), direction)

    train_ds = tf.data.Dataset.from_tensor_slices((train["inputs"], train_geometry, train_direction)).shuffle(len(train["inputs"]), seed=42).map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((val["inputs"], val_geometry, val_direction)).map(lambda image, geometry, direction: (image, (geometry, direction))).batch(16)
    model = build_model(); model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=["mse", "mse"], loss_weights=[1.0, 3.0])
    model.fit(train_ds, validation_data=val_ds, epochs=15, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=["mse", "mse"], loss_weights=[1.0, 3.0])
    qat.fit(train_ds, validation_data=val_ds, epochs=6, verbose=2)
    # why: export_int8 accepts a single-output Keras model, so this candidate
    # is kept as a diagnostic until its multi-output TFLite contract is stable.
    OUT.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(qat); converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([train["inputs"][i:i+1]] for i in np.linspace(0, len(train["inputs"]) - 1, min(256, len(train["inputs"])), dtype=int))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; converter.inference_input_type = tf.int8; converter.inference_output_type = tf.int8
    path = OUT / "gauge_center_tip_angle_v1_int8.tflite"; path.write_bytes(converter.convert())
    interpreter = tf.lite.Interpreter(model_path=str(path)); interpreter.allocate_tensors(); inp = interpreter.get_input_details()[0]; outputs = sorted(interpreter.get_output_details(), key=lambda item: int(item["shape"][-1]), reverse=True); predictions = []
    for sample in test["inputs"]:
        scale, zero = inp["quantization"]; interpreter.set_tensor(inp["index"], np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]); interpreter.invoke(); values=[]
        for output in outputs:
            raw = interpreter.get_tensor(output["index"]).astype(np.float32); scale, zero = output["quantization"]; values.append((raw - zero) * scale)
        predictions.append(values)
    geometry = np.concatenate([p[0] for p in predictions]); direction = np.concatenate([p[1] for p in predictions]); direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6; points = np.stack((geometry[:, :2], geometry[:, :2] + direction * geometry[:, 2:3]), axis=1); errors = np.linalg.norm((points - test["points"]) * SIZE, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "bytes": path.stat().st_size}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
