"""Train an int8 direction heatmap model with a learned needle radius."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_direction_aug_v1 import targets
from train_gauge_center_tip_fullframe_v1 import decode, tip_weighted_loss
from train_gauge_center_tip_v1 import build_model, configure_gpu, load_arrays


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / ("gauge_center_tip_direction_radius_littlegood_v3" if os.environ.get("ARBITRARY_ROTATION") == "1" else "gauge_center_tip_direction_radius_littlegood_v4" if os.environ.get("STUDENT_ONLY") == "1" and os.environ.get("STUDENT_135") == "1" else "gauge_center_tip_direction_radius_littlegood_v2" if os.environ.get("STUDENT_ONLY") == "1" else "gauge_center_tip_direction_radius_littlegood_v1")
RADIUS_SCALE = 0.5


def radius_targets(points: np.ndarray) -> np.ndarray:
    """Encode local center-to-tip distance into a normalized scalar."""
    return (np.linalg.norm(points[:, 1] - points[:, 0], axis=1, keepdims=True) / RADIUS_SCALE).astype(np.float32)


def model_with_radius() -> keras.Model:
    """Attach a small scalar radius head to the compact heatmap encoder."""
    base = build_model(); feature = base.get_layer("bottleneck_relu1").output; layers = keras.layers
    radius = layers.Conv2D(16, 20, padding="valid", use_bias=False, name="radius_collapse")(feature)
    radius = layers.BatchNormalization(name="radius_bn")(radius); radius = layers.ReLU(6.0, name="radius_relu")(radius); radius = layers.Flatten(name="radius_flatten")(radius)
    radius = layers.Dense(24, activation="relu", name="radius_dense")(radius); radius = layers.Dense(1, activation="sigmoid", name="radius")(radius)
    return keras.Model(base.input, [base.output, radius], name="gauge_center_tip_direction_radius")


def dataset(inputs: np.ndarray, heatmaps: np.ndarray, radii: np.ndarray, training: bool) -> tf.data.Dataset:
    """Build a rotation-augmented multi-output dataset."""
    ds = tf.data.Dataset.from_tensor_slices((inputs, heatmaps, radii))
    if training: ds = ds.shuffle(len(inputs), seed=42, reshuffle_each_iteration=True)
    def augment(image: tf.Tensor, heatmap: tf.Tensor, radius: tf.Tensor) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Rotate image and heatmaps while leaving scalar radius unchanged."""
        if os.environ.get("ARBITRARY_ROTATION") == "1":
            angle = tf.random.uniform((), -np.pi, np.pi, seed=42)
            cosine, sine = tf.cos(angle), tf.sin(angle)
            transform = tf.stack((cosine, sine, (1.0 - cosine) * 79.5 - sine * 79.5, -sine, cosine, sine * 79.5 + (1.0 - cosine) * 79.5, 0.0, 0.0))[None]
            image = tf.raw_ops.ImageProjectiveTransformV3(images=image[None], transforms=transform, output_shape=[160, 160], interpolation="BILINEAR", fill_mode="CONSTANT", fill_value=-1.0)[0]
            heatmap_transform = tf.stack((cosine, sine, (1.0 - cosine) * 39.5 - sine * 39.5, -sine, cosine, sine * 39.5 + (1.0 - cosine) * 39.5, 0.0, 0.0))[None]
            heatmap = tf.raw_ops.ImageProjectiveTransformV3(images=heatmap[None], transforms=heatmap_transform, output_shape=[80, 80], interpolation="BILINEAR", fill_mode="CONSTANT", fill_value=0.0)[0]
            return image, (heatmap, radius)
        k = tf.random.uniform((), 0, 4, dtype=tf.int32, seed=42)
        return tf.image.rot90(image, k), (tf.image.rot90(heatmap, k), radius)
    if training: ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else: ds = ds.map(lambda image, heatmap, radius: (image, (heatmap, radius)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(16).prefetch(tf.data.AUTOTUNE)


def export_int8(model: keras.Model, calibration: np.ndarray, path: Path) -> dict[str, object]:
    """Export and describe the multi-output full-int8 graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model); converter.optimizations = [tf.lite.Optimize.DEFAULT]
    indices = np.linspace(0, len(calibration) - 1, min(256, len(calibration)), dtype=int); converter.representative_dataset = lambda: ([calibration[i][None].astype(np.float32)] for i in indices)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; converter.inference_input_type = tf.int8; converter.inference_output_type = tf.int8; blob = converter.convert(); path.write_bytes(blob)
    interpreter = tf.lite.Interpreter(model_content=blob); interpreter.allocate_tensors(); return {"bytes": len(blob), "input": interpreter.get_input_details()[0]["shape"].tolist(), "outputs": [x["shape"].tolist() for x in interpreter.get_output_details()]}


def predict(path: Path, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the full-int8 model and return heatmaps plus normalized radius."""
    interpreter = tf.lite.Interpreter(model_path=str(path)); interpreter.allocate_tensors(); inp = interpreter.get_input_details()[0]; outputs = interpreter.get_output_details(); heat = next(x for x in outputs if len(x["shape"]) == 4); radius = next(x for x in outputs if x["shape"][-1] == 1 and len(x["shape"]) == 2); h_values=[]; r_values=[]
    for sample in inputs:
        scale, zero = inp["quantization"]; interpreter.set_tensor(inp["index"], np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]); interpreter.invoke()
        for detail, values in ((heat, h_values), (radius, r_values)):
            raw = interpreter.get_tensor(detail["index"]).astype(np.float32); scale, zero = detail["quantization"]; values.append((raw - zero) * scale)
    return np.concatenate(h_values), np.concatenate(r_values)


def main() -> None:
    """Train, QAT-export, and score the corrected untouched LittleGood test."""
    configure_gpu(); tf.keras.utils.set_random_seed(42); OUT.mkdir(parents=True, exist_ok=True)
    generic_train, generic_heat = load_arrays(DATA, "train"); generic_val, generic_val_heat = load_arrays(DATA, "val"); student = {s: np.load(STUDENT / f"{s}.npz") for s in ("train", "val", "test")}
    generic_points = decode(generic_heat); generic_val_points = decode(generic_val_heat)
    if os.environ.get("STUDENT_ONLY") == "1":
        x_train, h_train, r_train = student["train"]["inputs"], targets(student["train"]["points"]), radius_targets(student["train"]["points"])
        x_val, h_val, r_val = student["val"]["inputs"], targets(student["val"]["points"]), radius_targets(student["val"]["points"])
    else:
        x_train = np.concatenate((generic_train, student["train"]["inputs"])); h_train = np.concatenate((targets(generic_points), targets(student["train"]["points"]))); r_train = np.concatenate((radius_targets(generic_points), radius_targets(student["train"]["points"])))
        x_val = np.concatenate((generic_val, student["val"]["inputs"])); h_val = np.concatenate((targets(generic_val_points), targets(student["val"]["points"]))); r_val = np.concatenate((radius_targets(generic_val_points), radius_targets(student["val"]["points"])))
    losses = [tip_weighted_loss, keras.losses.Huber(0.03)]; model = model_with_radius(); model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses, loss_weights=[1.0, 4.0]); model.fit(dataset(x_train, h_train, r_train, True), validation_data=dataset(x_val, h_val, r_val, False), epochs=14, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model); qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=losses, loss_weights=[1.0, 4.0]); qat.fit(dataset(x_train, h_train, r_train, True), validation_data=dataset(x_val, h_val, r_val, False), epochs=5, verbose=2)
    path = OUT / "gauge_center_tip_direction_radius_v1_int8.tflite"; contract = export_int8(qat, x_train, path); heat, radius = predict(path, student["test"]["inputs"]); decoded = decode(heat); direction = decoded[:, 1] - decoded[:, 0]; direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6; prediction = np.stack((decoded[:, 0], decoded[:, 0] + direction * radius * RADIUS_SCALE), axis=1); errors = np.linalg.norm((prediction - student["test"]["points"]) * 160.0, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
