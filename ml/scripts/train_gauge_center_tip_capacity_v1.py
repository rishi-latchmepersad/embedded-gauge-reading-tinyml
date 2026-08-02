"""Train a higher-capacity, fully integer ellipse-conditioned keypoint model.

The input contract remains the deployed 160x160 grayscale crop plus ellipse
mask.  The extra channels are deliberate: the prior compact model localized
the center reasonably well but did not preserve enough needle detail for the
tip.  This candidate still uses only bounded convolutional activations and a
small scalar radius head so its SRAM budget can be checked before packaging.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_direction_aug_v1 import targets
from train_gauge_center_tip_fullframe_v1 import decode, tip_weighted_loss
from train_gauge_center_tip_v1 import configure_gpu, load_arrays


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
STUDENT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUT = ROOT / "artifacts" / "gauge_center_tip_capacity_littlegood_v1"
RADIUS_SCALE = 0.5


def block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Apply two quantization-friendly convolution stages."""
    for index in range(2):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{index}")(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn{index}")(x)
        # why: ReLU6 keeps activation ranges bounded for representative-data QAT.
        x = keras.layers.ReLU(6.0, name=f"{name}_relu{index}")(x)
    return x


def build_model() -> keras.Model:
    """Build the larger 160-to-80 heatmap encoder and radius head."""
    inputs = keras.Input((160, 160, 2), name="ellipse_conditioned_input")
    e1 = block(inputs, 24, "enc1")
    p1 = keras.layers.MaxPooling2D(2, name="pool1")(e1)
    e2 = block(p1, 40, "enc2")
    p2 = keras.layers.MaxPooling2D(2, name="pool2")(e2)
    e3 = block(p2, 64, "enc3")
    p3 = keras.layers.MaxPooling2D(2, name="pool3")(e3)
    bottleneck = block(p3, 96, "bottleneck")
    radius = keras.layers.Conv2D(24, 20, padding="valid", use_bias=False, name="radius_collapse")(bottleneck)
    radius = keras.layers.BatchNormalization(name="radius_bn")(radius)
    radius = keras.layers.ReLU(6.0, name="radius_relu")(radius)
    radius = keras.layers.Flatten(name="radius_flatten")(radius)
    radius = keras.layers.Dense(32, activation="relu", name="radius_dense")(radius)
    radius = keras.layers.Dense(1, activation="sigmoid", name="radius")(radius)
    u2 = keras.layers.UpSampling2D(2, interpolation="nearest", name="up2")(bottleneck)
    u2 = keras.layers.Concatenate(name="cat2")([u2, e3])
    u2 = block(u2, 64, "dec2")
    u1 = keras.layers.UpSampling2D(2, interpolation="nearest", name="up1")(u2)
    u1 = keras.layers.Concatenate(name="cat1")([u1, e2])
    u1 = block(u1, 40, "dec1")
    output = keras.layers.Conv2D(2, 1, activation="sigmoid", name="center_tip_heatmaps")(u1)
    return keras.Model(inputs, [output, radius], name="gauge_center_tip_capacity")


def radius_targets(points: np.ndarray) -> np.ndarray:
    """Encode local center-to-tip distances as a normalized scalar."""
    return (np.linalg.norm(points[:, 1] - points[:, 0], axis=1, keepdims=True) / RADIUS_SCALE).astype(np.float32)


def dataset(inputs: np.ndarray, heatmaps: np.ndarray, radii: np.ndarray, training: bool) -> tf.data.Dataset:
    """Create one-pass training data with quarter-turn geometry augmentation."""
    ds = tf.data.Dataset.from_tensor_slices((inputs, heatmaps, radii))
    if training:
        ds = ds.shuffle(len(inputs), seed=42, reshuffle_each_iteration=True)

    def augment(image: tf.Tensor, heatmap: tf.Tensor, radius: tf.Tensor) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        """Rotate pixels and heatmaps together without duplicating source files."""
        k = tf.random.uniform((), 0, 4, dtype=tf.int32, seed=42)
        return tf.image.rot90(image, k), (tf.image.rot90(heatmap, k), radius)

    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda image, heatmap, radius: (image, (heatmap, radius)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(16).prefetch(tf.data.AUTOTUNE)


def export_int8(model: keras.Model, calibration: np.ndarray, path: Path) -> dict[str, object]:
    """Export the QAT model with int8 inputs, outputs, and built-in operators."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    indices = np.linspace(0, len(calibration) - 1, min(256, len(calibration)), dtype=int)
    converter.representative_dataset = lambda: ([calibration[i][None].astype(np.float32)] for i in indices)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    path.write_bytes(blob)
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    return {"bytes": len(blob), "input": interpreter.get_input_details()[0]["shape"].tolist(), "outputs": [item["shape"].tolist() for item in interpreter.get_output_details()]}


def predict(path: Path, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the full-int8 graph and dequantize heatmap and radius outputs."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    heat_detail = next(item for item in output_details if len(item["shape"]) == 4)
    radius_detail = next(item for item in output_details if item["shape"][-1] == 1 and len(item["shape"]) == 2)
    heatmaps, radii = [], []
    for sample in inputs:
        scale, zero = input_detail["quantization"]
        encoded = np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)[None]
        interpreter.set_tensor(input_detail["index"], encoded)
        interpreter.invoke()
        for detail, destination in ((heat_detail, heatmaps), (radius_detail, radii)):
            raw = interpreter.get_tensor(detail["index"]).astype(np.float32)
            scale, zero = detail["quantization"]
            destination.append((raw - zero) * scale)
    return np.concatenate(heatmaps), np.concatenate(radii)


def main() -> None:
    """Train, quantize, evaluate the untouched test set, and write a report."""
    configure_gpu()
    tf.keras.utils.set_random_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    generic_train, generic_heat = load_arrays(DATA, "train")
    generic_val, generic_val_heat = load_arrays(DATA, "val")
    student = {split: np.load(STUDENT / f"{split}.npz") for split in ("train", "val", "test")}
    generic_points = decode(generic_heat)
    generic_val_points = decode(generic_val_heat)
    x_train = np.concatenate((generic_train, student["train"]["inputs"]))
    h_train = np.concatenate((targets(generic_points), targets(student["train"]["points"])))
    r_train = np.concatenate((radius_targets(generic_points), radius_targets(student["train"]["points"])))
    x_val = np.concatenate((generic_val, student["val"]["inputs"]))
    h_val = np.concatenate((targets(generic_val_points), targets(student["val"]["points"])))
    r_val = np.concatenate((radius_targets(generic_val_points), radius_targets(student["val"]["points"])))
    losses = [tip_weighted_loss, keras.losses.Huber(0.03)]
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses, loss_weights=[1.0, 4.0])
    model.fit(dataset(x_train, h_train, r_train, True), validation_data=dataset(x_val, h_val, r_val, False), epochs=14, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=losses, loss_weights=[1.0, 4.0])
    qat.fit(dataset(x_train, h_train, r_train, True), validation_data=dataset(x_val, h_val, r_val, False), epochs=5, verbose=2)
    path = OUT / "gauge_center_tip_capacity_v1_int8.tflite"
    contract = export_int8(qat, x_train, path)
    heat, radius = predict(path, student["test"]["inputs"])
    decoded = decode(heat)
    direction = decoded[:, 1] - decoded[:, 0]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-6
    prediction = np.stack((decoded[:, 0], decoded[:, 0] + direction * radius * RADIUS_SCALE), axis=1)
    errors = np.linalg.norm((prediction - student["test"]["points"]) * 160.0, axis=2)
    report = {"samples": len(errors), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
