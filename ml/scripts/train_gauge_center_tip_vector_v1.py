"""Train a direct int8 center/tip coordinate model without domain oversampling.

The heatmap model is accurate for centers but loses too many tips to coarse
peak selection.  This candidate predicts four normalized coordinates directly
and removes the runtime heatmap decoder.  It uses the naturally merged generic
and LittleGood corpora once each, with photometric-only augmentation so the
model learns invariances instead of memorizing repeated frames.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_gauge_center_tip_v1 import load_arrays


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
TEMP_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
ARTIFACTS = ROOT / "artifacts" / "gauge_center_tip_vector_littlegood_v7"
INPUT_SIZE = 160
BATCH_SIZE = 32
SEED = 42


def configure_gpu() -> None:
    """Apply the repository's 15 GB GPU cap before TensorFlow allocates memory."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_model() -> keras.Model:
    """Build a compact Conv/BN/ReLU coordinate regressor without MEAN pooling."""
    layers = keras.layers
    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 2), name="ellipse_conditioned_input")
    x = inputs
    for index, (filters, repeats) in enumerate(((16, 2), (24, 2), (40, 2), (64, 2))):
        for repeat in range(repeats):
            x = layers.Conv2D(filters, 3, strides=2 if repeat == 0 else 1, padding="same", use_bias=False, name=f"stage{index}_conv{repeat}")(x)
            x = layers.BatchNormalization(name=f"stage{index}_bn{repeat}")(x)
            x = layers.ReLU(6.0, name=f"stage{index}_relu{repeat}")(x)
    # why: a learned spatial collapse avoids the quantized GlobalAveragePool issue.
    # Four stride-2 stages reduce 160x160 to 10x10, so this kernel collapses
    # the remaining spatial grid exactly to 1x1 for the TFLite graph.
    x = layers.Conv2D(64, 10, padding="valid", use_bias=True, name="spatial_collapse")(x)
    x = layers.ReLU(6.0, name="spatial_collapse_relu")(x)
    x = layers.Flatten(name="spatial_flatten")(x)
    x = layers.Dense(64, activation="relu", name="head_relu")(x)
    outputs = layers.Dense(4, activation="sigmoid", name="center_tip_xy")(x)
    return keras.Model(inputs, outputs, name="gauge_center_tip_vector_v1")


def make_dataset(inputs: np.ndarray, targets: np.ndarray, training: bool) -> tf.data.Dataset:
    """Create a dataset with photometric-only augmentation on grayscale."""
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if training:
        ds = ds.shuffle(len(inputs), seed=SEED, reshuffle_each_iteration=True)

    def augment(image: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply photometric changes and occasional rotations with label updates."""
        gray, mask = image[..., :1], image[..., 1:]
        gray = tf.image.random_brightness(gray, max_delta=0.12, seed=SEED)
        gray = tf.image.random_contrast(gray, 0.75, 1.25, seed=SEED)
        image = tf.concat([tf.clip_by_value(gray, -1.0, 1.0), mask], axis=-1)
        # why: this simulates the center/scale error produced by a generalized
        # teacher crop, while using each source image only once per epoch.
        # why: padding must cover the full jitter range or crop extraction can
        # address outside the padded tensor on the largest translations.
        pad = 32
        # why: the detector teacher can leave the gauge noticeably off-center;
        # this models that crop error without duplicating any source image.
        dx = tf.random.uniform((), -32, 33, dtype=tf.int32, seed=SEED)
        dy = tf.random.uniform((), -32, 33, dtype=tf.int32, seed=SEED + 1)
        image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], constant_values=-1.0)
        image = tf.image.crop_to_bounding_box(image, pad - dy, pad - dx, INPUT_SIZE, INPUT_SIZE)
        points = tf.reshape(target, (2, 2)) + tf.cast(tf.stack([dx, dy]), tf.float32)[None, :] / float(INPUT_SIZE)
        return image, tf.reshape(tf.clip_by_value(points, 0.0, 1.0), (4,))

    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def coordinate_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Weight tip coordinates slightly more while retaining robust Huber loss."""
    # why: tips have larger residuals and are the harder target, so weighting
    # their coordinates changes the objective without duplicating any images.
    weights = tf.constant([1.0, 1.0, 3.0, 3.0], dtype=y_true.dtype)
    error = y_true - y_pred
    absolute = tf.abs(error)
    quadratic = tf.minimum(absolute, 0.03)
    linear = absolute - quadratic
    # why: keeping the coordinate axis explicit lets tip coordinates receive
    # higher weight without relying on a reduction-sensitive Keras helper.
    elementwise = 0.5 * tf.square(quadratic) + 0.03 * linear
    return tf.reduce_mean(elementwise * weights)


def export_int8(model: keras.Model, calibration: np.ndarray, path: Path) -> dict[str, object]:
    """Export a full-integer TFLite model and report its contract."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([sample[None].astype(np.float32)] for sample in calibration[:256])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    return {"bytes": len(blob), "input": {"shape": inp["shape"].tolist(), "quantization": inp["quantization"]}, "output": {"shape": out["shape"].tolist(), "quantization": out["quantization"]}, "operators": sorted({detail["op_name"] for detail in interpreter._get_ops_details() if detail["op_name"] != "DELEGATE"})}


def predict_int8(path: Path, inputs: np.ndarray) -> np.ndarray:
    """Run an int8 TFLite model and dequantize its four coordinates."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    predictions = []
    for sample in inputs:
        quantized = np.clip(np.round(sample / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"]).astype(np.float32)
        predictions.append((raw - out_zero) * out_scale)
    return np.concatenate(predictions, axis=0)


def main() -> None:
    """Train, QAT-finetune, export, and evaluate the direct coordinate model."""
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    x_train, y_train = load_arrays(DATA, "train")
    x_val, y_val = load_arrays(DATA, "val")
    x_test, y_test = load_arrays(DATA, "test")
    tx, ty = load_arrays(TEMP_DATA, "train")
    vx, vy = load_arrays(TEMP_DATA, "val")
    sx, sy = load_arrays(TEMP_DATA, "test")
    # The prepared heatmaps are converted back to point targets from metadata.
    def points(data_dir: Path, split: str) -> np.ndarray:
        """Load normalized center/tip coordinates from the split metadata."""
        rows = json.loads((data_dir / "metadata.json").read_text())["splits"][split]
        return np.asarray([row["center_xy_norm"] + row["tip_xy_norm"] for row in rows], dtype=np.float32)

    x_train = np.concatenate((x_train, tx))
    y_train = np.concatenate((points(DATA, "train"), points(TEMP_DATA, "train")))
    x_val = np.concatenate((x_val, vx))
    y_val = np.concatenate((points(DATA, "val"), points(TEMP_DATA, "val")))
    x_test = np.concatenate((x_test, sx))
    y_test = np.concatenate((points(DATA, "test"), points(TEMP_DATA, "test")))
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=coordinate_loss)
    model.fit(make_dataset(x_train, y_train, True), validation_data=make_dataset(x_val, y_val, False), epochs=15, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=coordinate_loss)
    qat.fit(make_dataset(x_train, y_train, True), validation_data=make_dataset(x_val, y_val, False), epochs=6, verbose=2)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    qat.save_weights(ARTIFACTS / "gauge_center_tip_vector_v1_qat.weights.h5")
    contract = export_int8(qat, x_train, ARTIFACTS / "gauge_center_tip_vector_v1_int8.tflite")
    prediction = predict_int8(ARTIFACTS / "gauge_center_tip_vector_v1_int8.tflite", x_test)
    errors = np.linalg.norm((prediction.reshape(-1, 2, 2) - y_test.reshape(-1, 2, 2)) * INPUT_SIZE, axis=2)
    report = {"samples": len(x_test), "center_within_8px": float(np.mean(errors[:, 0] <= 8.0)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8.0)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
