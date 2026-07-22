"""Train a QAT hybrid model with direct center and spatial tip outputs.

The center is easy to regress directly, while the tip benefits from a dense
80x80 heatmap and subpixel decoder.  Samples are merged once; no oversampling
is used.  Both outputs are exported as full-int8 TFLite tensors.
"""

from __future__ import annotations

import json
import os
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
ARTIFACTS = ROOT / "artifacts" / "gauge_center_tip_hybrid_littlegood_v5"
INPUT_SIZE = 160
HEATMAP_SIZE = 80
BATCH_SIZE = 16
SEED = 42


def configure_gpu() -> None:
    """Cap the first visible GPU at the repository's 15 GB limit."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Apply two quantization-friendly convolution blocks."""
    for index in range(2):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{index}")(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn{index}")(x)
        x = keras.layers.ReLU(6.0, name=f"{name}_relu{index}")(x)
    return x


def build_model() -> keras.Model:
    """Build shared encoder, direct center head, and 80x80 tip heatmap head."""
    layers = keras.layers
    inputs = keras.Input((INPUT_SIZE, INPUT_SIZE, 2), name="ellipse_conditioned_input")
    e1 = conv_block(inputs, 16, "enc1")
    e2 = conv_block(layers.MaxPooling2D(2, name="pool1")(e1), 24, "enc2")
    e3 = conv_block(layers.MaxPooling2D(2, name="pool2")(e2), 40, "enc3")
    bottleneck = conv_block(layers.MaxPooling2D(2, name="pool3")(e3), 64, "bottleneck")

    # why: learned collapse avoids the quantized GlobalAveragePool mismatch.
    center = layers.Conv2D(64, 20, padding="valid", name="center_collapse")(bottleneck)
    center = layers.ReLU(6.0, name="center_collapse_relu")(center)
    center = layers.Flatten(name="center_flatten")(center)
    center = layers.Dense(48, activation="relu", name="center_dense")(center)
    center = layers.Dense(2, activation="sigmoid", name="center_xy")(center)

    tip = layers.UpSampling2D(2, interpolation="nearest", name="tip_up40")(bottleneck)
    tip = layers.Concatenate(name="tip_cat40")([tip, e3])
    tip = conv_block(tip, 40, "tip_dec40")
    tip = layers.UpSampling2D(2, interpolation="nearest", name="tip_up80")(tip)
    tip = layers.Concatenate(name="tip_cat80")([tip, e2])
    tip = conv_block(tip, 24, "tip_dec80")
    tip = layers.Conv2D(1, 1, activation="sigmoid", name="tip_heatmap")(tip)
    return keras.Model(inputs, {"center_xy": center, "tip_heatmap": tip}, name="gauge_center_tip_hybrid_v1")


def points(data_dir: Path, split: str) -> np.ndarray:
    """Read normalized center coordinates and tip coordinates from metadata."""
    rows = json.loads((data_dir / "metadata.json").read_text())["splits"][split]
    return np.asarray([row["center_xy_norm"] + row["tip_xy_norm"] for row in rows], dtype=np.float32)


def coordinate_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Train direct center coordinates with robust elementwise Huber loss."""
    error = tf.abs(y_true - y_pred)
    quadratic = tf.minimum(error, 0.03)
    linear = error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + 0.03 * linear)


def tip_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Emphasize Gaussian tip peaks while retaining background supervision."""
    weights = 1.0 + 48.0 * y_true
    return tf.reduce_mean(weights * tf.square(y_pred - y_true))


def export_int8(model: keras.Model, calibration: np.ndarray, path: Path) -> dict[str, object]:
    """Export and describe the full-integer multi-output TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([sample[None].astype(np.float32)] for sample in calibration[:256])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    path.write_bytes(blob)
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    return {
        "bytes": len(blob),
        "inputs": interpreter.get_input_details()[0]["shape"].tolist(),
        "outputs": [{"name": x["name"], "shape": x["shape"].tolist(), "quantization": x["quantization"]} for x in interpreter.get_output_details()],
        "operators": sorted({x["op_name"] for x in interpreter._get_ops_details() if x["op_name"] != "DELEGATE"}),
    }


def make_dataset(inputs: np.ndarray, centers: np.ndarray, heatmaps: np.ndarray, training: bool) -> tf.data.Dataset:
    """Build a dataset with geometry-aware crop jitter and updated centers."""
    ds = tf.data.Dataset.from_tensor_slices((inputs, centers, heatmaps))
    if training:
        ds = ds.shuffle(len(inputs), seed=SEED, reshuffle_each_iteration=True)

    def augment(image: tf.Tensor, center: tf.Tensor, heatmap: tf.Tensor) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """Translate the crop and center label together to mimic teacher error."""
        # why: padding must cover the full jitter range or crop extraction can
        # address outside the padded tensor on the largest translations.
        pad = 32
        # why: the detector teacher can leave the gauge noticeably off-center;
        # this models that crop error without duplicating any source image.
        dx = tf.random.uniform((), -32, 33, dtype=tf.int32, seed=SEED)
        dy = tf.random.uniform((), -32, 33, dtype=tf.int32, seed=SEED + 1)
        padded = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], constant_values=-1.0)
        image = tf.image.crop_to_bounding_box(padded, pad - dy, pad - dx, INPUT_SIZE, INPUT_SIZE)
        points = tf.clip_by_value(center + tf.cast(tf.stack([dx, dy]), tf.float32) / float(INPUT_SIZE), 0.0, 1.0)
        # why: the low-resolution tip heatmap must move with the same crop.
        shifted = tf.roll(heatmap, shift=[dy // 2, dx // 2], axis=[0, 1])
        return image, {"center_xy": points, "tip_heatmap": shifted}

    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda image, center, heatmap: (image, {"center_xy": center, "tip_heatmap": heatmap}), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def predict_int8(path: Path, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the int8 graph and return dequantized center and tip heatmaps."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()
    center_detail = next(x for x in outputs if x["shape"].tolist() == [1, 2])
    tip_detail = next(x for x in outputs if x["shape"].tolist() == [1, HEATMAP_SIZE, HEATMAP_SIZE, 1])
    predictions_center, predictions_tip = [], []
    for sample in inputs:
        quantized = np.clip(np.round(sample / inp["quantization"][0] + inp["quantization"][1]), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        def dequantize(detail: dict[str, object]) -> np.ndarray:
            raw = interpreter.get_tensor(detail["index"]).astype(np.float32)
            scale, zero = detail["quantization"]
            return (raw - zero) * scale
        predictions_center.append(dequantize(center_detail))
        predictions_tip.append(dequantize(tip_detail))
    return np.concatenate(predictions_center), np.concatenate(predictions_tip)


def decode_tip(heatmaps: np.ndarray) -> np.ndarray:
    """Decode a local weighted centroid around the strongest heatmap peak."""
    decoded = []
    for heatmap in heatmaps[:, :, :, 0]:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y0, y1 = max(0, y - 4), min(HEATMAP_SIZE, y + 5)
        x0, x1 = max(0, x - 4), min(HEATMAP_SIZE, x + 5)
        window = np.maximum(heatmap[y0:y1, x0:x1] - 0.03, 0.0) ** 2
        yy, xx = np.mgrid[y0:y1, x0:x1]
        total = float(window.sum())
        decoded.append(((float((xx * window).sum() / total) + 0.5) / HEATMAP_SIZE, (float((yy * window).sum() / total) + 0.5) / HEATMAP_SIZE) if total else ((x + 0.5) / HEATMAP_SIZE, (y + 0.5) / HEATMAP_SIZE))
    return np.asarray(decoded, dtype=np.float32)


def main() -> None:
    """Train, quantize, export, and evaluate the hybrid model."""
    configure_gpu()
    tf.keras.utils.set_random_seed(SEED)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    xb, _ = load_arrays(DATA, "train"); xv, _ = load_arrays(DATA, "val"); xt, _ = load_arrays(DATA, "test")
    tb, _ = load_arrays(TEMP_DATA, "train"); tv, _ = load_arrays(TEMP_DATA, "val"); tt, _ = load_arrays(TEMP_DATA, "test")
    x_train, x_val, x_test = np.concatenate((xb, tb)), np.concatenate((xv, tv)), np.concatenate((xt, tt))
    y_train = np.concatenate((points(DATA, "train"), points(TEMP_DATA, "train")))
    y_val = np.concatenate((points(DATA, "val"), points(TEMP_DATA, "val")))
    y_test = np.concatenate((points(DATA, "test"), points(TEMP_DATA, "test")))
    hm_train = np.concatenate((load_arrays(DATA, "train")[1][:, :, :, 1:], load_arrays(TEMP_DATA, "train")[1][:, :, :, 1:]))
    hm_val = np.concatenate((load_arrays(DATA, "val")[1][:, :, :, 1:], load_arrays(TEMP_DATA, "val")[1][:, :, :, 1:]))
    model = build_model()
    losses = {"center_xy": coordinate_loss, "tip_heatmap": tip_loss}
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses, loss_weights={"center_xy": 2.0, "tip_heatmap": 3.0})
    model.fit(make_dataset(x_train, y_train[:, :2], hm_train, True), validation_data=make_dataset(x_val, y_val[:, :2], hm_val, False), epochs=12, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=losses, loss_weights={"center_xy": 2.0, "tip_heatmap": 3.0})
    qat.fit(make_dataset(x_train, y_train[:, :2], hm_train, True), validation_data=make_dataset(x_val, y_val[:, :2], hm_val, False), epochs=4, verbose=2)
    path = ARTIFACTS / "gauge_center_tip_hybrid_v1_int8.tflite"
    contract = export_int8(qat, x_train, path)
    center, tip_map = predict_int8(path, x_test)
    decoded = decode_tip(tip_map)
    errors = np.linalg.norm((np.stack((center, decoded), axis=1) - y_test.reshape(-1, 2, 2)) * INPUT_SIZE, axis=2)
    report = {"samples": len(x_test), "center_within_8px": float(np.mean(errors[:, 0] <= 8)), "tip_within_8px": float(np.mean(errors[:, 1] <= 8)), "center_error_px_mean": float(errors[:, 0].mean()), "tip_error_px_mean": float(errors[:, 1].mean()), "contract": contract}
    (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
