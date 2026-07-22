"""Train a fully integer ellipse-mask predictor for the STM32N6.

The earlier scalar ellipse regressors learned reasonable float outputs but
lost accuracy after conversion.  This model predicts a filled face mask on an
80x80 grid; firmware can recover the axis-aligned ellipse from the mask's
bounding box using only integer comparisons and min/max operations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras


IMAGE_SIZE = 320
MASK_SIZE = 80
SEED = 42


def configure_gpu() -> None:
    """Limit TensorFlow to 15 GB so WSL and the desktop retain headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build a compact grayscale encoder-decoder with an 80x80 mask output."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")

    def block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        """Apply two quantization-friendly Conv-BN-ReLU operations."""
        for index in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{index}")(x)
            x = layers.BatchNormalization(name=f"{name}_bn{index}")(x)
            x = layers.ReLU(6.0, name=f"{name}_relu{index}")(x)
        return x

    e1 = block(inputs, 16, "enc1")
    p1 = layers.MaxPooling2D(2, name="pool1")(e1)
    e2 = block(p1, 24, "enc2")
    p2 = layers.MaxPooling2D(2, name="pool2")(e2)
    e3 = block(p2, 40, "enc3")
    p3 = layers.MaxPooling2D(2, name="pool3")(e3)
    b = block(p3, 64, "bottleneck")
    u = layers.UpSampling2D(2, interpolation="nearest", name="up2")(b)
    u = layers.Concatenate(name="cat2")([u, e3])
    u = block(u, 40, "dec2")
    # The output is 80x80, while the largest live feature map is 160x160x16.
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="ellipse_mask")(u)
    return keras.Model(inputs, outputs, name="gauge_ellipse_mask_v1")


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load image paths and normalized ellipse targets from a prepared split."""
    image_paths = sorted((root / "images" / split).glob("*.png"))
    labels = []
    for image_path in image_paths:
        values = np.fromstring(
            (root / "labels" / split / f"{image_path.stem}.txt").read_text(), sep=" "
        )
        points = values[1:9].reshape(4, 2)
        low, high = points.min(axis=0), points.max(axis=0)
        center = (low + high) * 0.5
        radius = (high - low) * 0.5
        labels.append([center[0], center[1], radius[0], radius[1]])
    if not image_paths:
        raise FileNotFoundError(f"No ellipse samples under {root / 'images' / split}")
    return np.asarray([str(path) for path in image_paths]), np.asarray(labels, dtype=np.float32)


def dataset(paths: np.ndarray, targets: np.ndarray, batch: int, training: bool) -> tf.data.Dataset:
    """Create a streaming image-to-filled-ellipse-mask dataset."""
    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def decode(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Decode an image and rasterize its normalized ellipse on the target grid."""
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        image = tf.cast(image, tf.float32) / 255.0
        yy, xx = tf.meshgrid(
            (tf.range(MASK_SIZE, dtype=tf.float32) + 0.5) / MASK_SIZE,
            (tf.range(MASK_SIZE, dtype=tf.float32) + 0.5) / MASK_SIZE,
        )
        cx, cy, rx, ry = tf.unstack(target)
        mask = tf.cast(((xx - cx) / tf.maximum(rx, 1e-3)) ** 2 + ((yy - cy) / tf.maximum(ry, 1e-3)) ** 2 <= 1.0, tf.float32)
        return image, mask[..., None]

    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def mask_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Combine weighted BCE and Dice loss so the face interior is not ignored."""
    # why: a filled face covers a minority of the square when the gauge is small.
    weights = 1.0 + 4.0 * y_true
    bce = tf.reduce_mean(weights[..., 0] * keras.losses.binary_crossentropy(y_true, y_pred))
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    dice = tf.reduce_mean(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))
    return bce + dice


def export_int8(model: keras.Model, paths: np.ndarray, output: Path) -> dict[str, object]:
    """Export a full-integer TFLite graph and return its tensor contract."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative():
        """Yield representative grayscale inputs for activation calibration."""
        for path in paths[: min(256, len(paths))]:
            image = tf.io.decode_png(tf.io.read_file(path), channels=1)
            image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            yield [tf.cast(image[None], tf.float32) / 255.0]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    return {
        "bytes": len(blob),
        "input": {"shape": inp["shape"].tolist(), "quantization": inp["quantization"]},
        "output": {"shape": out["shape"].tolist(), "quantization": out["quantization"]},
        "operators": sorted({detail["op_name"] for detail in interpreter._get_ops_details() if detail["op_name"] != "DELEGATE"}),
    }


def predict_int8(path: Path, images: np.ndarray) -> np.ndarray:
    """Run the exported integer model and return dequantized masks."""
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    predictions = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"]).astype(np.float32)
        predictions.append((raw - out_zero) * out_scale)
    return np.concatenate(predictions, axis=0)


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """Recover normalized axis-aligned ellipse boxes from thresholded masks."""
    boxes = []
    for mask in masks[..., 0]:
        active = mask >= 0.5
        if not np.any(active):
            active.flat[int(np.argmax(mask))] = True
        yy, xx = np.where(active)
        low = np.array([xx.min(), yy.min()], dtype=np.float32) / MASK_SIZE
        high = np.array([xx.max() + 1, yy.max() + 1], dtype=np.float32) / MASK_SIZE
        boxes.append(np.concatenate(((low + high) * 0.5, (high - low) * 0.5)))
    return np.asarray(boxes, dtype=np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and evaluate the ellipse-mask candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "gauge_face_ellipse_v1_640_gray")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "gauge_ellipse_mask_littlegood_v1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    train_paths, train_targets = load_split(args.data, "train")
    val_paths, val_targets = load_split(args.data, "val")
    test_paths, test_targets = load_split(args.data, "test")
    train_ds = dataset(train_paths, train_targets, args.batch_size, True)
    val_ds = dataset(val_paths, val_targets, args.batch_size, False)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=mask_loss)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=mask_loss)
    qat.fit(train_ds, validation_data=val_ds, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "gauge_ellipse_mask_v1_qat.weights.h5")
    contract = export_int8(qat, train_paths, args.output / "gauge_ellipse_mask_v1_int8.tflite")

    test_images = np.stack([tf.image.resize(tf.io.decode_png(tf.io.read_file(path), channels=1), [IMAGE_SIZE, IMAGE_SIZE]).numpy() for path in test_paths]).astype(np.float32) / 255.0
    predicted = boxes_from_masks(predict_int8(args.output / "gauge_ellipse_mask_v1_int8.tflite", test_images))
    true_boxes = np.concatenate((test_targets[:, :2] - test_targets[:, 2:], test_targets[:, :2] + test_targets[:, 2:]), axis=1)
    pred_boxes = np.concatenate((predicted[:, :2] - predicted[:, 2:], predicted[:, :2] + predicted[:, 2:]), axis=1)
    intersection = np.maximum(0.0, np.minimum(true_boxes[:, 2:], pred_boxes[:, 2:]) - np.maximum(true_boxes[:, :2], pred_boxes[:, :2]))
    inter_area = intersection[:, 0] * intersection[:, 1]
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    iou = inter_area / np.maximum(true_area + pred_area - inter_area, 1e-6)
    center_error = np.linalg.norm(predicted[:, :2] - test_targets[:, :2], axis=1) * 640.0
    report = {"samples": len(test_paths), "int8_box_iou_mean": float(iou.mean()), "int8_box_iou_ge_0_5": float(np.mean(iou >= 0.5)), "int8_center_within_16px": float(np.mean(center_error <= 16.0)), "int8_center_error_px_mean": float(center_error.mean()), "int8_radius_error_px_mean": (np.abs(predicted[:, 2:] - test_targets[:, 2:]).mean(axis=0) * 640.0).round(4).tolist(), "contract": contract}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
