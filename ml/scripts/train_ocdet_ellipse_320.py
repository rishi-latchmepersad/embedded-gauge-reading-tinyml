#!/usr/bin/env python3
"""Train an OCDet-inspired class-agnostic gauge-center detector.

The model uses a MobileNet-like convolutional encoder, a semantic FPN, and a
single stride-4 dense head.  Every spatial cell can nominate the gauge, while
the winning cell predicts a subpixel center and ellipse radii.  This keeps the
architecture general across gauge types and directly prioritizes center error.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_ellipse_all_test_sets import _load_zip, _metrics  # noqa: E402
from train_ellipse_robust_384 import (  # noqa: E402
    BOARD_TRAIN_ZIPS,
    load_zips,
    make_scale_augmented_training_set,
)

IMAGE_SIZE = 320
GRID_SIZE = IMAGE_SIZE // 4
SEED = 42
CHANNELS = 6  # objectness, cell-x, cell-y, radius-x, radius-y, quality


def configure_gpu() -> None:
    """Limit TensorFlow to the approved 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def conv_block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    """Apply a quantization-friendly convolution, normalization, and ReLU."""
    layers = keras.layers
    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def lateral(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Project one encoder level into the shared FPN channel width."""
    layers = keras.layers
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_model() -> keras.Model:
    """Build a small Semantic-FPN detector with an explicit spatial contract."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    features: list[tf.Tensor] = []
    # why: the 1/4 feature map is kept alive because test_2 contains very small
    # faces whose center would be discarded by a 1/16 or 1/32-only detector.
    for stage, filters in enumerate((24, 32, 48, 64, 96)):
        x = conv_block(x, filters, 2, f"enc{stage}_down")
        x = conv_block(x, filters, 1, f"enc{stage}_refine")
        features.append(x)

    # why: this top-down pyramid gives the shallow center head global context
    # without requiring a large backbone or a global pooling bottleneck.
    p5 = lateral(features[4], 48, "fpn5")
    p4 = layers.Add(name="fpn4_add")([lateral(features[3], 48, "fpn4"), layers.UpSampling2D(2, interpolation="nearest")(p5)])
    p4 = conv_block(p4, 48, 1, "fpn4_refine")
    p3 = layers.Add(name="fpn3_add")([lateral(features[2], 48, "fpn3"), layers.UpSampling2D(2, interpolation="nearest")(p4)])
    p3 = conv_block(p3, 48, 1, "fpn3_refine")
    p2 = layers.Add(name="fpn2_add")([lateral(features[1], 48, "fpn2"), layers.UpSampling2D(2, interpolation="nearest")(p3)])
    p2 = conv_block(p2, 48, 1, "fpn2_refine")

    # why: all six values remain spatially aligned, making the exported graph
    # easy to decode on the NPU without custom tensor reshuffling.
    head = conv_block(p2, 32, 1, "head")
    dense = layers.Conv2D(CHANNELS, 1, activation="sigmoid", name="dense_contract")(head)
    return keras.Model(
        inputs,
        layers.Flatten(name="flat_contract")(dense),
        name="ocdet_ellipse_320",
    )


def make_targets(targets: np.ndarray) -> np.ndarray:
    """Create ellipse-shaped centerness maps and positive-cell geometry labels."""
    result = np.zeros((len(targets), GRID_SIZE, GRID_SIZE, CHANNELS), dtype=np.float32)
    yy, xx = np.mgrid[:GRID_SIZE, :GRID_SIZE]
    for index, (cx, cy, rx, ry, _quality) in enumerate(targets):
        gx = cx * GRID_SIZE
        gy = cy * GRID_SIZE
        # why: an anisotropic Gaussian follows the observed ellipse, so an
        # elongated perspective face contributes useful positive gradients.
        sigma_x = max(1.0, min(8.0, rx * GRID_SIZE * 0.35))
        sigma_y = max(1.0, min(8.0, ry * GRID_SIZE * 0.35))
        result[index, ..., 0] = np.exp(
            -0.5 * (((xx + 0.5 - gx) / sigma_x) ** 2 + ((yy + 0.5 - gy) / sigma_y) ** 2)
        )
        cell_x = int(np.clip(np.floor(gx), 0, GRID_SIZE - 1))
        cell_y = int(np.clip(np.floor(gy), 0, GRID_SIZE - 1))
        result[index, cell_y, cell_x, 0] = 1.0
        # The offset is a fraction within the winning cell, always in [0, 1).
        result[index, cell_y, cell_x, 1:5] = [
            np.clip(gx - cell_x, 0.0, 0.999),
            np.clip(gy - cell_y, 0.0, 0.999),
            np.clip(rx, 0.001, 1.0),
            np.clip(ry, 0.001, 1.0),
        ]
        result[index, cell_y, cell_x, 5] = 1.0
    return result.reshape(len(targets), -1)


def make_flip_augmented_training_set(
    images: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Add horizontal, vertical, and 180-degree views with corrected centers."""
    rng = np.random.default_rng(SEED + 7)
    # why: test_2 contains mirrored/framed variants; adding only half as many
    # views keeps the memory and training time bounded while breaking the
    # model's dependence on the original image orientation.
    indices = rng.choice(len(images), size=max(1, len(images) // 2), replace=False)
    modes = rng.integers(1, 4, size=len(indices))
    flipped_images: list[np.ndarray] = []
    flipped_targets: list[np.ndarray] = []
    for index, mode in zip(indices, modes):
        image = images[index]
        target = targets[index].copy()
        if mode in (1, 3):
            image = image[:, ::-1]
            target[0] = 1.0 - target[0]
        if mode in (2, 3):
            image = image[::-1, :]
            target[1] = 1.0 - target[1]
        flipped_images.append(image.copy())
        flipped_targets.append(target)
    return (
        np.concatenate([images, np.asarray(flipped_images)], axis=0),
        np.concatenate([targets, np.asarray(flipped_targets)], axis=0),
    )


def _focal_heatmap_loss(true_heat: tf.Tensor, pred_heat: tf.Tensor) -> tf.Tensor:
    """Apply balanced continuous focal loss to the centerness map."""
    target = tf.clip_by_value(true_heat, 0.0, 1.0)
    prediction = tf.clip_by_value(pred_heat, 1e-4, 1.0 - 1e-4)
    positive = tf.cast(target >= 0.95, tf.float32)
    negative = 1.0 - positive
    negative_weight = tf.pow(1.0 - target, 4.0)
    positive_loss = -tf.pow(1.0 - prediction, 2.0) * tf.math.log(prediction) * positive
    negative_loss = -tf.pow(prediction, 2.0) * tf.math.log(1.0 - prediction) * negative_weight * negative
    normalizer = tf.reduce_sum(positive, axis=(1, 2)) + 1.0
    return tf.reduce_mean(tf.reduce_sum(positive_loss + negative_loss, axis=(1, 2)) / normalizer)


def detector_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Optimize center selection first, then geometry only at the winner cell."""
    true_values = tf.reshape(y_true, (-1, GRID_SIZE, GRID_SIZE, CHANNELS))
    pred_values = tf.reshape(y_pred, (-1, GRID_SIZE, GRID_SIZE, CHANNELS))
    heat_loss = _focal_heatmap_loss(true_values[..., 0], pred_values[..., 0])
    # why: only the exact winning cell owns radius/offset supervision.  Using
    # the whole near-peak Gaussian made neighboring cells learn zero radii,
    # which caused valid heatmap peaks to decode as collapsed ellipses.
    positive = true_values[..., 5:6]
    geometry_error = tf.abs(true_values[..., 1:5] - pred_values[..., 1:5])
    geometry_loss = tf.reduce_sum(positive * geometry_error, axis=(1, 2, 3))
    geometry_loss /= tf.reduce_sum(positive, axis=(1, 2, 3)) + 1e-6
    quality_target = true_values[..., 5]
    quality_prediction = tf.clip_by_value(pred_values[..., 5], 1e-4, 1.0 - 1e-4)
    # why: the auxiliary quality map supplies a second spatial vote for the
    # face center and is balanced so the all-zero background cannot dominate.
    quality_bce = -(
        8.0 * quality_target * tf.math.log(quality_prediction)
        + (1.0 - quality_target) * tf.math.log(1.0 - quality_prediction)
    )
    quality_loss = tf.reduce_mean(quality_bce, axis=(1, 2))

    # A soft centroid discourages a high false peak far from the true face.
    coords = tf.cast(tf.range(GRID_SIZE), tf.float32) + 0.5
    grid_x = tf.reshape(coords / GRID_SIZE, (1, 1, GRID_SIZE))
    grid_y = tf.reshape(coords / GRID_SIZE, (1, GRID_SIZE, 1))
    soft_heat = tf.nn.relu(pred_values[..., 0] - 0.08) ** 2
    soft_heat /= tf.reduce_sum(soft_heat, axis=(1, 2), keepdims=True) + 1e-6
    predicted_center = tf.stack(
        [
            tf.reduce_sum(soft_heat * grid_x, axis=(1, 2)),
            tf.reduce_sum(soft_heat * grid_y, axis=(1, 2)),
        ],
        axis=1,
    )
    true_center = tf.stack(
        [
            tf.reduce_sum(true_values[..., 5] * (grid_x + true_values[..., 1] / GRID_SIZE), axis=(1, 2)),
            tf.reduce_sum(true_values[..., 5] * (grid_y + true_values[..., 2] / GRID_SIZE), axis=(1, 2)),
        ],
        axis=1,
    )
    center_alignment = tf.reduce_sum(tf.abs(predicted_center - true_center), axis=1)
    return (
        heat_loss
        + 10.0 * tf.reduce_mean(geometry_loss)
        + 0.5 * tf.reduce_mean(quality_loss)
        + 2.0 * tf.reduce_mean(center_alignment)
    )


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the QAT model as an integer-only TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield varied host-side samples for activation calibration."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run the exported model and return dequantized dense predictions."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    values: list[np.ndarray] = []
    for image in images:
        quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[None])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
        values.append((raw - output_zero) * output_scale)
    return np.asarray(values, dtype=np.float32).reshape(-1, GRID_SIZE, GRID_SIZE, CHANNELS)


def decode(predictions: np.ndarray) -> np.ndarray:
    """Decode the highest-centerness cell into normalized ellipse geometry."""
    decoded: list[list[float]] = []
    for values in predictions:
        heat = values[..., 0]
        y, x = np.unravel_index(np.argmax(heat), heat.shape)
        cell_x, cell_y = values[y, x, 1:3]
        rx, ry = values[y, x, 3:5]
        decoded.append([
            float(np.clip((x + cell_x) / GRID_SIZE, 0.0, 1.0)),
            float(np.clip((y + cell_y) / GRID_SIZE, 0.0, 1.0)),
            float(np.clip(rx, 1e-3, 1.0)),
            float(np.clip(ry, 1e-3, 1.0)),
        ])
    return np.asarray(decoded, dtype=np.float32)


def resize_images(images: np.ndarray) -> np.ndarray:
    """Resize arrays and keep the host-side training cache in float16.

    The QAT graph and TFLite representative samples still use float32 at the
    model boundary.  Keeping only the large host cache in float16 avoids a
    second 4-byte copy when TensorFlow builds the dataset, which is important
    for the full generic family under the 50 GB workstation RAM budget.
    """
    with tf.device("/CPU:0"):
        return tf.image.resize(images, (IMAGE_SIZE, IMAGE_SIZE)).numpy().astype(np.float16)


def main() -> None:
    """Train, QAT-finetune, export, parity-check, and score all test archives."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--qat-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generic-count", type=int, default=3000)
    parser.add_argument("--tiny-repeats", type=int, default=60)
    parser.add_argument("--board-repeats", type=int, default=3)
    args = parser.parse_args()

    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
    images = np.concatenate([
        generic_images[: args.generic_count],
        np.repeat(tiny_images, args.tiny_repeats, axis=0),
        np.repeat(board_images, args.board_repeats, axis=0),
    ])
    targets = np.concatenate([
        generic_targets[: args.generic_count],
        np.repeat(tiny_targets, args.tiny_repeats, axis=0),
        np.repeat(board_targets, args.board_repeats, axis=0),
    ])
    images, targets = make_scale_augmented_training_set(images, targets)
    images, targets = make_flip_augmented_training_set(images, targets)
    images = resize_images(images)
    contract_targets = make_targets(targets)
    dataset = (
        tf.data.Dataset.from_tensor_slices((images, contract_targets))
        # why: keep the large host cache float16, but feed the QAT graph the
        # float32 contract it was designed and parity-checked against.
        .map(lambda image, target: (tf.cast(image, tf.float32), target), num_parallel_calls=1)
        .shuffle(len(images), seed=SEED)
        .batch(args.batch_size)
        .prefetch(1)
    )
    print("training", images.shape, contract_targets.shape, flush=True)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=detector_loss)
    model.fit(dataset, epochs=args.epochs, verbose=2)

    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=detector_loss)
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)

    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")

    # Compare the contract before evaluating so export drift is visible in the report.
    float_values = np.asarray(qat.predict(images[:32], verbose=0), dtype=np.float32)
    int8_values = predict_int8(args.output / "model_int8.tflite", images[:32]).reshape(32, -1)
    parity_mae = float(np.mean(np.abs(float_values - int8_values)))
    report: dict[str, object] = {
        "input_size": IMAGE_SIZE,
        "grid_size": GRID_SIZE,
        "train_samples": int(len(images)),
        "keras_tflite_contract_mae": parity_mae,
        "tests": {},
    }
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name, image_size=IMAGE_SIZE)
        predictions = np.concatenate([
            decode(predict_int8(args.output / "model_int8.tflite", test_images)),
            np.ones((len(test_targets), 1), dtype=np.float32),
        ], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
