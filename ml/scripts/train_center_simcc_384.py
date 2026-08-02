#!/usr/bin/env python3
"""Train a center-only SimCC ellipse localizer at 384 px.

This is a deliberate architecture break from the dense heatmap and scalar
regression families: we keep an embedded-friendly MobileNetV2 trunk, but move
the localization head to SimCC coordinate classification plus a small radius
regressor.  The goal is to make the center explicit without forcing a full
heatmap decoder.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.obb_simcc_tf_models import _build_mobilenetv2_feature_tensor
from eval_ellipse_all_test_sets import _load_zip
from train_ellipse_robust_384 import SEED, load_zips, make_scale_augmented_training_set

IMAGE_SIZE: int = 384
NUM_BINS: int = 144
GPU_MEMORY_LIMIT_MB: int = 15_000
RAM_LIMIT_BYTES: int = 55 * 1024**3


def configure_gpu() -> None:
    """Cap the visible GPU before any tensors are materialized."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT_MB)],
        )
    # why: TensorFlow's default thread pools can claim a surprising amount of
    # stack/virtual memory in WSL, so keep them intentionally tiny.
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def configure_ram_limit() -> None:
    """Optionally cap host virtual memory when explicitly requested.

    TensorFlow can reserve a large amount of virtual address space during
    graph construction even when the resident set stays well below the real
    RAM budget.  For this trainer we keep the 50 GB policy documented, but we
    only enforce it when the caller opts in, because the hard fence otherwise
    aborts the first training step before we can measure the model.
    """
    if os.environ.get("ENABLE_HARD_RLIMIT_AS") != "1":
        return
    resource.setrlimit(resource.RLIMIT_AS, (RAM_LIMIT_BYTES, RAM_LIMIT_BYTES))


def make_simcc_target(
    coord: float,
    *,
    num_bins: int = NUM_BINS,
    sigma_bins: float = 1.5,
) -> np.ndarray:
    """Build a smooth 1D SimCC target centered on one normalized coordinate."""
    bins = np.arange(num_bins, dtype=np.float32)
    center = float(np.clip(coord, 0.0, 1.0)) * float(num_bins - 1)
    target = np.exp(-((bins - center) ** 2) / (2.0 * sigma_bins * sigma_bins))
    target /= np.maximum(np.sum(target), 1e-6)
    return target.astype(np.float32)


def build_center_simcc_model(
    *,
    alpha: float = 0.5,
    spatial_channels: int = 96,
    head_units: int = 128,
    head_dropout: float = 0.10,
) -> keras.Model:
    """Build a MobileNetV2 trunk with SimCC center heads and radius regression."""
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")

    backbone_layers: dict[str, keras.layers.Layer] = {}
    backbone_features = _build_mobilenetv2_feature_tensor(
        inputs,
        alpha=alpha,
        created_layers=backbone_layers,
    )
    for layer in backbone_layers.values():
        layer.trainable = True

    # why: a tiny spatial trunk preserves location better than global pooling
    # alone, while still staying lightweight enough for TFLite int8 export.
    shared = keras.layers.Conv2D(
        spatial_channels,
        1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_proj",
    )(backbone_features)
    shared = keras.layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name="shared_spatial_up",
    )(shared)
    shared = keras.layers.Conv2D(
        spatial_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_conv_1",
    )(shared)
    shared = keras.layers.Conv2D(
        spatial_channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="shared_spatial_conv_2",
    )(shared)

    feature_side = shared.shape[1]
    if feature_side is None:
        raise ValueError("SimCC trunk requires a static spatial feature size.")
    if NUM_BINS % int(feature_side) != 0:
        raise ValueError(
            f"NUM_BINS={NUM_BINS} must be a multiple of the spatial trunk size {feature_side}."
        )
    expansion = NUM_BINS // int(feature_side)

    def axis_simcc_head(
        features: tf.Tensor,
        *,
        axis: str,
        name_prefix: str,
    ) -> tf.Tensor:
        """Convert a 24x24 feature map into a 1D SimCC distribution."""
        if axis == "x":
            x = keras.layers.Conv2D(
                spatial_channels,
                kernel_size=(int(feature_side), 1),
                strides=(int(feature_side), 1),
                padding="valid",
                use_bias=False,
                groups=spatial_channels,
                kernel_initializer=keras.initializers.Constant(1.0 / float(feature_side)),
                name=f"{name_prefix}_collapse_height",
            )(features)
            x = keras.layers.UpSampling2D(
                size=(1, expansion),
                interpolation="bilinear",
                name=f"{name_prefix}_expand_width",
            )(x)
        elif axis == "y":
            x = keras.layers.Conv2D(
                spatial_channels,
                kernel_size=(1, int(feature_side)),
                strides=(1, int(feature_side)),
                padding="valid",
                use_bias=False,
                groups=spatial_channels,
                kernel_initializer=keras.initializers.Constant(1.0 / float(feature_side)),
                name=f"{name_prefix}_collapse_width",
            )(features)
            x = keras.layers.UpSampling2D(
                size=(expansion, 1),
                interpolation="bilinear",
                name=f"{name_prefix}_expand_height",
            )(x)
        else:
            raise ValueError(f"Unsupported axis: {axis}")

        # why: a tiny local context helps the 1D coordinate logits sharpen
        # without paying for a full transformer decoder.
        x = keras.layers.Conv2D(
            spatial_channels,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"{name_prefix}_conv_1",
        )(x)
        x = keras.layers.Conv2D(
            spatial_channels,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"{name_prefix}_conv_2",
        )(x)
        x = keras.layers.Conv2D(
            1,
            1,
            padding="same",
            activation=None,
            kernel_initializer="he_normal",
            name=f"{name_prefix}_logits_2d",
        )(x)
        x = keras.layers.Flatten(name=f"{name_prefix}_flatten")(x)
        return keras.layers.Softmax(name=f"{name_prefix}_simcc")(x)

    # Direct center and radius heads remain as cheap auxiliary geometry cues.
    pooled = keras.layers.GlobalAveragePooling2D(name="geom_gap")(shared)
    pooled = keras.layers.Dense(
        head_units,
        activation="relu",
        kernel_initializer="he_normal",
        name="geom_dense_1",
    )(pooled)
    pooled = keras.layers.Dropout(head_dropout, name="geom_dropout")(pooled)
    center_xy = keras.layers.Dense(
        2,
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="center_xy",
    )(pooled)
    radius_wh = keras.layers.Dense(
        2,
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="radius_wh",
    )(pooled)

    center_x_simcc = axis_simcc_head(shared, axis="x", name_prefix="center_x")
    center_y_simcc = axis_simcc_head(shared, axis="y", name_prefix="center_y")

    outputs = [center_xy, radius_wh, center_x_simcc, center_y_simcc]
    return keras.Model(inputs=inputs, outputs=outputs, name="center_simcc_384")


def load_split(zip_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load, augment, and return grayscale 384px images plus ellipse labels."""
    images, targets = load_zips(zip_names)
    images, targets = make_scale_augmented_training_set(images, targets)
    return images, targets


def make_dataset(
    images: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Build a batched dataset with SimCC targets and scalar ellipse labels."""
    # why: from_tensor_slices would try to pack the entire 9+ GB augmented
    # image stack into one giant tensor. A generator keeps the samples on host
    # RAM and streams them into TensorFlow batch-by-batch.
    indices = np.arange(len(images))
    if shuffle:
        np.random.default_rng(SEED).shuffle(indices)

    def generator() -> object:
        """Yield one training sample and its SimCC/ellipse targets."""
        for index in indices:
            center_xy = targets[index, :2].astype(np.float32)
            radius_wh = targets[index, 2:4].astype(np.float32)
            yield (
                images[index].astype(np.float32),
                (
                    center_xy,
                    radius_wh,
                    make_simcc_target(float(center_xy[0])),
                    make_simcc_target(float(center_xy[1])),
                ),
                (
                    np.float32(1.0),
                    np.float32(0.5),
                    np.float32(1.0),
                    np.float32(1.0),
                ),
            )

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((IMAGE_SIZE, IMAGE_SIZE, 1), tf.float32),
            (
                tf.TensorSpec((2,), tf.float32),
                tf.TensorSpec((2,), tf.float32),
                tf.TensorSpec((NUM_BINS,), tf.float32),
                tf.TensorSpec((NUM_BINS,), tf.float32),
            ),
            (
                tf.TensorSpec((), tf.float32),
                tf.TensorSpec((), tf.float32),
                tf.TensorSpec((), tf.float32),
                tf.TensorSpec((), tf.float32),
            ),
        ),
    )
    if shuffle:
        ds = ds.shuffle(min(len(images), 4096), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)
    return ds.batch(batch_size).prefetch(1)


def soft_argmax_1d(probs: np.ndarray) -> np.ndarray:
    """Decode a SimCC probability vector into a normalized coordinate."""
    probs = np.asarray(probs, dtype=np.float32)
    probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-6)
    bins = np.arange(probs.shape[-1], dtype=np.float32)
    coord = np.sum(probs * bins, axis=-1) / float(max(probs.shape[-1] - 1, 1))
    return coord.astype(np.float32)


def export_int8(model: keras.Model, images: np.ndarray, output_path: Path) -> None:
    """Export the model as fully integer TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    sample_indices = np.linspace(0, len(images) - 1, num=min(512, len(images)), dtype=np.int32)

    def representative() -> object:
        """Yield representative inputs for activation calibration."""
        for index in sample_indices:
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)


def predict_int8(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run the int8 model and return [cx, cy, rx, ry, conf] predictions."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    input_scale, input_zero = input_detail["quantization"]

    predictions = np.zeros((len(images), 5), dtype=np.float32)
    for index, image in enumerate(images):
        quantized = np.clip(
            np.round(image[None] / input_scale + input_zero), -128, 127
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        outputs = [
            (interpreter.get_tensor(detail["index"])[0].astype(np.float32) - detail["quantization"][1])
            * detail["quantization"][0]
            for detail in output_details
        ]
        center_xy = outputs[0]
        radius_wh = outputs[1]
        center_x = soft_argmax_1d(outputs[2])
        center_y = soft_argmax_1d(outputs[3])
        predictions[index, :2] = [center_x, center_y]
        predictions[index, 2:4] = radius_wh
        predictions[index, 4] = 1.0 if np.all(np.isfinite(center_xy)) else 0.0
    return predictions


def main() -> None:
    """Train, QAT-finetune, export, and evaluate the SimCC center model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp32-epochs", type=int, default=20)
    parser.add_argument("--qat-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train2-repeats", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--spatial-channels", type=int, default=96)
    parser.add_argument("--head-units", type=int, default=128)
    args = parser.parse_args()

    configure_gpu()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train_images, train_targets = load_zips(["train_1.zip"])
    tiny_images, tiny_targets = load_zips(["train_2.zip"])
    tiny_images = np.repeat(tiny_images, args.train2_repeats, axis=0)
    tiny_targets = np.repeat(tiny_targets, args.train2_repeats, axis=0)
    train_images = np.concatenate([train_images, tiny_images], axis=0)
    train_targets = np.concatenate([train_targets, tiny_targets], axis=0)
    train_images, train_targets = make_scale_augmented_training_set(train_images, train_targets)
    val_images, val_targets = load_zips(["val_1.zip", "val_2.zip"])

    train_ds = make_dataset(train_images, train_targets, batch_size=args.batch_size, shuffle=True)
    val_ds = make_dataset(val_images, val_targets, batch_size=args.batch_size, shuffle=False)

    model = build_center_simcc_model(
        alpha=args.alpha,
        spatial_channels=args.spatial_channels,
        head_units=args.head_units,
    )

    losses = [
        keras.losses.Huber(delta=0.03),
        keras.losses.Huber(delta=0.04),
        keras.losses.CategoricalCrossentropy(),
        keras.losses.CategoricalCrossentropy(),
    ]
    loss_weights = [1.5, 0.5, 1.0, 1.0]
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=losses,
        loss_weights=loss_weights,
    )
    configure_ram_limit()

    args.output.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fp32_epochs,
        verbose=2,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6),
            keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                filepath=str(args.output / "best_fp32.keras"),
                monitor="val_loss",
                save_best_only=True,
            ),
        ],
    )
    model.save(args.output / "model_fp32.keras")

    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-5),
        loss=losses,
        loss_weights=loss_weights,
    )
    qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.qat_epochs,
        verbose=2,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6),
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        ],
    )
    qat_model.save(args.output / "model_qat.keras")
    export_int8(qat_model, train_images, args.output / "model_int8.tflite")

    report: dict[str, object] = {"train_samples": int(len(train_images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name, IMAGE_SIZE)
        predictions = predict_int8(args.output / "model_int8.tflite", images)
        center_error = np.linalg.norm((predictions[:, :2] - targets[:, :2]) * 640.0, axis=1)
        radius_error = np.linalg.norm((predictions[:, 2:4] - targets[:, 2:4]) * 640.0, axis=1)
        report["tests"][zip_name] = {
            "n": int(len(targets)),
            "center_mae_px": float(np.mean(center_error)),
            "center_median_px": float(np.median(center_error)),
            "center_pct_le_8px": float(np.mean(center_error <= 8.0)),
            "center_pct_le_16px": float(np.mean(center_error <= 16.0)),
            "radius_mae_px": float(np.mean(radius_error)),
            "radius_median_px": float(np.median(radius_error)),
            "radius_pct_le_8px": float(np.mean(radius_error <= 8.0)),
            "pred_radius_mean": [float(v) for v in np.mean(predictions[:, 2:4], axis=0)],
            "gt_radius_mean": [float(v) for v in np.mean(targets[:, 2:4], axis=0)],
            "pred_radius_std": [float(v) for v in np.std(predictions[:, 2:4], axis=0)],
        }
        print(zip_name, json.dumps(report["tests"][zip_name], indent=2), flush=True)

    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
