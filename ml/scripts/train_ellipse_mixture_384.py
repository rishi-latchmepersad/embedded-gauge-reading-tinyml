#!/usr/bin/env python3
"""Train a QAT-safe mixture-of-domains 384px ellipse detector.

The shared encoder learns common gauge features, while three small heads retain
the different geometry priors of generic, tiny high-resolution, and board data.
An int8 domain head selects one ellipse head at inference time.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from train_ellipse_robust_384 import (
    BOARD_TRAIN_ZIPS,
    IMAGE_SIZE,
    SEED,
    _block,
    build_model,
    load_zips,
    make_scale_augmented_training_set,
)


def configure_gpu() -> None:
    """Cap the first visible GPU at 15 GB for host headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


class MixtureEllipseLoss(keras.losses.Loss):
    """Train only the active ellipse head and always train the domain selector."""

    def __init__(self, domain_weight: float = 0.25, **kwargs: object) -> None:
        """Initialize the domain classification contribution."""
        super().__init__(**kwargs)
        self.domain_weight = domain_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return masked ellipse Huber error plus domain cross entropy."""
        true_ellipses = tf.reshape(y_true[:, :15], (-1, 3, 5))
        pred_ellipses = tf.reshape(y_pred[:, :15], (-1, 3, 5))
        domain_true = y_true[:, 15:]
        domain_pred = y_pred[:, 15:]
        error = tf.abs(true_ellipses - pred_ellipses)
        quadratic = tf.minimum(error, 0.05)
        linear = error - quadratic
        huber = 0.5 * tf.square(quadratic) + 0.05 * linear
        # why: inactive heads must not pull toward the wrong domain geometry.
        active = domain_true[..., None]
        ellipse_loss = tf.reduce_sum(huber * active) / tf.maximum(tf.reduce_sum(active), 1.0)
        domain_loss = tf.keras.losses.categorical_crossentropy(domain_true, domain_pred)
        return ellipse_loss + self.domain_weight * domain_loss

    def get_config(self) -> dict[str, object]:
        """Return a serializable loss configuration."""
        return {**super().get_config(), "domain_weight": self.domain_weight}


def build_mixture_model() -> keras.Model:
    """Build a compact shared encoder with three ellipse heads and a selector."""
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    for stage, filters in enumerate((24, 32, 48, 64, 96)):
        x = _block(x, filters, 2, f"s{stage}_down")
        x = _block(x, filters, 1, f"s{stage}_refine")
    # why: preserve enough absolute layout for tiny and board placements.
    x = keras.layers.Conv2D(32, 1, padding="same", use_bias=False, name="spatial_project")(x)
    x = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name="spatial_project_bn")(x)
    x = keras.layers.ReLU(name="spatial_project_relu")(x)
    x = keras.layers.Flatten(name="spatial_flatten")(x)
    shared = keras.layers.Dense(64, activation="relu", name="spatial_shared")(x)
    ellipses = keras.layers.Dense(15, activation="sigmoid", name="ellipse_heads")(shared)
    domain = keras.layers.Dense(3, activation="softmax", name="domain_selector")(shared)
    outputs = keras.layers.Concatenate(name="mixture_output")([ellipses, domain])
    return keras.Model(inputs, outputs, name="ellipse_mixture_384")


def make_domain_dataset(
    images: np.ndarray,
    targets: np.ndarray,
    domains: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Stream five-value ellipse targets and one-hot domain labels in batches."""
    packed = np.concatenate(
        [np.tile(targets[:, None, :], (1, 3, 1)).reshape(len(targets), 15), domains],
        axis=1,
    ).astype(np.float32)

    def samples() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield samples from host memory without staging the complete set on GPU."""
        indices = np.arange(len(images))
        if shuffle:
            np.random.default_rng(SEED).shuffle(indices)
        for index in indices:
            yield images[index], packed[index]

    dataset = tf.data.Dataset.from_generator(
        samples,
        output_signature=(
            tf.TensorSpec((IMAGE_SIZE, IMAGE_SIZE, 1), tf.float32),
            tf.TensorSpec((18,), tf.float32),
        ),
    )
    return dataset.batch(batch_size).prefetch(1)


def representative_dataset(images: np.ndarray) -> Iterable[list[np.ndarray]]:
    """Yield varied augmented images for full-integer calibration."""
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(images), size=min(512, len(images)), replace=False)
    for index in indices:
        yield [images[index : index + 1].astype(np.float32)]


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> dict[str, object]:
    """Export and inspect the fully integer mixture contract."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(images)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return {
        "bytes": len(blob),
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": str(input_detail["dtype"]),
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": str(output_detail["dtype"]),
        "input_quantization": [float(x) for x in input_detail["quantization"]],
        "output_quantization": [float(x) for x in output_detail["quantization"]],
    }


def load_domain_pool(repeats: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and augment all non-test domains with balanced domain labels."""
    sources = [
        (["train_1.zip", "val_1.zip"], ("GaugeFace",), 0, 1),
        (["train_2.zip", "val_2.zip"], ("GaugeFace",), 1, repeats),
        (BOARD_TRAIN_ZIPS, ("temp_dial",), 2, 3),
    ]
    image_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    domain_parts: list[np.ndarray] = []
    for zips, labels, domain, repeat in sources:
        images, targets = load_zips(zips, labels=labels)
        images = np.repeat(images, repeat, axis=0)
        targets = np.repeat(targets, repeat, axis=0)
        images, targets = make_scale_augmented_training_set(images, targets)
        image_parts.append(images)
        target_parts.append(targets)
        one_hot = np.zeros((len(targets), 3), dtype=np.float32)
        one_hot[:, domain] = 1.0
        domain_parts.append(one_hot)
    return (
        np.concatenate(image_parts),
        np.concatenate(target_parts),
        np.concatenate(domain_parts),
    )


def main() -> None:
    """Train, QAT-finetune, export, and record the mixture model contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp32-epochs", type=int, default=50)
    parser.add_argument("--qat-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tiny-repeats", type=int, default=100)
    args = parser.parse_args()

    configure_gpu()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    images, targets, domains = load_domain_pool(args.tiny_repeats)
    print("train", images.shape, targets.shape, domains.shape)

    model = build_mixture_model()
    loss = MixtureEllipseLoss()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=loss,
    )
    model.fit(
        make_domain_dataset(images, targets, domains, args.batch_size, shuffle=True),
        epochs=args.fp32_epochs,
        verbose=2,
    )
    fp32_path = args.output / "model_fp32.keras"
    model.save(fp32_path)
    del model
    keras.backend.clear_session()
    gc.collect()
    model = keras.models.load_model(fp32_path, custom_objects={"MixtureEllipseLoss": MixtureEllipseLoss}, compile=False)
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-5),
        loss=loss,
    )
    qat_model.fit(
        make_domain_dataset(images, targets, domains, args.batch_size, shuffle=True),
        epochs=args.qat_epochs,
        verbose=2,
    )
    model_info = export_int8(qat_model, images, args.output / "model_int8.tflite")
    metadata = {
        "model": "ellipse_mixture_384",
        "non_test_sources": ["train_1.zip", "val_1.zip", "train_2.zip", "val_2.zip", *BOARD_TRAIN_ZIPS],
        "held_out_test_zips": ["test_1.zip", "test_2.zip", "test_3.zip"],
        "domain_order": ["generic", "tiny", "board"],
        "train_samples": int(len(images)),
        "fp32_epochs": args.fp32_epochs,
        "qat_epochs": args.qat_epochs,
        "tflite": model_info,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
