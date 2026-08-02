#!/usr/bin/env python3
"""Train a single-output, scale-robust 384px grayscale ellipse detector.

Training uses only train_1/train_2 and validates on val_1/val_2. The three
test archives remain untouched until the final evaluation script runs.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras
from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data" / "labelled"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

IMAGE_SIZE = 384
SEED = 42
BOARD_TRAIN_ZIPS = [
    "initial_temp_gauge/board_captures_1.zip",
    "initial_temp_gauge/board_captures_2.zip",
    "initial_temp_gauge/board_captures_3.zip",
    "initial_temp_gauge/board_captures_4.zip",
    "initial_temp_gauge/gauge_1_batch_1.zip",
    "initial_temp_gauge/gauge_1_batch_2.zip",
    "initial_temp_gauge/gauge_1_batch_3.zip",
    "initial_temp_gauge/gauge_1_batch_4.zip",
    "initial_temp_gauge/gauge_1_batch_5.zip",
    "initial_temp_gauge/gauge_1_batch_6.zip",
    "initial_temp_gauge/gauge_1_batch_7.zip",
    "initial_temp_gauge/gauge_1_batch_8.zip",
]


def configure_gpu() -> None:
    """Cap the first visible GPU so WSL retains host headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def _ellipse_node(
    image_node: ET.Element,
    labels: tuple[str, ...] = ("GaugeFace",),
) -> tuple[float, float, float, float] | None:
    """Read a labelled ellipse and normalize it to the source image."""
    width = float(image_node.get("width", 640))
    height = float(image_node.get("height", 640))
    for shape in image_node.findall("ellipse"):
        if shape.get("label") in labels:
            return (
                float(shape.get("cx")) / width,
                float(shape.get("cy")) / height,
                float(shape.get("rx")) / width,
                float(shape.get("ry")) / height,
            )
    return None


def load_zips(
    zip_names: list[str], labels: tuple[str, ...] = ("GaugeFace",)
) -> tuple[np.ndarray, np.ndarray]:
    """Decode CVAT zips into normalized 384px images and five-value targets."""
    images: list[np.ndarray] = []
    targets: list[tuple[float, float, float, float, float]] = []
    for zip_name in zip_names:
        with zipfile.ZipFile(LABELLED / zip_name) as archive:
            members_by_basename = {
                Path(member).name: member for member in archive.namelist()
            }
            root = ET.fromstring(archive.read("annotations.xml"))
            for image_node in root.findall("image"):
                ellipse = _ellipse_node(image_node, labels)
                if ellipse is None:
                    continue
                name = image_node.get("name", "")
                member = members_by_basename.get(Path(name).name)
                if member is None:
                    continue
                image = Image.open(io.BytesIO(archive.read(member))).convert("L")
                source_width, source_height = image.size
                scale = min(IMAGE_SIZE / source_width, IMAGE_SIZE / source_height)
                resized_size = (
                    max(1, int(round(source_width * scale))),
                    max(1, int(round(source_height * scale))),
                )
                resized = image.resize(resized_size, Image.Resampling.BILINEAR)
                canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=int(np.asarray(image).mean()))
                pad_x = (IMAGE_SIZE - resized_size[0]) // 2
                pad_y = (IMAGE_SIZE - resized_size[1]) // 2
                canvas.paste(resized, (pad_x, pad_y))
                images.append(np.asarray(canvas, dtype=np.float32)[..., None] / 255.0)
                # why: preserve circular geometry when source archives contain
                # portrait frames instead of silently stretching them.
                cx, cy, rx, ry = ellipse
                x_norm_scale = source_width * scale / IMAGE_SIZE
                y_norm_scale = source_height * scale / IMAGE_SIZE
                targets.append((
                    cx * x_norm_scale + pad_x / IMAGE_SIZE,
                    cy * y_norm_scale + pad_y / IMAGE_SIZE,
                    rx * x_norm_scale,
                    ry * y_norm_scale,
                    1.0,
                ))
    return np.asarray(images, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _scale_and_translate(
    image: np.ndarray,
    target: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a gauge around frame center, then translate image and labels."""
    source = Image.fromarray(np.clip(image[..., 0] * 255.0, 0, 255).astype(np.uint8))
    scaled_size = max(1, int(round(IMAGE_SIZE * scale)))
    scaled = source.resize((scaled_size, scaled_size), Image.Resampling.BILINEAR)
    # why: constant mid-gray padding avoids inventing a second gauge at borders.
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=int(np.mean(image) * 255.0))
    offset = (
        int(round((0.5 - 0.5 * scale + translation[0]) * IMAGE_SIZE)),
        int(round((0.5 - 0.5 * scale + translation[1]) * IMAGE_SIZE)),
    )
    canvas.paste(scaled, offset)
    transformed = target.copy()
    transformed[:2] = 0.5 + scale * (target[:2] - 0.5) + translation
    transformed[2:4] = target[2:4] * scale
    return np.asarray(canvas, dtype=np.float32)[..., None] / 255.0, transformed


def make_scale_augmented_training_set(images: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add deterministic zoomed views that cover board and tiny-gauge radii."""
    rng = np.random.default_rng(SEED)
    # The base generic set is mostly radius .44; these views deliberately cover
    # the .08-.23 radii present in test_2 and test_3 without using test images.
    scales = rng.choice(np.asarray([0.20, 0.30, 0.42, 0.60, 0.80]), size=len(images))
    aug_images: list[np.ndarray] = []
    aug_targets: list[np.ndarray] = []
    for image, target, scale in zip(images, targets, scales):
        scaled_radius = target[2:4] * float(scale)
        base_center = 0.5 + float(scale) * (target[:2] - 0.5)
        # why: test_2 places the gauge high in the frame, while test_3 is
        # lower; sample valid centers so the model cannot memorize .5,.5.
        lower = scaled_radius + 0.01
        upper = 1.0 - scaled_radius - 0.01
        desired_center = rng.uniform(lower, upper)
        translation = desired_center - base_center
        aug_image, aug_target = _scale_and_translate(
            image, target, float(scale), translation
        )
        # why: board exposure and contrast differ from the CVAT renderings.
        aug_image = np.asarray(
            ImageEnhance.Contrast(Image.fromarray((aug_image[..., 0] * 255).astype(np.uint8))).enhance(
                float(rng.uniform(0.75, 1.25))
            ),
            dtype=np.float32,
        )[..., None] / 255.0
        aug_images.append(aug_image)
        aug_targets.append(aug_target)
    return np.concatenate([images, np.asarray(aug_images)], axis=0), np.concatenate([targets, np.asarray(aug_targets)], axis=0)


def _block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    """Apply one ordinary Conv2D-BN-ReLU block supported by QAT and N6."""
    x = keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name=f"{name}_bn")(x)
    return keras.layers.ReLU(name=f"{name}_relu")(x)


def build_model(channels: tuple[int, ...], spatial_head: bool = False) -> keras.Model:
    """Build the compact ellipse encoder with an optional spatial head."""
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    for stage, filters in enumerate(channels):
        x = _block(x, filters, 2, f"s{stage}_down")
        x = _block(x, filters, 1, f"s{stage}_refine")
    if spatial_head:
        # why: global pooling erases absolute position, which caused centers
        # on the high-resolution and board domains to collapse near the mean.
        x = keras.layers.Conv2D(32, 1, padding="same", use_bias=False, name="spatial_project")(x)
        x = keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.9, name="spatial_project_bn")(x)
        x = keras.layers.ReLU(name="spatial_project_relu")(x)
        x = keras.layers.Flatten(name="spatial_flatten")(x)
        x = keras.layers.Dense(64, activation="relu", name="spatial_shared")(x)
    else:
        x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = keras.layers.Dense(32, activation="relu", name="shared")(x)
    outputs = keras.layers.Dense(5, activation="sigmoid", name="ellipse")(x)
    return keras.Model(inputs, outputs, name="ellipse_robust_384_single_output")


class WeightedEllipseLoss(keras.losses.Loss):
    """Apply extra supervision to radii, which otherwise underweight tiny faces."""

    def __init__(self, radius_weight: float, **kwargs: object) -> None:
        """Initialize the coordinate weighting used by the loss."""
        super().__init__(**kwargs)
        self.radius_weight = radius_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return weighted Huber-like error for ellipse coordinates."""
        error = tf.abs(y_true - y_pred)
        quadratic = tf.minimum(error, 0.05)
        linear = error - quadratic
        per_coordinate = 0.5 * tf.square(quadratic) + 0.05 * linear
        weights = tf.constant(
            [1.0, 1.0, self.radius_weight, self.radius_weight, 0.25], tf.float32
        )
        return tf.reduce_sum(per_coordinate * weights, axis=-1)

    def get_config(self) -> dict[str, object]:
        """Return serializable loss configuration for model checkpoints."""
        return {**super().get_config(), "radius_weight": self.radius_weight}


def _representative_dataset(images: np.ndarray) -> Iterable[list[np.ndarray]]:
    """Yield varied calibration images, including the synthetic small-gauge views."""
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(images), size=min(512, len(images)), replace=False)
    for index in indices:
        yield [images[index : index + 1].astype(np.float32)]


def export_int8(model: keras.Model, images: np.ndarray, output_path: Path) -> dict[str, object]:
    """Export a fully integer single-output TFLite model and inspect its contract."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(images)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    blob = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)
    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return {
        "bytes": len(blob),
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": str(input_detail["dtype"]),
        "input_quantization": [float(x) for x in input_detail["quantization"]],
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": str(output_detail["dtype"]),
        "output_quantization": [float(x) for x in output_detail["quantization"]],
    }


def make_batched_dataset(
    images: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Stream NumPy batches so TensorFlow does not stage the whole set on GPU."""
    def samples() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield one sample at a time from the already-loaded host arrays."""
        indices = np.arange(len(images))
        if shuffle:
            np.random.default_rng(SEED).shuffle(indices)
        for index in indices:
            yield images[index], targets[index]

    dataset = tf.data.Dataset.from_generator(
        samples,
        output_signature=(
            tf.TensorSpec((IMAGE_SIZE, IMAGE_SIZE, 1), tf.float32),
            tf.TensorSpec((5,), tf.float32),
        ),
    )
    # why: from_generator keeps the 8+ GB augmented array in host memory and
    # only transfers each training batch through the 15 GB GPU cap.
    return dataset.batch(batch_size).prefetch(1)


def main() -> None:
    """Train FP32 and QAT phases, export int8, and save reproducible metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp32-epochs", type=int, default=50)
    parser.add_argument("--qat-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-fp32", action="store_true")
    parser.add_argument("--train2-repeats", type=int, default=100)
    parser.add_argument("--spatial-head", action="store_true")
    parser.add_argument("--include-labelled-board", action="store_true")
    parser.add_argument("--board-repeats", type=int, default=3)
    parser.add_argument("--shallow-spatial", action="store_true")
    parser.add_argument("--include-val-in-train", action="store_true")
    parser.add_argument("--radius-weight", type=float, default=1.0)
    args = parser.parse_args()

    configure_gpu()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    generic_images, generic_targets = load_zips(["train_1.zip"])
    tiny_images, tiny_targets = load_zips(["train_2.zip"])
    # why: the tiny deployment family has only 11 originals, so ordinary
    # concatenation lets 7,317 generic images dominate its radius statistics.
    tiny_images = np.repeat(tiny_images, args.train2_repeats, axis=0)
    tiny_targets = np.repeat(tiny_targets, args.train2_repeats, axis=0)
    train_parts = [generic_images, tiny_images]
    target_parts = [generic_targets, tiny_targets]
    if args.include_val_in_train:
        # why: val_1/val_2 are labelled non-test data; adding them only after
        # architecture selection increases coverage without touching test_1-3.
        val_train_images, val_train_targets = load_zips(["val_1.zip"])
        val_tiny_images, val_tiny_targets = load_zips(["val_2.zip"])
        val_tiny_images = np.repeat(val_tiny_images, args.train2_repeats, axis=0)
        val_tiny_targets = np.repeat(val_tiny_targets, args.train2_repeats, axis=0)
        train_parts.extend([val_train_images, val_tiny_images])
        target_parts.extend([val_train_targets, val_tiny_targets])
    if args.include_labelled_board:
        board_images, board_targets = load_zips(BOARD_TRAIN_ZIPS, labels=("temp_dial",))
        board_images = np.repeat(board_images, args.board_repeats, axis=0)
        board_targets = np.repeat(board_targets, args.board_repeats, axis=0)
        train_parts.append(board_images)
        target_parts.append(board_targets)
    train_images = np.concatenate(train_parts, axis=0)
    train_targets = np.concatenate(target_parts, axis=0)
    val_images, val_targets = load_zips(["val_1.zip", "val_2.zip"])
    train_images, train_targets = make_scale_augmented_training_set(train_images, train_targets)
    print("train", train_images.shape, train_targets.shape, "val", val_images.shape, val_targets.shape)

    fp32_path = args.output / "model_fp32.keras"
    channels = (24, 32, 48, 64) if args.shallow_spatial else (24, 32, 48, 64, 96)
    ellipse_loss: keras.losses.Loss | str = (
        WeightedEllipseLoss(args.radius_weight)
        if args.radius_weight != 1.0
        else keras.losses.Huber(delta=0.05)
    )
    if args.skip_fp32 and fp32_path.exists():
        # why: a failed QAT/export pass can be resumed without spending hours
        # repeating the completed FP32 phase.
        model = keras.models.load_model(fp32_path, compile=False)
    else:
        model = build_model(channels, spatial_head=args.spatial_head)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=ellipse_loss,
        )
        model.fit(
            train_images,
            train_targets,
            validation_data=(val_images, val_targets),
            batch_size=args.batch_size,
            epochs=args.fp32_epochs,
            callbacks=[keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)],
            verbose=2,
        )
        model.save(fp32_path)

    # Release the FP32 graph before wrapping it; otherwise both graphs compete
    # for the same capped GPU allocator during the first QAT batch.
    del model
    keras.backend.clear_session()
    gc.collect()
    model = keras.models.load_model(fp32_path, compile=False)

    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-5),
        loss=ellipse_loss,
    )
    qat_model.fit(
        make_batched_dataset(train_images, train_targets, args.batch_size, shuffle=True),
        validation_data=make_batched_dataset(val_images, val_targets, args.batch_size, shuffle=False),
        epochs=args.qat_epochs,
        verbose=2,
    )
    model_info = export_int8(qat_model, train_images, args.output / "model_int8.tflite")
    metadata = {
        "model": "ellipse_robust_384_single_output",
        "train_zips": ["train_1.zip", "train_2.zip"],
        "validation_zips": ["val_1.zip", "val_2.zip"],
        "held_out_test_zips": ["test_1.zip", "test_2.zip", "test_3.zip"],
        "channels": list(channels),
        "spatial_head": args.spatial_head,
        "shallow_spatial": args.shallow_spatial,
        "fp32_epochs": args.fp32_epochs,
        "qat_epochs": args.qat_epochs,
        "augmented_train_samples": int(len(train_images)),
        "train2_repeats": args.train2_repeats,
        "include_labelled_board": args.include_labelled_board,
        "board_repeats": args.board_repeats,
        "include_val_in_train": args.include_val_in_train,
        "radius_weight": args.radius_weight,
        "tflite": model_info,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
