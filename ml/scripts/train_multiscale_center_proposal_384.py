#!/usr/bin/env python3
"""Train a multi-scale proposal network for gauge-center localization.

This experiment moves away from scalar ellipse regression and instead learns
two center-heatmap proposals at different scales plus a lightweight radius
head.  The intent is to let the model first propose where the gauge is, then
let a finer spatial head sharpen the center without forcing a single pooled
vector to explain the whole scene.
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

from embedded_gauge_reading_tinyml.heatmap_losses import (  # noqa: E402
    focal_heatmap_loss,
    softargmax_coordinate_loss,
)
from eval_ellipse_all_test_sets import _load_zip, _metrics  # noqa: E402
from train_ellipse_robust_384 import (  # noqa: E402
    BOARD_TRAIN_ZIPS,
    load_zips,
    make_scale_augmented_training_set,
)

IMAGE_SIZE = 384
FINE_SIZE = 96
COARSE_SIZE = 24
SEED = 42

# why: board_captures_2 is an exact image-basename duplicate of refreshed
# test_3 and must never contribute training pixels to a generalization
# experiment. The remaining BOARD_TRAIN_ZIPS entries carry no image bytes,
# so the effective board pool is board_captures_1 (201 images).
TRAIN_BOARD_ZIPS = [
    z for z in BOARD_TRAIN_ZIPS if z != "initial_temp_gauge/board_captures_2.zip"
]


def configure_gpu() -> None:
    """Limit TensorFlow to the project-approved 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def block(x: tf.Tensor, filters: int, stride: int, name: str) -> tf.Tensor:
    """Apply one quantization-friendly convolutional block."""
    layers = keras.layers
    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(epsilon=1e-3, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_model() -> keras.Model:
    """Build a proposal-first dual-scale heatmap model."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    # why: multi-scale features let the network keep both tiny-gauge detail
    # and enough global context to avoid confusing dials with clutter.
    for stage, filters in enumerate((24, 32, 48, 64, 96)):
        x = block(x, filters, 2, f"enc{stage}_down")
        x = block(x, filters, 1, f"enc{stage}_refine")
        skips.append(x)

    # why: the radius head is deliberately lightweight; center localization is
    # the primary task while radius only needs to be generally plausible.
    radius = layers.GlobalAveragePooling2D(name="radius_gap")(x)
    radius = layers.Dense(64, activation="relu", name="radius_shared")(radius)
    radius = layers.Dense(2, activation="sigmoid", name="radius")(radius)

    # Coarse proposal head: 24x24 heatmap from the deeper, lower-resolution
    # features.  This is the first-stage "where is the gauge?" signal.
    coarse = layers.Conv2D(1, 1, activation="sigmoid", name="coarse_heatmap")(skips[3])

    # Fine proposal head: decode back to 96x96 so the model can sharpen the
    # center after the coarse proposal has narrowed the search area.
    x = layers.UpSampling2D(2, interpolation="nearest", name="dec0_up")(skips[4])
    x = layers.Concatenate(name="dec0_join")([x, skips[3]])
    x = block(x, 64, 1, "dec0")
    x = layers.UpSampling2D(2, interpolation="nearest", name="dec1_up")(x)
    x = layers.Concatenate(name="dec1_join")([x, skips[2]])
    x = block(x, 48, 1, "dec1")
    x = layers.UpSampling2D(2, interpolation="nearest", name="dec2_up")(x)
    x = layers.Concatenate(name="dec2_join")([x, skips[1]])
    x = block(x, 32, 1, "dec2")
    x = block(x, 24, 1, "head_refine")
    fine = layers.Conv2D(1, 1, activation="sigmoid", name="fine_heatmap")(x)

    return keras.Model(inputs, [coarse, fine, radius], name="multiscale_center_proposal_384")


def make_heatmap_targets(targets: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """Rasterize center heatmap targets at the requested output resolution."""
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    cx = targets[:, 0, None, None]
    cy = targets[:, 1, None, None]
    heatmap = np.exp(-((xx[None] - cx) ** 2 + (yy[None] - cy) ** 2) / (2.0 * sigma**2))
    return heatmap.astype(np.float32)[..., None]


def make_targets(targets: np.ndarray) -> dict[str, np.ndarray]:
    """Build coarse/fine heatmap targets plus a radius regression target."""
    return {
        "coarse_heatmap": make_heatmap_targets(targets, COARSE_SIZE, sigma=0.040),
        "fine_heatmap": make_heatmap_targets(targets, FINE_SIZE, sigma=0.020),
        "radius": targets[:, 2:4].astype(np.float32),
    }


def _iter_samples(images: np.ndarray, targets: dict[str, np.ndarray]) -> object:
    """Return a generator over (image, (coarse, fine, radius)) host arrays.

    The closure keeps the big NumPy arrays in host RAM; TensorFlow's
    ``from_generator`` then pulls samples lazily so only each batch crosses
    the GPU memory cap.
    """

    def samples() -> object:
        """Yield one sample at a time from the loaded host arrays."""
        for index in range(len(images)):
            yield (
                images[index],
                (
                    targets["coarse_heatmap"][index],
                    targets["fine_heatmap"][index],
                    targets["radius"][index],
                ),
            )

    return samples


class MultiScaleLoss(keras.losses.Loss):
    """Combine heatmap proposal supervision with a small radius loss."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return a focal heatmap term plus a coordinate consistency term."""
        return focal_heatmap_loss(y_true, y_pred) + 0.25 * softargmax_coordinate_loss(y_true, y_pred)


def export_int8(model: keras.Model, images: np.ndarray, output: Path) -> None:
    """Export the QAT model as a fully integer TFLite graph."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield representative images for activation calibration."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images), min(512, len(images)), replace=False):
            yield [images[index : index + 1].astype(np.float32)]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the int8 model and return dequantized coarse/fine heatmaps plus radius."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()
    in_scale, in_zero = inputs["quantization"]
    out_q = [out["quantization"] for out in outputs]
    values: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for image in images:
        quantized = np.clip(np.round(image / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(inputs["index"], quantized[None])
        interpreter.invoke()
        coarse = interpreter.get_tensor(outputs[0]["index"])[0].astype(np.float32)
        fine = interpreter.get_tensor(outputs[1]["index"])[0].astype(np.float32)
        radius = interpreter.get_tensor(outputs[2]["index"])[0].astype(np.float32)
        coarse = (coarse - out_q[0][1]) * out_q[0][0]
        fine = (fine - out_q[1][1]) * out_q[1][0]
        radius = (radius - out_q[2][1]) * out_q[2][0]
        values.append((coarse, fine, radius))
    coarse_values = np.asarray([value[0] for value in values], dtype=np.float32)
    fine_values = np.asarray([value[1] for value in values], dtype=np.float32)
    radius_values = np.asarray([value[2] for value in values], dtype=np.float32)
    return coarse_values, fine_values, radius_values


def decode_heatmap(values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode a flattened heatmap to normalized center coordinates and confidence."""
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    centers: list[list[float]] = []
    confidences: list[float] = []
    for row in values:
        heatmap = row[..., 0]
        peak = float(np.max(heatmap))
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y0, y1 = max(0, y - 4), min(size, y + 5)
        x0, x1 = max(0, x - 4), min(size, x + 5)
        local = np.maximum(heatmap[y0:y1, x0:x1] - 0.03, 0.0) ** 2
        total = float(local.sum())
        if total > 1e-6:
            centers.append([
                float((local * xx[y0:y1, x0:x1]).sum() / total),
                float((local * yy[y0:y1, x0:x1]).sum() / total),
            ])
        else:
            centers.append([float((x + 0.5) / size), float((y + 0.5) / size)])
        confidences.append(peak)
    return np.asarray(centers, dtype=np.float32), np.asarray(confidences, dtype=np.float32)


def decode_predictions(coarse: np.ndarray, fine: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Combine coarse and fine proposals into a single ellipse prediction."""
    coarse_center, coarse_conf = decode_heatmap(coarse, COARSE_SIZE)
    fine_center, fine_conf = decode_heatmap(fine, FINE_SIZE)
    alpha = np.clip((fine_conf - 0.20) / 0.55, 0.0, 1.0)[:, None]
    center = (1.0 - alpha) * coarse_center + alpha * fine_center
    center = np.clip(center, 0.0, 1.0)
    return np.concatenate([center, np.clip(radius, 1e-3, 1.0)], axis=1).astype(np.float32)


def main() -> None:
    """Train, QAT-finetune, export, and score the multi-scale proposal model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--qat-epochs", type=int, default=5)
    parser.add_argument("--generic-limit", type=int, default=0, help="0 = use all generic train/val images")
    parser.add_argument("--tiny-repeats", type=int, default=100)
    parser.add_argument("--board-repeats", type=int, default=4)
    args = parser.parse_args()

    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(TRAIN_BOARD_ZIPS, labels=("temp_dial",))
    if args.generic_limit > 0:
        generic_images, generic_targets = generic_images[: args.generic_limit], generic_targets[: args.generic_limit]
    images = np.concatenate([
        generic_images,
        np.repeat(tiny_images, args.tiny_repeats, axis=0),
        np.repeat(board_images, args.board_repeats, axis=0),
    ])
    targets = np.concatenate([
        generic_targets,
        np.repeat(tiny_targets, args.tiny_repeats, axis=0),
        np.repeat(board_targets, args.board_repeats, axis=0),
    ])
    images, targets = make_scale_augmented_training_set(images, targets)
    train_targets = make_targets(targets)
    # why: from_generator keeps the ~16 GB augmented arrays in host memory and
    # only transfers each batch through the 15 GB GPU cap; from_tensor_slices
    # staged the whole set on the GPU and died with "Dst tensor is not
    # initialized" once the full data pool was enabled.
    dataset = (
        tf.data.Dataset.from_generator(
            _iter_samples(images, train_targets),
            output_signature=(
                tf.TensorSpec((IMAGE_SIZE, IMAGE_SIZE, 1), tf.float32),
                (
                    tf.TensorSpec((COARSE_SIZE, COARSE_SIZE, 1), tf.float32),
                    tf.TensorSpec((FINE_SIZE, FINE_SIZE, 1), tf.float32),
                    tf.TensorSpec((2,), tf.float32),
                ),
            ),
        )
        .shuffle(len(images), seed=SEED, reshuffle_each_iteration=True)
        .batch(8)
        .prefetch(1)
    )
    print("training", images.shape, flush=True)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=[MultiScaleLoss(), MultiScaleLoss(), keras.losses.Huber(delta=0.05)],
        loss_weights=[1.0, 1.5, 2.0],
    )
    model.fit(dataset, epochs=args.epochs, verbose=2)

    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(
        optimizer=keras.optimizers.Adam(2e-4),
        loss=[MultiScaleLoss(), MultiScaleLoss(), keras.losses.Huber(delta=0.05)],
        loss_weights=[1.0, 1.5, 2.0],
    )
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)

    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images, args.output / "model_int8.tflite")

    report: dict[str, object] = {
        "train_samples": int(len(images)),
        # why: board_captures_2 is an exact duplicate of refreshed test_3 and
        # must never appear in a generalization training set; record it so the
        # report is self-documenting (same convention as iteration 3).
        "excluded_test3_source": "initial_temp_gauge/board_captures_2.zip",
        "generic_limit": args.generic_limit,
        "tiny_repeats": args.tiny_repeats,
        "board_repeats": args.board_repeats,
        "tests": {},
    }
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        coarse, fine, radius = predict_int8(args.output / "model_int8.tflite", test_images)
        predictions = np.concatenate([decode_predictions(coarse, fine, radius), np.ones((len(test_images), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)

    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
