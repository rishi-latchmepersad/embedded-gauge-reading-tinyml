#!/usr/bin/env python3
"""Train a single scale-adaptive, QAT-compatible ellipse proposal model.

The network has three spatial resolutions but no domain-specific heads.  A
learned scale confidence chooses the appropriate heatmap/radius head for each
image, which handles large, tiny, and board gauges without memorizing dataset
identity.  The refreshed board holdout is deliberately excluded from training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras
from PIL import Image, ImageEnhance

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_robust_384 import (
    BOARD_TRAIN_ZIPS,
    IMAGE_SIZE,
    SEED,
    _block,
    load_zips,
)


SIZES = (24, 48, 96)
MAP_VALUES = tuple(size * size for size in SIZES)
TOTAL_MAP_VALUES = sum(2 * value for value in MAP_VALUES)
GEOMETRY_OFFSET = TOTAL_MAP_VALUES
CONFIDENCE_OFFSET = GEOMETRY_OFFSET + 12

# why: a full-size shuffle buffer on a 25,650-sample generator dataset held
# ~17 GB and, combined with float32 image storage, OOM-killed the WSL box.
# A 4096-sample buffer shuffles well enough while capping host RAM.
SHUFFLE_BUFFER = 4096

# why: 40 GB leaves headroom for TF runtime, OS, and the shell under the
# 50 GB policy while still aborting long before the kernel OOM killer.
MEMORY_BUDGET_MB = 40000

# why: board_captures_2 is an exact image-basename duplicate of refreshed
# test_3 and must never contribute training pixels to a generalization
# experiment. The other BOARD_TRAIN_ZIPS entries carry real image bytes now
# (repaired from ml/data/raw + clean_board_captures), so the board pool is
# board_captures_1/3/4 plus gauge_1_batch_1..8 = 598 images.
TRAIN_BOARD_ZIPS = [
    z for z in BOARD_TRAIN_ZIPS if z != "initial_temp_gauge/board_captures_2.zip"
]


def configure_gpu() -> None:
    """Cap TensorFlow's first GPU at 15 GB."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build shared encoder/decoder heads at three spatial resolutions."""
    layers = keras.layers
def build_model(channels: tuple[int, ...] = (16, 24, 32, 48, 64)) -> keras.Model:
    """Build the universal multiscale model with a configurable encoder.

    ``channels`` gives the per-stage filter counts; 6-stage variants are
    supported (the decoder warms up with one extra upsample so its heads
    still land on 24/48/96).
    """
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    for stage, filters in enumerate(channels):
        x = _block(x, filters, 2, f"enc{stage}_down")
        x = _block(x, filters, 1, f"enc{stage}_refine")
        skips.append(x)

    bottleneck_gap = layers.GlobalAveragePooling2D(name="scale_gap")(x)
    scale_confidence = layers.Dense(3, activation="softmax", name="scale_confidence")(bottleneck_gap)
    heads: list[tf.Tensor] = []
    geometry_heads: list[tf.Tensor] = []

    # why: for encoders deeper than 5 stages the bottleneck is below 12x12,
    # so warm up with extra upsampling before the 24/48/96 head loop.
    warmups = len(channels) - 5
    for warmup in range(warmups):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"warmup_up{warmup}")(x)
        x = layers.Concatenate(name=f"warmup_join{warmup}")([x, skips[4 - warmup]])
        x = _block(x, 48, 1, f"warmup_block{warmup}")

    # why: each head sees the same gauge but has a resolution suited to one
    # size band; selecting by learned confidence is not a gauge/domain route.
    for head_index, (filters, skip_index) in enumerate(((48, 3), (32, 2), (24, 1))):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{head_index}")(x)
        x = layers.Concatenate(name=f"join{head_index}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"dec{head_index}")
        center = layers.Conv2D(1, 1, activation="sigmoid", name=f"center_{SIZES[head_index]}")(x)
        rim = layers.Conv2D(1, 1, activation="sigmoid", name=f"rim_{SIZES[head_index]}")(x)
        heads.extend([layers.Flatten(name=f"center_flat_{SIZES[head_index]}")(center), layers.Flatten(name=f"rim_flat_{SIZES[head_index]}")(rim)])
        geometry = layers.GlobalAveragePooling2D(name=f"geometry_gap_{SIZES[head_index]}")(x)
        geometry = layers.Dense(24, activation="relu", name=f"geometry_shared_{SIZES[head_index]}")(geometry)
        geometry_heads.append(layers.Dense(4, activation="sigmoid", name=f"geometry_{SIZES[head_index]}")(geometry))

    output = layers.Concatenate(name="multiscale_contract")(
        [*heads, *geometry_heads, scale_confidence]
    )
    return keras.Model(inputs, output, name="ellipse_multiscale_universal_384")


def make_map_targets(geometry: np.ndarray) -> np.ndarray:
    """Create center/rim targets, scale-specific geometry, and scale labels."""
    all_maps: list[np.ndarray] = []
    for size in SIZES:
        coords = (np.arange(size, dtype=np.float32) + 0.5) / size
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        centers: list[np.ndarray] = []
        rims: list[np.ndarray] = []
        for cx, cy, rx, ry in geometry[:, :4]:
            sigma = 1.25 / size
            center = np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
            distance = ((xx - cx) / max(float(rx), 1e-3)) ** 2 + ((yy - cy) / max(float(ry), 1e-3)) ** 2
            rim = np.exp(-0.5 * ((distance - 1.0) / 0.16) ** 2)
            centers.append(center.astype(np.float32).reshape(-1))
            rims.append(rim.astype(np.float32).reshape(-1))
        all_maps.extend([np.asarray(centers), np.asarray(rims)])
    # why: the target scale is determined by radius, not archive identity.
    max_radius = np.max(geometry[:, 2:4], axis=1)
    scale_index = np.where(max_radius < 0.14, 2, np.where(max_radius < 0.30, 1, 0))
    one_hot = np.eye(3, dtype=np.float32)[scale_index]
    return np.concatenate([*all_maps, *[geometry[:, :4]] * 3, one_hot], axis=1).astype(np.float32)


class MultiScaleLoss(keras.losses.Loss):
    """Train all spatial heads while supervising scale confidence by radius."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return center, rim, geometry, and scale-selection losses."""
        total = tf.zeros(tf.shape(y_pred)[0], dtype=tf.float32)
        offset = 0
        for size in SIZES:
            values = size * size
            true_center = tf.reshape(y_true[:, offset : offset + values], (-1, size, size))
            pred_center = tf.reshape(y_pred[:, offset : offset + values], (-1, size, size))
            offset += values
            true_rim = tf.reshape(y_true[:, offset : offset + values], (-1, size, size))
            pred_rim = tf.reshape(y_pred[:, offset : offset + values], (-1, size, size))
            offset += values
            center_weight = 1.0 + 12.0 * true_center
            center_loss = tf.reduce_mean(center_weight * tf.square(true_center - pred_center), axis=(1, 2))
            rim_clip = tf.clip_by_value(pred_rim, 1e-5, 1.0 - 1e-5)
            rim_bce = -(true_rim * tf.math.log(rim_clip) + (1.0 - true_rim) * tf.math.log(1.0 - rim_clip))
            rim_loss = tf.reduce_mean((1.0 + 3.0 * true_rim) * rim_bce, axis=(1, 2))
            total += 12.0 * center_loss + 1.5 * rim_loss
        for _ in SIZES:
            true_geometry = y_true[:, offset : offset + 4]
            pred_geometry = y_pred[:, offset : offset + 4]
            total += 2.0 * tf.reduce_sum(tf.abs(true_geometry - pred_geometry), axis=-1)
            offset += 4
        total += 2.0 * keras.losses.categorical_crossentropy(y_true[:, offset:], y_pred[:, offset:])
        return total

    def get_config(self) -> dict[str, object]:
        """Return the serializable loss configuration."""
        return super().get_config()


def export_int8(model: keras.Model, images_u8: np.ndarray, output: Path) -> None:
    """Export a fully integer TFLite model.

    ``images_u8`` is the uint8 [0,255] training set; the representative
    samples are converted to float32 [0,1] one batch at a time so the export
    never materializes a full float32 copy of the dataset.
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative() -> object:
        """Yield varied frames for activation calibration."""
        rng = np.random.default_rng(SEED)
        for index in rng.choice(len(images_u8), min(512, len(images_u8)), replace=False):
            yield [images_u8[index : index + 1].astype(np.float32) / 255.0]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def predict_int8(model_path: Path, images: np.ndarray) -> np.ndarray:
    """Run the int8 model and return its dequantized contract vector."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
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
    return np.asarray(values, dtype=np.float32)


def decode_contract(contract: np.ndarray) -> np.ndarray:
    """Select one scale head and decode center, radii, and confidence."""
    confidence = contract[:, CONFIDENCE_OFFSET : CONFIDENCE_OFFSET + 3]
    selected = np.argmax(confidence, axis=1)
    predictions = np.zeros((len(contract), 5), dtype=np.float32)
    geometry_offset = GEOMETRY_OFFSET
    for row, head in enumerate(selected):
        size = SIZES[head]
        values = size * size
        center_start = sum(2 * v for v in MAP_VALUES[:head])
        center = contract[row, center_start : center_start + values].reshape(size, size)
        coords = (np.arange(size, dtype=np.float32) + 0.5) / size
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        weights = np.maximum(center - 0.05, 0.0) ** 4.0
        total = max(float(weights.sum()), 1e-6)
        predictions[row, :2] = [(weights * xx).sum() / total, (weights * yy).sum() / total]
        predictions[row, 2:4] = contract[row, geometry_offset + 4 * head : geometry_offset + 4 * head + 4][2:4]
        predictions[row, 4] = confidence[row, head]
    return predictions


def _iter_samples(images: np.ndarray, targets: np.ndarray) -> object:
    """Return a generator over (image, contract) host arrays.

    Images are stored as uint8 to keep the full set at ~3.8 GB instead of
    ~15 GB; each sample is converted to float32 on the fly, so only batches
    cross into GPU memory.  ``from_tensor_slices`` plus a full-size shuffle
    buffer OOM-killed the WSL box at 53 GB, so the buffer is capped instead.
    """

    def samples() -> object:
        """Yield one sample at a time from the loaded host arrays."""
        for index in range(len(images)):
            yield images[index].astype(np.float32) / 255.0, targets[index]

    return samples


def _augment_uint8(images: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale/translate/contrast/flip-augment uint8 images, keeping targets aligned.

    Mirrors ``make_scale_augmented_training_set`` from the robust trainer but
    works on uint8 frames so the doubled set costs ~3.8 GB instead of ~15 GB.
    A random horizontal flip is applied to half the augmented views: the
    test_2 IMG_144x family contains hflip variants that otherwise fail
    (~178px center error) because no flipped views ever reach training.
    """

    rng = np.random.default_rng(SEED)
    scales = rng.choice(np.asarray([0.20, 0.30, 0.42, 0.60, 0.80]), size=len(images))
    flips = rng.random(len(images)) < 0.5
    aug_images = np.empty_like(images)
    aug_targets = np.empty_like(targets)
    for index, (image, target, scale) in enumerate(zip(images, targets, scales)):
        scaled_radius = target[2:4] * float(scale)
        base_center = 0.5 + float(scale) * (target[:2] - 0.5)
        lower = scaled_radius + 0.01
        upper = 1.0 - scaled_radius - 0.01
        desired_center = rng.uniform(lower, upper)
        translation = desired_center - base_center
        source = Image.fromarray(image[..., 0])
        if flips[index]:
            # why: mirror the frame first so the flip is applied in source
            # space; the center x is mirrored in the same coordinate system.
            source = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        scaled_size = max(1, int(round(IMAGE_SIZE * float(scale))))
        scaled = source.resize((scaled_size, scaled_size), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=int(np.mean(image)))
        offset = (
            int(round((0.5 - 0.5 * scale + translation[0]) * IMAGE_SIZE)),
            int(round((0.5 - 0.5 * scale + translation[1]) * IMAGE_SIZE)),
        )
        canvas.paste(scaled, offset)
        canvas = ImageEnhance.Contrast(canvas).enhance(float(rng.uniform(0.75, 1.25)))
        aug_images[index] = np.asarray(canvas, dtype=np.uint8)[..., None]
        transformed = target.copy()
        transformed[0] = 1.0 - transformed[0] if flips[index] else transformed[0]
        transformed[:2] = 0.5 + scale * (transformed[:2] - 0.5) + translation
        transformed[2:4] = target[2:4] * scale
        aug_targets[index] = transformed
    return (
        np.concatenate([images, aug_images], axis=0),
        np.concatenate([targets, aug_targets], axis=0),
    )


def _memory_preflight(n_samples: int) -> None:
    """Abort with a readable error if the dataset footprint exceeds the RAM budget.

    why: the 2026-07-31 crash grew anon RSS to 53 GB and the kernel OOM
    killer took down the whole WSL instance.  Estimating before allocation
    lets the process fail cleanly instead of killing the box.
    """

    image_bytes = n_samples * IMAGE_SIZE * IMAGE_SIZE  # uint8
    contract_bytes = n_samples * (TOTAL_MAP_VALUES + 12 + 3) * 4  # float32
    shuffle_bytes = min(n_samples, SHUFFLE_BUFFER) * (
        IMAGE_SIZE * IMAGE_SIZE + (TOTAL_MAP_VALUES + 12 + 3) * 4
    )
    total_mb = (image_bytes + contract_bytes + shuffle_bytes) / 1e6
    print(f"memory preflight: {total_mb / 1024:.1f} GiB estimated for {n_samples} samples", flush=True)
    if total_mb > MEMORY_BUDGET_MB:
        raise SystemExit(
            f"aborting: estimated {total_mb / 1024:.1f} GiB exceeds {MEMORY_BUDGET_MB / 1024:.0f} GiB budget"
        )


def main() -> None:
    """Train, quantize, export, and evaluate the clean-split model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--qat-epochs", type=int, default=10)
    parser.add_argument("--tiny-repeats", type=int, default=100)
    parser.add_argument("--board-repeats", type=int, default=4)
    parser.add_argument("--channels", type=str, default="16,24,32,48,64",
                        help="Comma-separated encoder filter counts per stage.")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training; export int8 from saved QAT weights and evaluate.")
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    # Use every non-test labelled archive. The held-out test zips remain
    # untouched so the per-domain gate stays a valid acceptance test.
    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(TRAIN_BOARD_ZIPS, labels=("temp_dial",))
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    # why: store uint8 so the 25,650-image set is ~3.8 GB, not ~15 GB; the
    # generator converts each sample to float32 on the fly.
    images_u8 = np.clip(np.round(images * 255.0), 0, 255).astype(np.uint8)
    del images
    images_u8, targets = _augment_uint8(images_u8, targets)
    contract_targets = make_map_targets(targets)
    _memory_preflight(len(images_u8))
    # why: from_generator keeps the uint8 image array and contract targets in
    # host memory; only each batch crosses the 15 GB GPU cap, and the shuffle
    # buffer is capped so it cannot re-materialize the whole dataset.
    dataset = (
        tf.data.Dataset.from_generator(
            _iter_samples(images_u8, contract_targets),
            output_signature=(
                tf.TensorSpec((IMAGE_SIZE, IMAGE_SIZE, 1), tf.float32),
                tf.TensorSpec((TOTAL_MAP_VALUES + 12 + 3,), tf.float32),
            ),
        )
        .shuffle(min(SHUFFLE_BUFFER, len(images_u8)), seed=SEED, reshuffle_each_iteration=True)
        .batch(32)
        .prefetch(1)
    )
    print("training", images_u8.shape, contract_targets.shape, "board_source=all_except_test3_duplicate", flush=True)

    channels = tuple(int(f) for f in args.channels.split(","))
    if args.export_only:
        # why: the training/export pipeline may die after QAT (memory guard);
        # this path resumes from the saved QAT weights without retraining.
        qat = tfmot.quantization.keras.quantize_model(build_model(channels))
        qat.load_weights(args.output / "model_qat.weights.h5")
        qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=MultiScaleLoss())
        args.output.mkdir(parents=True, exist_ok=True)
        export_int8(qat, images_u8, args.output / "model_int8.tflite")
        report: dict[str, object] = {
            "train_samples": int(len(images_u8)),
            "excluded_test3_source": "initial_temp_gauge/board_captures_2.zip",
            "tiny_repeats": args.tiny_repeats,
            "board_repeats": args.board_repeats,
            "channels": list(channels),
            "export_only_resume": True,
            "tests": {},
        }
        for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
            test_images, test_targets = _load_zip(zip_name)
            with tf.device("/CPU:0"):
                test_images = tf.image.resize(test_images, [IMAGE_SIZE, IMAGE_SIZE]).numpy()
            predictions = decode_contract(predict_int8(args.output / "model_int8.tflite", test_images))
            report["tests"][zip_name] = _metrics(predictions, test_targets)
            print(zip_name, report["tests"][zip_name], flush=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return
    model = build_model(channels); model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=MultiScaleLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=MultiScaleLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    # why: pass the uint8 set directly; export_int8 converts each
    # representative sample on the fly. A float32 copy here is ~17.7 GB
    # and tripped the memory guard at export time.
    export_int8(qat, images_u8, args.output / "model_int8.tflite")

    report: dict[str, object] = {
        "train_samples": int(len(images_u8)),
        # why: board_captures_2 is an exact duplicate of refreshed test_3 and
        # must never appear in a generalization training set; record it so the
        # report is self-documenting (same convention as iteration 3).
        "excluded_test3_source": "initial_temp_gauge/board_captures_2.zip",
        "tiny_repeats": args.tiny_repeats,
        "board_repeats": args.board_repeats,
        "channels": list(channels),
        "tests": {},
    }
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        with tf.device("/CPU:0"):
            test_images = tf.image.resize(test_images, [IMAGE_SIZE, IMAGE_SIZE]).numpy()
        predictions = decode_contract(predict_int8(args.output / "model_int8.tflite", test_images))
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
