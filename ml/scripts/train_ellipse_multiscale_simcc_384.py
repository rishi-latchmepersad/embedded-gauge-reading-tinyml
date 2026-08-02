#!/usr/bin/env python3
"""Train a multiscale universal ellipse model with a SimCC fine-center head.

Architecture (iteration 5, literature-driven change over iter4):
- Same encoder/decoder family as ``train_ellipse_multiscale_universal_384``
  (5-stage encoder, 3-scale rim heatmaps, per-scale geometry, scale
  confidence), but the center keypoint is predicted by a SimCC coordinate
  classification head (Li et al., arXiv:2107.03332) instead of a coarse
  heatmap.

Why this is a big change worth testing:
- Tiny gauges (test_2, radius ~0.09) route to the 24x24 head in the pure
  heatmap design: 16 px cells mean ~4-5 px center error before decoding.
- SimCC classifies x and y into 1152 sub-pixel bins (3 bins per pixel), so
  center accuracy is no longer limited by heatmap stride, and softmax logits
  quantize cleanly for the INT8 N6 export.
- The rim ring heatmaps are kept at 3 scales: they define the ellipse radius
  band, which is where multiscale routing still helps.

Contract (flattened single output):
- 3 rim heatmaps (24x24, 48x48, 96x96) = 12,096 values
- 3 geometry heads (cx, cy, rx, ry) = 12 values
- scale confidence (3) = 3 values
- SimCC x logits (1152) + SimCC y logits (1152) = 2,304 values
- total = 14,415 values

All memory safeguards from AGENTS.md apply: uint8 image storage, capped
shuffle buffer, per-sample float conversion, preflight estimate, and the
run_wsl_guarded.sh launch wrapper.
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
RIM_VALUES = tuple(size * size for size in SIZES)
TOTAL_RIM_VALUES = sum(RIM_VALUES)
GEOMETRY_OFFSET = TOTAL_RIM_VALUES
CONFIDENCE_OFFSET = GEOMETRY_OFFSET + 12
SIMCC_OFFSET = CONFIDENCE_OFFSET + 3
SIMCC_BINS = 1152  # 3 sub-pixel bins per 384px
CONTRACT_SIZE = TOTAL_RIM_VALUES + 12 + 3 + 2 * SIMCC_BINS

SHUFFLE_BUFFER = 4096
MEMORY_BUDGET_MB = 40000

# why: board_captures_2 is an exact image-basename duplicate of refreshed
# test_3 and must never contribute training pixels to a generalization
# experiment (same rule as iterations 3-4).
TRAIN_BOARD_ZIPS = [
    z for z in BOARD_TRAIN_ZIPS if z != "initial_temp_gauge/board_captures_2.zip"
]


def configure_gpu() -> None:
    """Cap TensorFlow's first GPU at 15 GB so WSL retains headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build the SimCC-fine-center multiscale ellipse model."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    skips: list[tf.Tensor] = []
    x = inputs
    for stage, filters in enumerate((16, 24, 32, 48, 64)):
        x = _block(x, filters, 2, f"enc{stage}_down")
        x = _block(x, filters, 1, f"enc{stage}_refine")
        skips.append(x)

    bottleneck_gap = layers.GlobalAveragePooling2D(name="scale_gap")(x)
    scale_confidence = layers.Dense(3, activation="softmax", name="scale_confidence")(bottleneck_gap)

    heads: list[tf.Tensor] = []
    geometry_heads: list[tf.Tensor] = []

    # why: each rim head sees the same gauge but has a resolution suited to
    # one size band; the learned confidence selects the band at decode time.
    for head_index, (filters, skip_index) in enumerate(((48, 3), (32, 2), (24, 1))):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{head_index}")(x)
        x = layers.Concatenate(name=f"join{head_index}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"dec{head_index}")
        rim = layers.Conv2D(1, 1, activation="sigmoid", name=f"rim_{SIZES[head_index]}")(x)
        heads.append(layers.Flatten(name=f"rim_flat_{SIZES[head_index]}")(rim))
        geometry = layers.GlobalAveragePooling2D(name=f"geometry_gap_{SIZES[head_index]}")(x)
        geometry = layers.Dense(24, activation="relu", name=f"geometry_shared_{SIZES[head_index]}")(geometry)
        geometry_heads.append(layers.Dense(4, activation="sigmoid", name=f"geometry_{SIZES[head_index]}")(geometry))

    # SimCC center head: x and y axis logits from the fine decoder features.
    # why: classification into 3-bins-per-pixel beats heatmap stride limits
    # for tiny gauges and quantizes cleanly (softmax logits).
    fine_features = skips[0]
    simcc_proj = layers.Conv2D(32, 1, padding="same", activation="relu", name="simcc_proj")(fine_features)
    simcc_pool = layers.GlobalAveragePooling2D(name="simcc_gap")(simcc_proj)
    simcc_shared = layers.Dense(128, activation="relu", name="simcc_shared")(simcc_pool)
    simcc_x = layers.Dense(SIMCC_BINS, name="simcc_x")(simcc_shared)
    simcc_y = layers.Dense(SIMCC_BINS, name="simcc_y")(simcc_shared)

    output = layers.Concatenate(name="simcc_contract")(
        [*heads, *geometry_heads, scale_confidence, simcc_x, simcc_y]
    )
    return keras.Model(inputs, output, name="ellipse_multiscale_simcc_384")


def make_rim_targets(geometry: np.ndarray) -> list[np.ndarray]:
    """Create Gaussian rim-ring targets at each scale for one sample set."""
    all_maps: list[np.ndarray] = []
    for size in SIZES:
        coords = (np.arange(size, dtype=np.float32) + 0.5) / size
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        rims: list[np.ndarray] = []
        for cx, cy, rx, ry in geometry[:, :4]:
            distance = ((xx - cx) / max(float(rx), 1e-3)) ** 2 + ((yy - cy) / max(float(ry), 1e-3)) ** 2
            rim = np.exp(-0.5 * ((distance - 1.0) / 0.16) ** 2)
            rims.append(rim.astype(np.float32).reshape(-1))
        all_maps.append(np.asarray(rims))
    return all_maps


def make_simcc_targets(geometry: np.ndarray, bins: int, sigma_bins: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Create soft Gaussian targets for SimCC x/y classification.

    ``geometry`` columns are (cx, cy, rx, ry) normalized [0,1]; the target
    bin is center * bins, and a Gaussian around it (sigma_bins) softens the
    label so nearby bins share supervision.  Rows are normalized to sum to 1
    so the targets are proper probability distributions for cross-entropy.
    """
    x_bins = geometry[:, 0] * bins
    y_bins = geometry[:, 1] * bins
    bin_centers = np.arange(bins, dtype=np.float32) + 0.5
    sigma = np.float32(sigma_bins)
    x_targets = np.exp(-0.5 * ((bin_centers[None, :] - x_bins[:, None]) / sigma) ** 2).astype(np.float32)
    y_targets = np.exp(-0.5 * ((bin_centers[None, :] - y_bins[:, None]) / sigma) ** 2).astype(np.float32)
    x_targets /= x_targets.sum(axis=1, keepdims=True)
    y_targets /= y_targets.sum(axis=1, keepdims=True)
    return x_targets, y_targets


def make_contract_targets(geometry: np.ndarray) -> np.ndarray:
    """Build the full 14,415-value contract target for a sample batch."""
    rim_maps = make_rim_targets(geometry)
    max_radius = np.max(geometry[:, 2:4], axis=1)
    scale_index = np.where(max_radius < 0.14, 2, np.where(max_radius < 0.30, 1, 0))
    one_hot = np.eye(3, dtype=np.float32)[scale_index]
    simcc_x, simcc_y = make_simcc_targets(geometry, SIMCC_BINS)
    return np.concatenate(
        [*rim_maps, *[geometry[:, :4]] * 3, one_hot, simcc_x, simcc_y],
        axis=1,
    ).astype(np.float32)


class SimCCLoss(keras.losses.Loss):
    """Train rim rings, geometry, scale confidence, and SimCC center logits."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return a weighted sum over all contract terms."""
        total = tf.zeros(tf.shape(y_pred)[0], dtype=tf.float32)
        offset = 0
        for size in SIZES:
            values = size * size
            true_rim = tf.reshape(y_true[:, offset : offset + values], (-1, size, size))
            pred_rim = tf.reshape(y_pred[:, offset : offset + values], (-1, size, size))
            offset += values
            rim_clip = tf.clip_by_value(pred_rim, 1e-5, 1.0 - 1e-5)
            rim_bce = -(true_rim * tf.math.log(rim_clip) + (1.0 - true_rim) * tf.math.log(1.0 - rim_clip))
            total += 1.5 * tf.reduce_mean((1.0 + 3.0 * true_rim) * rim_bce, axis=(1, 2))
        for _ in SIZES:
            true_geometry = y_true[:, offset : offset + 4]
            pred_geometry = y_pred[:, offset : offset + 4]
            total += 2.0 * tf.reduce_sum(tf.abs(true_geometry - pred_geometry), axis=-1)
            offset += 4
        total += 2.0 * keras.losses.categorical_crossentropy(y_true[:, offset : offset + 3], y_pred[:, offset : offset + 3])
        offset += 3
        true_x = y_true[:, offset : offset + SIMCC_BINS]
        pred_x = y_pred[:, offset : offset + SIMCC_BINS]
        offset += SIMCC_BINS
        true_y = y_true[:, offset : offset + SIMCC_BINS]
        pred_y = y_pred[:, offset : offset + SIMCC_BINS]
        # why: the SimCC heads emit linear logits, so cross-entropy must use
        # from_logits=True; the default softmax path NaNs on zero logits in
        # this tf_keras version.
        total += 0.5 * keras.losses.categorical_crossentropy(true_x, pred_x, from_logits=True)
        total += 0.5 * keras.losses.categorical_crossentropy(true_y, pred_y, from_logits=True)
        return total

    def get_config(self) -> dict[str, object]:
        """Return the serializable loss configuration."""
        return super().get_config()


def export_int8(model: keras.Model, images_u8: np.ndarray, output: Path) -> None:
    """Export a fully integer TFLite model.

    ``images_u8`` is the uint8 [0,255] training set; representative samples
    are converted to float32 [0,1] one at a time so the export never
    materializes a full float32 copy.
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


def softargmax_bins(logits: np.ndarray, bins: int) -> np.ndarray:
    """Decode SimCC logits to a continuous normalized coordinate [0,1]."""
    bin_centers = np.arange(bins, dtype=np.float32) + 0.5
    probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True) + 1e-6
    return np.sum(probabilities * bin_centers[None, :], axis=1) / bins


def decode_contract(contract: np.ndarray) -> np.ndarray:
    """Decode the SimCC contract into (cx, cy, rx, ry, confidence)."""
    confidence = contract[:, CONFIDENCE_OFFSET : CONFIDENCE_OFFSET + 3]
    selected = np.argmax(confidence, axis=1)
    simcc_x = contract[:, SIMCC_OFFSET : SIMCC_OFFSET + SIMCC_BINS]
    simcc_y = contract[:, SIMCC_OFFSET + SIMCC_BINS : SIMCC_OFFSET + 2 * SIMCC_BINS]
    centers_x = softargmax_bins(simcc_x, SIMCC_BINS)
    centers_y = softargmax_bins(simcc_y, SIMCC_BINS)
    predictions = np.zeros((len(contract), 5), dtype=np.float32)
    for row, head in enumerate(selected):
        predictions[row, 0] = centers_x[row]
        predictions[row, 1] = centers_y[row]
        predictions[row, 2:4] = contract[row, GEOMETRY_OFFSET + 4 * head : GEOMETRY_OFFSET + 4 * head + 4][2:4]
        predictions[row, 4] = confidence[row, head]
    return predictions


def _iter_samples(images: np.ndarray, targets: np.ndarray) -> object:
    """Return a generator over (image, contract) host arrays.

    Images are stored as uint8 to keep the full set at ~3.8 GB instead of
    ~15 GB; each sample is converted to float32 on the fly, so only batches
    cross into GPU memory.
    """

    def samples() -> object:
        """Yield one sample at a time from the loaded host arrays."""
        for index in range(len(images)):
            yield images[index].astype(np.float32) / 255.0, targets[index]

    return samples


def _augment_uint8(images: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale/translate/contrast-augment uint8 images, keeping targets aligned.

    Mirrors the iter4 augmenter; works on uint8 frames so the doubled set
    costs ~3.8 GB instead of ~15 GB.
    """
    rng = np.random.default_rng(SEED)
    scales = rng.choice(np.asarray([0.20, 0.30, 0.42, 0.60, 0.80]), size=len(images))
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
        transformed[:2] = 0.5 + scale * (target[:2] - 0.5) + translation
        transformed[2:4] = target[2:4] * scale
        aug_targets[index] = transformed
    return (
        np.concatenate([images, aug_images], axis=0),
        np.concatenate([targets, aug_targets], axis=0),
    )


def _memory_preflight(n_samples: int) -> None:
    """Abort with a readable error if the dataset footprint exceeds the RAM budget."""
    image_bytes = n_samples * IMAGE_SIZE * IMAGE_SIZE  # uint8
    contract_bytes = n_samples * CONTRACT_SIZE * 4  # float32
    shuffle_bytes = min(n_samples, SHUFFLE_BUFFER) * (
        IMAGE_SIZE * IMAGE_SIZE + CONTRACT_SIZE * 4
    )
    total_mb = (image_bytes + contract_bytes + shuffle_bytes) / 1e6
    print(f"memory preflight: {total_mb / 1024:.1f} GiB estimated for {n_samples} samples", flush=True)
    if total_mb > MEMORY_BUDGET_MB:
        raise SystemExit(
            f"aborting: estimated {total_mb / 1024:.1f} GiB exceeds {MEMORY_BUDGET_MB / 1024:.0f} GiB budget"
        )


def main() -> None:
    """Train, quantize, export, and evaluate the SimCC-center model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--qat-epochs", type=int, default=10)
    parser.add_argument("--tiny-repeats", type=int, default=100)
    parser.add_argument("--board-repeats", type=int, default=4)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(TRAIN_BOARD_ZIPS, labels=("temp_dial",))
    images = np.concatenate([generic_images, np.repeat(tiny_images, args.tiny_repeats, axis=0), np.repeat(board_images, args.board_repeats, axis=0)])
    targets = np.concatenate([generic_targets, np.repeat(tiny_targets, args.tiny_repeats, axis=0), np.repeat(board_targets, args.board_repeats, axis=0)])
    images_u8 = np.clip(np.round(images * 255.0), 0, 255).astype(np.uint8)
    del images
    images_u8, targets = _augment_uint8(images_u8, targets)
    contract_targets = make_contract_targets(targets)
    _memory_preflight(len(images_u8))
    dataset = (
        tf.data.Dataset.from_generator(
            _iter_samples(images_u8, contract_targets),
            output_signature=(
                tf.TensorSpec((IMAGE_SIZE, IMAGE_SIZE, 1), tf.float32),
                tf.TensorSpec((CONTRACT_SIZE,), tf.float32),
            ),
        )
        .shuffle(min(SHUFFLE_BUFFER, len(images_u8)), seed=SEED, reshuffle_each_iteration=True)
        .batch(32)
        .prefetch(1)
    )
    print("training", images_u8.shape, contract_targets.shape, "simcc_center", flush=True)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=SimCCLoss())
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=SimCCLoss())
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images_u8, args.output / "model_int8.tflite")

    report: dict[str, object] = {
        "train_samples": int(len(images_u8)),
        "excluded_test3_source": "initial_temp_gauge/board_captures_2.zip",
        "tiny_repeats": args.tiny_repeats,
        "board_repeats": args.board_repeats,
        "simcc_bins": SIMCC_BINS,
        "tests": {},
    }
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        predictions = decode_contract(predict_int8(args.output / "model_int8.tflite", test_images))
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
