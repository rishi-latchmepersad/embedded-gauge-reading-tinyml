"""Train and export the compact grayscale gauge ellipse detector.

The model is a one-object detector for the ``GaugeFace`` CVAT annotation.  It
uses only Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, and
sigmoid operations so the graph has a straightforward STM32N6/TFLite path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot


IMAGE_SIZE = 640
SEED = 42
DEFAULT_EPOCHS = 40
DEFAULT_QAT_EPOCHS = 10


def _configure_gpu() -> None:
    """Cap the local GPU at 15 GB so TensorFlow leaves WSL headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def build_model() -> keras.Model:
    """Build the NPU-oriented single-ellipse regression network."""
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    # The first feature map is 160x160x16 = 400 KiB at int8, below the SRAM limit.
    for index, (filters, stride, repeats) in enumerate(
        ((16, 4, 1), (24, 2, 1), (48, 2, 2), (96, 2, 2), (128, 2, 1))
    ):
        for repeat in range(repeats):
            x = keras.layers.Conv2D(
                filters,
                3,
                strides=stride if repeat == 0 else 1,
                padding="same",
                use_bias=True,
                name=f"stage{index}_conv{repeat}",
            )(x)
            # why: the previous BN+QAT graph collapsed to an input-independent
            # constant after TFLite conversion; bias-only convolutions keep the
            # exported integer graph numerically faithful on the N6 toolchain.
            x = keras.layers.ReLU(name=f"stage{index}_relu{repeat}")(x)
    # why: the previous integer exports showed large drift through TFLite's
    # quantized MEAN reduction; a learned 10x10 convolution collapses the
    # spatial map without introducing that reduction operator.
    x = keras.layers.Conv2D(128, 10, padding="valid", use_bias=True, name="spatial_collapse")(x)
    x = keras.layers.ReLU(name="spatial_collapse_relu")(x)
    x = keras.layers.Flatten(name="spatial_flatten")(x)
    x = keras.layers.Dense(64, activation="relu", name="head_relu")(x)
    # [center_x, center_y, radius_x, radius_y, confidence], all normalized 0..1.
    outputs = keras.layers.Dense(5, activation="sigmoid", name="ellipse")(x)
    return keras.Model(inputs, outputs, name="gauge_ellipse_v1")


def _target_from_label(path: Path) -> np.ndarray:
    """Read one YOLO-OBB rectangle and convert it to ellipse parameters."""
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")
    if values.size < 9:
        raise ValueError(f"Expected one YOLO OBB line in {path}")
    points = values[1:9].reshape(4, 2)
    # The source CVAT ellipse has axis-aligned radii; the OBB corners preserve them.
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) * 0.5
    radius = (high - low) * 0.5
    return np.asarray([center[0], center[1], radius[0], radius[1], 1.0], dtype=np.float32)


def _load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load grayscale PNG paths and normalized ellipse targets for one split."""
    image_paths = sorted((root / "images" / split).glob("*.png"))
    label_paths = [root / "labels" / split / f"{p.stem}.txt" for p in image_paths]
    if not image_paths:
        raise FileNotFoundError(f"No prepared images found for split={split!r} under {root}")
    targets = np.stack([_target_from_label(path) for path in label_paths])
    return np.asarray([str(path) for path in image_paths]), targets


def _dataset(paths: np.ndarray, targets: np.ndarray, batch: int, training: bool) -> tf.data.Dataset:
    """Create a deterministic decoded grayscale input pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def decode(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Decode a PNG and scale it to the model's low-factor input range."""
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        image = tf.cast(image, tf.float32) / 255.0
        return image, target

    return dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def _compile(model: keras.Model, learning_rate: float) -> None:
    """Compile with robust normalized-coordinate regression losses."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.Huber(delta=0.05),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )


def _representative(paths: np.ndarray) -> Iterable[list[np.ndarray]]:
    """Yield calibration images for full integer post-training quantization."""
    for path in paths[: min(200, len(paths))]:
        image = tf.io.decode_png(tf.io.read_file(path), channels=1)
        yield [tf.cast(image[None], tf.float32) / 255.0]


def _export_int8(model: keras.Model, paths: np.ndarray, output: Path) -> dict[str, object]:
    """Export a full-integer TFLite model and record its quantization contract."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative(paths)
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
        "path": str(output),
        "bytes": len(blob),
        "input": {"shape": input_detail["shape"].tolist(), "quantization": input_detail["quantization"]},
        "output": {"shape": output_detail["shape"].tolist(), "quantization": output_detail["quantization"]},
    }


def _activation_report(model: keras.Model) -> dict[str, object]:
    """Compute a conservative int8 activation-size report from static shapes."""
    tensors: list[dict[str, object]] = []
    for layer in model.layers:
        shape = getattr(layer, "output_shape", None)
        if shape is None and hasattr(layer, "output"):
            shape = tuple(layer.output.shape)
        if not shape or not isinstance(shape, tuple):
            continue
        dims = [int(dim) for dim in shape[1:] if dim is not None]
        if len(dims) != 3:
            continue
        size = int(np.prod(dims))
        tensors.append({"layer": layer.name, "shape": dims, "int8_bytes": size})
    largest = max(tensors, key=lambda item: int(item["int8_bytes"]))
    # A two-buffer execution bound covers producer/consumer lifetimes for this chain.
    peak_bound = int(largest["int8_bytes"]) * 2
    return {"largest_activation": largest, "two_buffer_peak_bound_bytes": peak_bound, "under_1MiB": peak_bound <= 1024 * 1024, "all": tensors}


def main() -> None:
    """Train FP32, optionally fine-tune with QAT, export, and write reports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "gauge_face_ellipse_v1_640_gray")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "gauge_ellipse_v1")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--qat-epochs", type=int, default=DEFAULT_QAT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--temp-data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "initial_temp_gauge_v1" / "ellipse",
        help="Optional LittleGood dataset to merge once per sample.",
    )
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    _configure_gpu()

    train_paths, train_targets = _load_split(args.data, "train")
    val_paths, val_targets = _load_split(args.data, "val")
    test_paths, test_targets = _load_split(args.data, "test")
    temp_train_paths, temp_train_targets = _load_split(args.temp_data, "train")
    temp_val_paths, temp_val_targets = _load_split(args.temp_data, "val")
    temp_test_paths, temp_test_targets = _load_split(args.temp_data, "test")
    # Keep each domain sample once; repetition would improve adaptation at the
    # cost of making the model memorize this one gauge family.
    train_paths = np.concatenate((train_paths, temp_train_paths))
    train_targets = np.concatenate((train_targets, temp_train_targets))
    val_paths = np.concatenate((val_paths, temp_val_paths))
    val_targets = np.concatenate((val_targets, temp_val_targets))
    test_paths = np.concatenate((test_paths, temp_test_paths))
    test_targets = np.concatenate((test_targets, temp_test_targets))
    train_ds = _dataset(train_paths, train_targets, args.batch_size, True)
    val_ds = _dataset(val_paths, val_targets, args.batch_size, False)
    test_ds = _dataset(test_paths, test_targets, args.batch_size, False)

    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model()
    _compile(model, 1e-3)
    callbacks = [keras.callbacks.ModelCheckpoint(args.output / "best.weights.h5", save_best_only=True, save_weights_only=True)]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.load_weights(args.output / "best.weights.h5")
    # Save a reloadable FP32 artifact before wrapping it with tfmot QAT layers.
    model.save(args.output / "gauge_ellipse_v1_fp32.keras")
    qat_epochs = 0
    if args.qat_epochs > 0:
        qat_model = tfmot.quantization.keras.quantize_model(model)
        _compile(qat_model, 1e-5)
        qat_model.fit(train_ds, validation_data=val_ds, epochs=args.qat_epochs)
        model = qat_model
        qat_epochs = args.qat_epochs
        # tfmot's generated wrapper is not reliably reloadable as a .keras file;
        # weights remain portable when the same graph is reconstructed for checks.
        model.save_weights(args.output / "gauge_ellipse_v1_qat.weights.h5")
    test_metrics = {key: float(value) for key, value in model.evaluate(test_ds, return_dict=True).items()}
    report = {"model": "gauge_ellipse_v1", "input": [1, IMAGE_SIZE, IMAGE_SIZE, 1], "train_images": len(train_paths), "val_images": len(val_paths), "test_images": len(test_paths), "original_train_images": len(train_paths) - len(temp_train_paths), "little_good_train_images": len(temp_train_paths), "little_good_val_images": len(temp_val_paths), "little_good_test_images": len(temp_test_paths), "temp_oversample_factor": 1, "qat_epochs": qat_epochs, "test_metrics": test_metrics, "activation": _activation_report(model)}
    report["tflite_int8"] = _export_int8(model, train_paths, args.output / "gauge_ellipse_v1_int8.tflite")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
