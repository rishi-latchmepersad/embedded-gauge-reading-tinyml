#!/usr/bin/env python3
"""Train a tiny spatial YOLO-like gauge-face detector with QAT.

The head predicts one face heatmap plus normalized center/radius values at the
winning grid cell.  This preserves spatial evidence while remaining int8 and
small enough for the STM32N6 deployment path.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import resource
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = 320
GRID_SIZE = 80
SEED = 42


def configure_runtime() -> None:
    """Limit host virtual memory and configure the requested GPU budget."""
    resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3, 50 * 1024**3))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def load_labels(zips: list[Path], limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Load clean CVAT ellipse images and normalized center/radius targets."""
    records: list[tuple[Path, str]] = []
    targets: list[np.ndarray] = []
    for path in zips:
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("annotations.xml"))
            for node in root.findall("image"):
                width = float(node.get("width", 640))
                height = float(node.get("height", 640))
                shape = next((s for s in node if s.get("label") in {"GaugeFace", "temp_dial"}), None)
                if shape is None or shape.tag != "ellipse":
                    continue
                target = np.array([
                    float(shape.get("cx")) / width, float(shape.get("cy")) / height,
                    float(shape.get("rx")) / width, float(shape.get("ry")) / height,
                ], dtype=np.float32)
                records.append((path, node.get("name", "")))
                targets.append(target)
    if limit and len(records) > limit:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(records), limit, replace=False)
        records = [records[i] for i in keep]
        targets = [targets[i] for i in keep]
    images: list[np.ndarray] = []
    handles: dict[str, zipfile.ZipFile] = {}
    for index, (path, name) in enumerate(records):
        archive = handles.setdefault(str(path), zipfile.ZipFile(path))
        member = next(m for m in archive.namelist() if Path(m).name == Path(name).name)
        image = Image.open(io.BytesIO(archive.read(member))).convert("L")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        images.append(np.asarray(image, dtype=np.float32)[..., None] / 255.0)
        if (index + 1) % 1000 == 0:
            print(f"Decoded {index + 1}/{len(records)} images", flush=True)
    for archive in handles.values():
        archive.close()
    return np.asarray(images, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def make_targets(geometry: np.ndarray) -> np.ndarray:
    """Rasterize one Gaussian face-center target and geometry at its cell."""
    targets = np.zeros((len(geometry), GRID_SIZE, GRID_SIZE, 5), dtype=np.float32)
    for index, (cx, cy, rx, ry) in enumerate(geometry):
        gx, gy = cx * GRID_SIZE, cy * GRID_SIZE
        ix, iy = min(GRID_SIZE - 1, int(gx)), min(GRID_SIZE - 1, int(gy))
        yy, xx = np.mgrid[:GRID_SIZE, :GRID_SIZE]
        sigma = 2.0
        targets[index, ..., 0] = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma**2))
        targets[index, iy, ix, 1:] = geometry[index]
    return targets


def build_model() -> keras.Model:
    """Build a compact stride-four spatial detector with an int8-friendly head."""
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x = inputs
    # Two downsampling stages give an 80x80 stride-four detection grid.
    for filters, stride in ((24, 2), (32, 2), (48, 1), (64, 1)):
        x = keras.layers.Conv2D(filters, 3, stride, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    # why: retain the 80x80 map; global pooling would erase position.
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    outputs = keras.layers.Conv2D(5, 1, activation="sigmoid", name="yolo_spatial_head")(x)
    return keras.Model(inputs, outputs, name="tiny_yolo_spatial_ellipse")


def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Weight center heatmap learning and regress geometry only at face cells."""
    heat_true, heat_pred = y_true[..., :1], y_pred[..., :1]
    geom_true, geom_pred = y_true[..., 1:], y_pred[..., 1:]
    heat_loss = tf.keras.losses.binary_crossentropy(heat_true, heat_pred)
    mask = tf.cast(heat_true > 0.5, tf.float32)
    geom_loss = tf.reduce_sum(tf.abs(geom_true - geom_pred) * mask, axis=-1)
    return tf.reduce_mean(heat_loss) + 5.0 * tf.reduce_sum(geom_loss) / (tf.reduce_sum(mask) + 1.0)


def export_int8(model: keras.Model, images: np.ndarray, path: Path) -> None:
    """Export the trained spatial head as fully quantized int8 TFLite."""
    def representative():
        for image in images[:: max(1, len(images) // 256)]:
            yield [image[None]]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(converter.convert())


def main() -> None:
    """Train, QAT-finetune, and export the clean spatial detector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-images", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--qat-epochs", type=int, default=5)
    args = parser.parse_args()
    configure_runtime()
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    labelled = ROOT / "data" / "labelled"
    zips = [labelled / "train_1.zip", labelled / "initial_temp_gauge" / "board_captures_1.zip"]
    images, geometry = load_labels(zips, args.max_train_images)
    targets = make_targets(geometry)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss_fn)
    model.fit(images, targets, batch_size=16, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss=loss_fn)
    qat.fit(images, targets, batch_size=16, epochs=args.qat_epochs, verbose=2)
    export_int8(qat, images, args.output / "tiny_yolo_spatial_int8.tflite")
    (args.output / "report.json").write_text(json.dumps({"train_images": len(images), "input": [1, 320, 320, 1]} , indent=2))


if __name__ == "__main__":
    main()
