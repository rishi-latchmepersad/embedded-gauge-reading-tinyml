#!/usr/bin/env python3
"""Train a pretrained MobileNetV3-Small P2/P3/P4 dense ellipse detector."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_center_ellipse_detector_384 import decode, detector_loss, export_int8, make_targets, predict_int8
from train_ellipse_robust_384 import SEED, load_zips, make_scale_augmented_training_set

IMAGE_SIZE = 384
WEIGHTS = Path("/home/rishi_latchmepersad/.keras/models/weights_mobilenet_v3_small_224_0.75_float_no_top_v2.h5")


def configure_gpu() -> None:
    """Limit TensorFlow to the project's 15 GB GPU budget."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)])


def build_model() -> keras.Model:
    """Build MobileNetV3 P2 fusion with a dense ellipse prediction head."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    rgb = layers.Concatenate(name="gray_to_rgb")([inputs, inputs, inputs])
    backbone = keras.applications.MobileNetV3Small(include_top=False, weights=str(WEIGHTS), input_tensor=rgb, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), alpha=0.75)
    backbone.trainable = True
    # why: these layers preserve stride-4 detail while deeper maps provide
    # semantic context for large and partially occluded gauges.
    p2 = backbone.get_layer("expanded_conv/project/BatchNorm").output
    p3 = backbone.get_layer("expanded_conv_2/Add").output
    p4 = backbone.get_layer("expanded_conv_5/Add").output
    p2_proj = layers.Conv2D(24, 1, padding="same", activation="relu", name="p2_proj")(p2)
    p3_proj = layers.Conv2D(24, 1, padding="same", activation="relu", name="p3_proj")(p3)
    p4_proj = layers.Conv2D(24, 1, padding="same", activation="relu", name="p4_proj")(p4)
    p3_up = layers.UpSampling2D(2, interpolation="nearest", name="p3_up")(p3_proj)
    p4_up = layers.UpSampling2D(4, interpolation="nearest", name="p4_up")(p4_proj)
    fused = layers.Add(name="pyramid_add")([p2_proj, p3_up, p4_up])
    fused = layers.Conv2D(24, 3, padding="same", activation="relu", name="pyramid_refine")(fused)
    output = layers.Conv2D(6, 1, activation="sigmoid", name="ellipse_dense")(fused)
    return keras.Model(inputs, layers.Flatten(name="ellipse_contract")(output), name="mobilenetv3_p2_dense_384")


def main() -> None:
    """Train, QAT-export, and score the V3 dense detector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--qat-epochs", type=int, default=2)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    generic_images, generic_targets = load_zips(["train_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(["initial_temp_gauge/board_captures_1.zip"], labels=("temp_dial",))
    images = np.concatenate([generic_images[:1000], np.repeat(tiny_images, 50, axis=0), np.repeat(board_images, 5, axis=0)])
    targets = np.concatenate([generic_targets[:1000], np.repeat(tiny_targets, 50, axis=0), np.repeat(board_targets, 5, axis=0)])
    images, targets = make_scale_augmented_training_set(images, targets)
    dataset = tf.data.Dataset.from_tensor_slices((images, make_targets(targets))).shuffle(len(images), seed=SEED).map(lambda x, y: (tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)), y), num_parallel_calls=tf.data.AUTOTUNE).batch(2).prefetch(tf.data.AUTOTUNE)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(2e-4), loss=detector_loss)
    model.fit(dataset, epochs=args.epochs, verbose=2)
    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(1e-4), loss=detector_loss)
    qat.fit(dataset, epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "model_qat.weights.h5")
    export_int8(qat, images[:256], args.output / "model_int8.tflite")
    report: dict[str, object] = {"train_samples": int(len(images)), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        test_images, test_targets = _load_zip(zip_name)
        resized = tf.image.resize(test_images, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
        predictions = np.concatenate([decode(predict_int8(args.output / "model_int8.tflite", resized)), np.ones((len(test_targets), 1), dtype=np.float32)], axis=1)
        report["tests"][zip_name] = _metrics(predictions, test_targets)
        print(zip_name, report["tests"][zip_name], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
