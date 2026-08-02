#!/usr/bin/env python3
"""Train a 320-pixel fully-convolutional ellipse model on both domains.

Each source image appears once in the merged training set.  The model keeps
the 640-pixel grayscale frame contract at dataset level, but preserves more
small-gauge detail than the 160-pixel vector regressor before producing an
80x80 integer mask.
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

from train_gauge_ellipse_mask_v1 import (
    SEED,
    boxes_from_masks,
    build_model,
    dataset,
    export_int8,
    load_split,
    predict_int8,
)


def configure_gpu() -> None:
    """Cap the training GPU at 15 GB so WSL retains desktop headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


def merge_split(
    generic_root: Path, littlegood_root: Path, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """Merge generic and LittleGood paths without duplicating any frame."""
    generic_paths, generic_targets = load_split(generic_root, split)
    student_paths, student_targets = load_split(littlegood_root, split)
    return (
        np.concatenate((generic_paths, student_paths)),
        np.concatenate((generic_targets, student_targets)),
    )


def evaluate(
    model_path: Path, paths: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """Evaluate the exported int8 mask decoder on a held-out split."""
    images = np.stack(
        [
            tf.image.resize(
                tf.io.decode_png(tf.io.read_file(path), channels=1), [320, 320]
            ).numpy()
            for path in paths
        ]
    ).astype(np.float32) / 255.0
    predicted = boxes_from_masks(predict_int8(model_path, images))
    true_low = targets[:, :2] - targets[:, 2:]
    true_high = targets[:, :2] + targets[:, 2:]
    pred_low = predicted[:, :2] - predicted[:, 2:]
    pred_high = predicted[:, :2] + predicted[:, 2:]
    overlap = np.maximum(0.0, np.minimum(true_high, pred_high) - np.maximum(true_low, pred_low))
    intersection = overlap[:, 0] * overlap[:, 1]
    true_area = np.prod(true_high - true_low, axis=1)
    pred_area = np.prod(pred_high - pred_low, axis=1)
    iou = intersection / np.maximum(true_area + pred_area - intersection, 1e-6)
    center_error = np.linalg.norm(predicted[:, :2] - targets[:, :2], axis=1) * 640.0
    return {
        "samples": float(len(paths)),
        "iou_mean": float(iou.mean()),
        "iou_ge_0_5": float(np.mean(iou >= 0.5)),
        "center_within_16px": float(np.mean(center_error <= 16.0)),
        "center_error_px_mean": float(center_error.mean()),
    }


def main() -> None:
    """Train, QAT-finetune, export, and test the mixed-domain candidate."""
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--generic-data", type=Path, default=root / "data" / "gauge_face_ellipse_v1_640_gray")
    parser.add_argument("--littlegood-data", type=Path, default=root / "data" / "initial_temp_gauge_v1" / "ellipse")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "gauge_ellipse_mask_mixed_littlegood_v1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--qat-epochs", type=int, default=5)
    args = parser.parse_args()
    configure_gpu()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    train_paths, train_targets = merge_split(args.generic_data, args.littlegood_data, "train")
    val_paths, val_targets = merge_split(args.generic_data, args.littlegood_data, "val")
    test_paths, test_targets = load_split(args.littlegood_data, "test")
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    model.fit(dataset(train_paths, train_targets, 32, True), validation_data=dataset(val_paths, val_targets, 32, False), epochs=args.epochs, verbose=2)

    qat = tfmot.quantization.keras.quantize_model(model)
    qat.compile(optimizer=keras.optimizers.Adam(2e-4), loss="binary_crossentropy")
    qat.fit(dataset(train_paths, train_targets, 32, True), validation_data=dataset(val_paths, val_targets, 32, False), epochs=args.qat_epochs, verbose=2)
    args.output.mkdir(parents=True, exist_ok=True)
    qat.save_weights(args.output / "gauge_ellipse_mask_mixed_v1_qat.weights.h5")
    contract = export_int8(qat, train_paths, args.output / "gauge_ellipse_mask_mixed_v1_int8.tflite")
    metrics = evaluate(args.output / "gauge_ellipse_mask_mixed_v1_int8.tflite", test_paths, test_targets)
    report = {"train_samples": len(train_paths), "val_samples": len(val_paths), "littlegood_test": metrics, "qat_epochs": args.qat_epochs, "contract": contract}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
