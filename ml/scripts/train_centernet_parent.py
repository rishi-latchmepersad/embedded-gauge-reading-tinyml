#!/usr/bin/env python3
"""Train a CenterNet (Objects as Points) parent model for gauge center detection.

Phase 1 of the center detector pipeline:
  - Trains a ResNet-50 CenterNet on the geometry manifest to predict the
    gauge dial center as a keypoint heatmap + sub-pixel offset.
  - Target: center MAE < 5px at source resolution (downstream SimCC + polar
    voting models handle angle → temperature).
  - The trained model serves as the teacher for knowledge distillation
    into a MobileNetV2 student model (Phase 2).

Usage:
    cd ml
    poetry run python scripts/train_centernet_parent.py
    poetry run python scripts/train_centernet_parent.py --epochs 80 --batch-size 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

# Add ml/src to sys.path for imports.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
_SRC_DIR: Path = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from embedded_gauge_reading_tinyml.centernet import (
    CenterNetConfig,
    centernet_loss,
    GeometryManifestRow,
    build_centernet_resnet50,
    build_centernet_tf_dataset,
    decode_centernet_batch,
    load_geometry_manifest,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ML_ROOT: Path = _PROJECT_ROOT
DATA_DIR: Path = ML_ROOT / "data"
ARTIFACTS_DIR: Path = ML_ROOT / "artifacts"

# Use the most comprehensive geometry manifest.
DEFAULT_MANIFEST: Path = DATA_DIR / "geometry_heatmap_v12_all_data_manifest.csv"

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Configuration for CenterNet parent training."""

    # Data.
    manifest_path: Path = field(default_factory=lambda: DEFAULT_MANIFEST)
    input_height: int = 384
    input_width: int = 384
    heatmap_height: int = 96
    heatmap_width: int = 96

    # Training.
    batch_size: int = 2
    epochs: int = 60
    initial_lr: float = 1.25e-4
    min_lr: float = 1e-7

    # Loss weights.
    heatmap_weight: float = 1.0
    offset_weight: float = 1.0
    focal_alpha: float = 2.0
    focal_beta: float = 4.0

    # Augmentation.
    augment: bool = True
    sigma_pixels: float = 2.0

    # Regularization.
    l2_weight_decay: float = 1e-5

    # Scheduling.
    lr_patience: int = 8
    lr_factor: float = 0.5
    early_stop_patience: int = 20

    # Logging.
    output_dir: Path = field(
        default_factory=lambda: ARTIFACTS_DIR
        / "training"
        / f"centernet_parent_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_name: str = "centernet_resnet50_parent"

    # Hardware.
    gpu_memory_growth: bool = True
    mixed_precision: bool = False  # ResNet-50 is stable in float32.


# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------


def _setup_gpu(config: TrainConfig) -> None:
    """Configure GPU memory growth and optional mixed precision."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("WARNING: No GPU found. Training will be slow on CPU.")
        return

    for gpu in gpus:
        if config.gpu_memory_growth:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU: {gpu.name}")

    if config.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: enabled")


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def _build_model(config: TrainConfig) -> keras.Model:
    """Build the CenterNet ResNet-50 parent model."""
    centernet_cfg = CenterNetConfig(
        input_height=config.input_height,
        input_width=config.input_width,
        heatmap_height=config.heatmap_height,
        heatmap_width=config.heatmap_width,
        backbone="resnet50",
        backbone_weights="imagenet",
        backbone_trainable=True,
        decoder_filters=256,
        head_filters=64,
    )
    model = build_centernet_resnet50(centernet_cfg)

    # Add L2 weight decay to all Conv2D and Conv2DTranspose layers.
    if config.l2_weight_decay > 0:
        l2_reg = keras.regularizers.L2(config.l2_weight_decay)
        for layer in model.layers:
            if isinstance(
                layer, (keras.layers.Conv2D, keras.layers.Conv2DTranspose)
            ):
                if layer.kernel_regularizer is None:
                    layer.kernel_regularizer = l2_reg

    return model


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def _build_datasets(
    config: TrainConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build train and validation tf.data.Dataset pipelines."""
    # Load all rows.
    all_rows: list[GeometryManifestRow] = load_geometry_manifest(
        config.manifest_path, splits=("train", "val")
    )

    # Split into train/val by the 'split' column.
    train_rows = [r for r in all_rows if r.split == "train"]
    val_rows = [r for r in all_rows if r.split == "val"]

    print(f"Train samples: {len(train_rows)}, Val samples: {len(val_rows)}")

    train_ds = build_centernet_tf_dataset(
        train_rows,
        input_height=config.input_height,
        input_width=config.input_width,
        heatmap_height=config.heatmap_height,
        heatmap_width=config.heatmap_width,
        sigma_pixels=config.sigma_pixels,
        batch_size=config.batch_size,
        shuffle=True,
        augment=config.augment,
    )

    val_ds = build_centernet_tf_dataset(
        val_rows,
        input_height=config.input_height,
        input_width=config.input_width,
        heatmap_height=config.heatmap_height,
        heatmap_width=config.heatmap_width,
        sigma_pixels=config.sigma_pixels,
        batch_size=config.batch_size,
        shuffle=False,
        augment=False,
    )

    # Log dataset shapes for debugging.
    for img, target in train_ds.take(1):
        print(f"Train batch shapes: img={img.shape}, target={target.shape}  (H,W,3)")

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _build_callbacks(config: TrainConfig) -> list[keras.callbacks.Callback]:
    """Build training callbacks: LR scheduler, early stopping, checkpointing."""
    callbacks: list[keras.callbacks.Callback] = []

    # Reduce LR on plateau.
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        )
    )

    # Early stopping.
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # Model checkpoint — save best weights.
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    )

    # CSV logger.
    callbacks.append(
        keras.callbacks.CSVLogger(
            str(config.output_dir / "training_log.csv"),
        )
    )

    return callbacks


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_on_val(
    model: keras.Model,
    val_rows: list[GeometryManifestRow],
    config: TrainConfig,
) -> dict[str, float]:
    """Evaluate center MAE in both heatmap pixels and source-image pixels."""
    # Build a non-augmented val dataset for prediction.
    val_ds = build_centernet_tf_dataset(
        val_rows,
        input_height=config.input_height,
        input_width=config.input_width,
        heatmap_height=config.heatmap_height,
        heatmap_width=config.heatmap_width,
        sigma_pixels=config.sigma_pixels,
        batch_size=config.batch_size,
        shuffle=False,
        augment=False,
    )

    pixel_errors: list[float] = []
    source_pixel_errors: list[float] = []

    output_stride = config.input_height // config.heatmap_height

    for batch_idx, (images, target) in enumerate(val_ds):
        pred = model.predict(images, verbose=0)

        # Split concatenated tensors: heatmap, offset.
        hm_pred = pred[..., 0:1]
        off_pred = pred[..., 1:3]
        hm_gt_np = target.numpy()[..., 0:1]
        off_gt_np = target.numpy()[..., 1:3]

        # Decode predictions to heatmap coordinates.
        detections = decode_centernet_batch(
            hm_pred, off_pred, topk=1, min_score=0.1
        )

        for i, dets in enumerate(detections):
            if not dets:
                pixel_errors.append(float(config.heatmap_height))
                continue

            cx_hm, cy_hm, _ = dets[0]

            # Ground truth: find argmax in the ground truth heatmap.
            gt_hm = hm_gt_np[i].squeeze()
            gt_cy, gt_cx = np.unravel_index(np.argmax(gt_hm), gt_hm.shape)
            gt_cx += off_gt_np[i, int(gt_cy), int(gt_cx), 0]
            gt_cy += off_gt_np[i, int(gt_cy), int(gt_cx), 1]

            # Error in heatmap pixels.
            hm_error = np.sqrt((cx_hm - gt_cx) ** 2 + (cy_hm - gt_cy) ** 2)
            pixel_errors.append(float(hm_error))

            # Error in source pixels.
            src_error = hm_error * output_stride
            source_pixel_errors.append(float(src_error))

        # Only evaluate first 200 samples for speed.
        if (batch_idx + 1) * config.batch_size >= 200:
            break

    results: dict[str, float] = {
        "center_mae_hm_pixels": float(np.mean(pixel_errors)),
        "center_mae_source_pixels": float(np.mean(source_pixel_errors)),
        "center_rmse_hm_pixels": float(np.sqrt(np.mean(np.square(pixel_errors)))),
    }

    print(f"\nValidation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")

    return results


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(config: TrainConfig) -> keras.Model:
    """Run the full CenterNet parent training pipeline."""
    # Setup.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_gpu(config)

    # Save config.
    with open(config.output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    # Data.
    train_ds, val_ds = _build_datasets(config)

    # Model.
    model = _build_model(config)
    model.summary()

    # Compile.
    # Use functools.partial to bind loss hyperparameters into a plain callable.
    from functools import partial
    loss_fn = partial(
        centernet_loss,
        heatmap_weight=config.heatmap_weight,
        offset_weight=config.offset_weight,
        focal_alpha=config.focal_alpha,
        focal_beta=config.focal_beta,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.initial_lr,
            clipnorm=1.0,
        ),
        loss=loss_fn,
    )

    # Callbacks.
    callbacks = _build_callbacks(config)

    # Train.
    print(f"\nStarting training: {config.run_name}")
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"Output dir: {config.output_dir}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model (recompile with serializable loss first for model.save()).
    model.compile(optimizer=model.optimizer, loss="mse")
    model.save(config.output_dir / "final_model.keras")
    print(f"Model saved to {config.output_dir / 'final_model.keras'}")

    # Evaluate.
    all_rows = load_geometry_manifest(
        config.manifest_path, splits=("val",)
    )
    results = evaluate_on_val(model, all_rows, config)

    with open(config.output_dir / "val_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CenterNet ResNet-50 parent for gauge center detection."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to geometry manifest CSV.",
    )
    parser.add_argument(
        "--epochs", type=int, default=60, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size (2 fits in 4GB GPU)."
    )
    parser.add_argument(
        "--lr", type=float, default=1.25e-4, help="Initial learning rate."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Custom output directory."
    )
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable augmentation."
    )

    args = parser.parse_args()

    config = TrainConfig(
        manifest_path=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_lr=args.lr,
        augment=not args.no_augment,
    )

    if args.output_dir is not None:
        config.output_dir = args.output_dir

    train(config)


if __name__ == "__main__":
    _main()
