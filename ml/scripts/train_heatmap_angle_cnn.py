"""Training pipeline for heatmap-based angle CNN.

This script trains a MobileNetV2-based heatmap model to predict needle center
and tip positions from cropped gauge images. The model outputs:
- center_heatmap (112x112): Gaussian peak at dial center
- tip_heatmap (112x112): Gaussian peak at needle tip  
- confidence (scalar): Whether needle is visible

Training data comes from two sources:
1. Geometry-labeled phone photos (353 samples from CVAT annotations)
2. Board captures with synthetic keypoints (76 samples from temperature labels)

The merged dataset provides ~430 labeled samples for training.

Usage:
    # Quick test run
    python ml/scripts/train_heatmap_angle_cnn.py --epochs 5 --steps-per-epoch 20
    
    # Full training run
    python ml/scripts/train_heatmap_angle_cnn.py --epochs 40 --batch-size 8
    
    # With luma crop augmentation
    python ml/scripts/train_heatmap_angle_cnn.py --use-luma-crop --epochs 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import project modules
from embedded_gauge_reading_tinyml.models_geometry import build_heatmap_angle_model
from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)
from embedded_gauge_reading_tinyml.heatmap_losses import (
    weighted_heatmap_mse_loss,
    softargmax_coordinate_loss,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_SIZE: Final[int] = 224
HEATMAP_SIZE: Final[int] = 112
GAUSSIAN_SIGMA: Final[float] = 8.0  # Heatmap Gaussian spread


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    data_dir: Path
    geometry_manifest: str = "geometry_reader_manifest_v2_clean.csv"
    board_manifest: str = "board_captures_labeled_v2.csv"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 21

    # Model
    image_size: int = DEFAULT_IMAGE_SIZE
    heatmap_size: int = HEATMAP_SIZE
    backbone_alpha: float = 0.35
    backbone_frozen_epochs: int = 5
    backbone_frozen: bool = True

    # Training
    batch_size: int = 8
    epochs: int = 40
    learning_rate: float = 1e-4
    steps_per_epoch: int | None = None
    validation_steps: int | None = None

    # Augmentation
    use_luma_crop: bool = False
    jitter_shift: int = 20
    jitter_scale_min: float = 0.85
    jitter_scale_max: float = 1.25

    # Loss weights
    center_heatmap_weight: float = 1.0
    tip_heatmap_weight: float = 2.0  # Tip weighted higher for stability
    coord_loss_weight: float = 1.0
    confidence_weight: float = 0.1

    # Output
    output_dir: Path | None = None

    def save(self, path: Path) -> None:
        """Save config to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "data_dir": str(self.data_dir),
                    "geometry_manifest": self.geometry_manifest,
                    "board_manifest": self.board_manifest,
                    "val_fraction": self.val_fraction,
                    "test_fraction": self.test_fraction,
                    "seed": self.seed,
                    "image_size": self.image_size,
                    "heatmap_size": self.heatmap_size,
                    "backbone_alpha": self.backbone_alpha,
                    "backbone_frozen_epochs": self.backbone_frozen_epochs,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "use_luma_crop": self.use_luma_crop,
                    "center_heatmap_weight": self.center_heatmap_weight,
                    "tip_heatmap_weight": self.tip_heatmap_weight,
                    "coord_loss_weight": self.coord_loss_weight,
                    "confidence_weight": self.confidence_weight,
                },
                f,
                indent=2,
            )


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_geometry_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load geometry-labeled samples from CSV manifest.

    Args:
        manifest_path: Path to geometry_reader_manifest_v2_clean.csv

    Returns:
        List of sample dicts with image_path, center_x/y, tip_x/y, temperature_c, split
    """
    import csv

    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("image_path") or not row.get("center_x_source"):
                continue
            try:
                samples.append(
                    {
                        "image_path": row["image_path"],
                        "center_x": float(row["center_x_source"]),
                        "center_y": float(row["center_y_source"]),
                        "tip_x": float(row["tip_x_source"]),
                        "tip_y": float(row["tip_y_source"]),
                        "temperature_c": float(row["temperature_c"]),
                        "split": row.get("split", "train"),
                        "source_width": int(row["source_width"]),
                        "source_height": int(row["source_height"]),
                        "loose_crop_x1": int(float(row["loose_crop_x1"])),
                        "loose_crop_y1": int(float(row["loose_crop_y1"])),
                        "loose_crop_x2": int(float(row["loose_crop_x2"])),
                        "loose_crop_y2": int(float(row["loose_crop_y2"])),
                        "label_source": "cvat",
                    }
                )
            except (ValueError, KeyError):
                continue
    return samples


def load_board_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load board capture samples from labeled manifest.

    Args:
        manifest_path: Path to board_captures_labeled_v2.csv

    Returns:
        List of sample dicts with image_path, center_x/y, tip_x/y, temperature_c
    """
    import csv

    samples = []
    if not manifest_path.exists():
        return samples

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("image_path") or not row.get("center_x"):
                continue
            try:
                samples.append(
                    {
                        "image_path": row["image_path"],
                        "center_x": float(row["center_x"]),
                        "center_y": float(row["center_y"]),
                        "tip_x": float(row["tip_x"]),
                        "tip_y": float(row["tip_y"]),
                        "temperature_c": float(row["temperature_c"]),
                        "split": "train",  # Board captures all go to train
                        "source_width": int(row["source_width"]),
                        "source_height": int(row["source_height"]),
                        "loose_crop_x1": 0,
                        "loose_crop_y1": 0,
                        "loose_crop_x2": int(row["source_width"]),
                        "loose_crop_y2": int(row["source_height"]),
                        "label_source": row.get("label_source", "synthetic"),
                    }
                )
            except (ValueError, KeyError):
                continue
    return samples


# ---------------------------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------------------------


def build_gaussian_heatmap(
    center_x: float,
    center_y: float,
    heatmap_size: int = HEATMAP_SIZE,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Build a 2D Gaussian heatmap centered at the given coordinates.

    Args:
        center_x: X coordinate in heatmap pixel space [0, heatmap_size)
        center_y: Y coordinate in heatmap pixel space [0, heatmap_size)
        heatmap_size: Output heatmap dimensions
        sigma: Gaussian standard deviation

    Returns:
        Heatmap array of shape (heatmap_size, heatmap_size), values in [0, 1]
    """
    y, x = np.ogrid[:heatmap_size, :heatmap_size]
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    return heatmap.astype(np.float32)


def load_and_preprocess_image(
    image_path: str,
    crop_x1: int,
    crop_y1: int,
    crop_x2: int,
    crop_y2: int,
    center_x_src: float,
    center_y_src: float,
    tip_x_src: float,
    tip_y_src: float,
    image_size: int = DEFAULT_IMAGE_SIZE,
    heatmap_size: int = HEATMAP_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Load and preprocess one training sample.

    Args:
        image_path: Path to image file
        crop_*: Crop box coordinates in source image space
        center_x_src, center_y_src: Center coordinates in source image space
        tip_x_src, tip_y_src: Tip coordinates in source image space
        image_size: Output image size
        heatmap_size: Output heatmap size

    Returns:
        Tuple of (image, center_heatmap, tip_heatmap, angle_degrees, temperature_c)
    """
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)

    # Crop
    crop_h = crop_y2 - crop_y1
    crop_w = crop_x2 - crop_x1
    img = tf.image.crop_to_bounding_box(
        img,
        offset_height=max(0, crop_y1),
        offset_width=max(0, crop_x1),
        target_height=min(crop_h, tf.shape(img)[0] - crop_y1),
        target_width=min(crop_w, tf.shape(img)[1] - crop_x1),
    )

    # Resize with pad
    img = tf.image.resize_with_pad(
        img, image_size, image_size, method="bilinear"
    )
    img = img / 255.0  # Normalize to [0, 1]

    # Transform coordinates to crop space
    crop_x1_f = float(crop_x1)
    crop_y1_f = float(crop_y1)

    # Handle edge cases where crop dimensions might be zero
    crop_w = max(1.0, float(crop_x2) - crop_x1_f)
    crop_h = max(1.0, float(crop_y2) - crop_y1_f)

    center_x_crop = (center_x_src - crop_x1_f) / crop_w
    center_y_crop = (center_y_src - crop_y1_f) / crop_h
    tip_x_crop = (tip_x_src - crop_x1_f) / crop_w
    tip_y_crop = (tip_y_src - crop_y1_f) / crop_h

    # Scale to heatmap coordinates
    center_x_hm = center_x_crop * heatmap_size
    center_y_hm = center_y_crop * heatmap_size
    tip_x_hm = tip_x_crop * heatmap_size
    tip_y_hm = tip_y_crop * heatmap_size

    # Build heatmaps
    center_hm = build_gaussian_heatmap(center_x_hm, center_y_hm, heatmap_size)
    tip_hm = build_gaussian_heatmap(tip_x_hm, tip_y_hm, heatmap_size)

    # Compute angle and temperature
    angle = angle_degrees_from_center_to_tip(
        center_x_hm, center_y_hm, tip_x_hm, tip_y_hm
    )
    temp = celsius_from_inner_dial_angle_degrees(angle)

    return img, center_hm, tip_hm, angle, temp


def create_dataset(
    samples: list[dict[str, Any]],
    batch_size: int = 8,
    image_size: int = DEFAULT_IMAGE_SIZE,
    heatmap_size: int = HEATMAP_SIZE,
    shuffle: bool = True,
    seed: int = 21,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from samples.

    Args:
        samples: List of sample dicts
        batch_size: Batch size
        image_size: Output image size
        heatmap_size: Output heatmap size
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        tf.data.Dataset yielding (image, (center_heatmap, tip_heatmap, confidence)) tuples
    """

    def generator():
        for s in samples:
            try:
                result = load_and_preprocess_image(
                    s["image_path"],
                    s["loose_crop_x1"],
                    s["loose_crop_y1"],
                    s["loose_crop_x2"],
                    s["loose_crop_y2"],
                    s["center_x"],
                    s["center_y"],
                    s["tip_x"],
                    s["tip_y"],
                    image_size,
                    heatmap_size,
                )
                img, center_hm, tip_hm, angle, temp = result
                # Confidence label: 1.0 for CVAT labels, 0.8 for synthetic
                conf = 1.0 if s.get("label_source") == "cvat" else 0.8
                yield img, (center_hm, tip_hm, np.array([conf], dtype=np.float32))
            except Exception as e:
                print(f"Error loading {s['image_path']}: {e}")
                continue

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(heatmap_size, heatmap_size), dtype=tf.float32),
            tf.TensorSpec(shape=(heatmap_size, heatmap_size), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        ),
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(samples), seed=seed)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Model Compilation
# ---------------------------------------------------------------------------


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-4,
    center_weight: float = 1.0,
    tip_weight: float = 2.0,
    coord_weight: float = 1.0,
    confidence_weight: float = 0.1,
) -> None:
    """Compile the heatmap model with appropriate losses and metrics.

    Args:
        model: Heatmap model to compile
        learning_rate: Learning rate
        center_weight: Weight for center heatmap loss
        tip_weight: Weight for tip heatmap loss
        coord_weight: Weight for coordinate loss
        confidence_weight: Weight for confidence loss
    """

    def center_loss(y_true, y_pred):
        return center_weight * keras.losses.MSE(y_true, y_pred)

    def tip_loss(y_true, y_pred):
        return tip_weight * keras.losses.MSE(y_true, y_pred)

    def confidence_loss(y_true, y_pred):
        return confidence_weight * keras.losses.binary_crossentropy(y_true, y_pred)

    # Angle MAE metric (computed from soft-argmax decoded coordinates)
    def angle_mae_deg(y_true_center, y_pred_center):
        # Soft-argmax decode
        center_pred = tf.nn.softmax(tf.reshape(y_pred_center, [-1, HEATMAP_SIZE * HEATMAP_SIZE]), axis=1)
        center_pred = tf.reshape(center_pred, [-1, HEATMAP_SIZE, HEATMAP_SIZE])
        
        # Compute coordinates (simplified - full soft-argmax in training.py)
        cy = tf.reduce_sum(tf.range(HEATMAP_SIZE, dtype=tf.float32)[:, None] * center_pred, axis=(1, 2))
        cx = tf.reduce_sum(tf.range(HEATMAP_SIZE, dtype=tf.float32)[None, :] * center_pred, axis=(1, 2))
        
        # Placeholder - actual angle computation happens in custom training loop
        return tf.constant(0.0)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={
            "center_heatmap": center_loss,
            "tip_heatmap": tip_loss,
            "confidence": confidence_loss,
        },
        metrics={
            "center_heatmap": "mae",
            "tip_heatmap": "mae",
        },
    )


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train(config: TrainConfig) -> keras.Model:
    """Run the full training pipeline.

    Args:
        config: Training configuration

    Returns:
        Trained model
    """
    print(f"Loading data from {config.data_dir}...")

    # Load manifests
    geometry_path = config.data_dir / config.geometry_manifest
    board_path = config.data_dir / config.board_manifest

    geometry_samples = load_geometry_manifest(geometry_path)
    board_samples = load_board_manifest(board_path)

    print(f"Geometry samples: {len(geometry_samples)}")
    print(f"Board samples: {len(board_samples)}")

    # Merge samples
    all_samples = geometry_samples + board_samples
    print(f"Total samples: {len(all_samples)}")

    # Split into train/val/test
    np.random.seed(config.seed)
    indices = np.random.permutation(len(all_samples))
    n_val = int(len(all_samples) * config.val_fraction)
    n_test = int(len(all_samples) * config.test_fraction)
    n_train = len(all_samples) - n_val - n_test

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    test_samples = [all_samples[i] for i in test_indices]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Create datasets
    train_ds = create_dataset(
        train_samples,
        config.batch_size,
        config.image_size,
        config.heatmap_size,
        shuffle=True,
        seed=config.seed,
    )
    val_ds = create_dataset(
        val_samples,
        config.batch_size,
        config.image_size,
        config.heatmap_size,
        shuffle=False,
    )

    # Build model
    print("Building model...")
    model = build_heatmap_angle_model(
        input_shape=(config.image_size, config.image_size, 3),
        alpha=config.backbone_alpha,
        backbone_frozen=config.backbone_frozen,
        heatmap_size=config.heatmap_size,
    )
    model.summary()

    # Compile
    compile_model(
        model,
        config.learning_rate,
        config.center_heatmap_weight,
        config.tip_heatmap_weight,
        config.coord_loss_weight,
        config.confidence_weight,
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.output_dir / "checkpoint_{epoch:02d}.keras") if config.output_dir else "checkpoint_{epoch:02d}.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        keras.callbacks.CSVLogger(
            str(config.output_dir / "training.log") if config.output_dir else "training.log"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    # Train
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_steps=config.validation_steps,
        callbacks=callbacks,
    )

    # Save final model
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        model.save(config.output_dir / "model.keras")
        config.save(config.output_dir / "config.json")
        print(f"Model saved to {config.output_dir / 'model.keras'}")

    return model


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train heatmap angle CNN")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Path to ml/data directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Steps per epoch (None = use all data)",
    )
    parser.add_argument(
        "--backbone-frozen-epochs",
        type=int,
        default=5,
        help="Epochs with frozen backbone",
    )
    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        steps_per_epoch=args.steps_per_epoch,
        backbone_frozen_epochs=args.backbone_frozen_epochs,
    )

    train(config)


if __name__ == "__main__":
    main()
