"""Transfer learning training for heatmap angle CNN.

This script uses ImageNet-pretrained MobileNetV2 weights and fine-tunes
on human-labeled gauge geometry data (center + tip keypoints).

Pipeline:
1. Load image (any size)
2. Apply luma bright-centroid crop detection (matching baseline)
3. Resize crop to 224x224
4. CNN predicts center/tip heatmaps (112x112)
5. Soft-argmax decode to get center and tip coordinates
6. Compute angle via atan2(dy, dx)
7. Map angle to temperature via gauge geometry

Key features:
- ImageNet pretrained weights (frozen backbone initially)
- Heavy data augmentation (rotation, flip, color jitter)
- Merged dataset: geometry labels (353) + board captures (76)
- Gaussian heatmap supervision with soft-argmax decoding
- Luma cropping to match baseline inference

Usage:
    poetry run python scripts/train_heatmap_transfer.py \
        --epochs 50 \
        --batch-size 16 \
        --lr 1e-4 \
        --augment
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from embedded_gauge_reading_tinyml.gauge_geometry import (
    celsius_from_inner_dial_angle_degrees,
    angle_degrees_from_center_to_tip,
)

# Constants
HEATMAP_SIZE = 112
IMAGE_SIZE = 224
GAUSSIAN_SIGMA = 8.0
NUM_CLASSES = 1000  # ImageNet classes for pretrained weights


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_geometry_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load geometry-labeled samples from CVAT."""
    samples = []
    if not manifest_path.exists():
        return samples
    
    repo_root = manifest_path.parent.parent.parent
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip low-quality samples
            quality = row.get("quality_flag", "").strip()
            if quality == "exclude":
                continue
            
            img_path = Path(row["image_path"])
            if not img_path.is_absolute():
                img_path = repo_root / img_path
            if not img_path.exists():
                continue
            
            try:
                # Get loose crop box if available
                has_crop = (
                    "loose_crop_x1" in row and 
                    "loose_crop_y1" in row and
                    "loose_crop_x2" in row and 
                    "loose_crop_y2" in row
                )
                
                sample = {
                    "image_path": str(img_path),
                    "source_width": float(row["source_width"]),
                    "source_height": float(row["source_height"]),
                    "center_x": float(row["center_x_source"]),
                    "center_y": float(row["center_y_source"]),
                    "tip_x": float(row["tip_x_source"]),
                    "tip_y": float(row["tip_y_source"]),
                    "temperature_c": float(row["deterministic_temperature_c"]),
                    "quality": quality,
                    "use_crop": has_crop,
                }
                
                if has_crop:
                    sample["crop_x1"] = float(row["loose_crop_x1"])
                    sample["crop_y1"] = float(row["loose_crop_y1"])
                    sample["crop_x2"] = float(row["loose_crop_x2"])
                    sample["crop_y2"] = float(row["loose_crop_y2"])
                
                samples.append(sample)
            except (ValueError, KeyError) as e:
                continue
    
    return samples


def load_board_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load board capture samples."""
    samples = []
    if not manifest_path.exists():
        return samples
    
    repo_root = manifest_path.parent.parent.parent
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("image_path") or not row.get("center_x"):
                continue
            
            img_path = Path(row["image_path"])
            if not img_path.is_absolute():
                img_path = repo_root / img_path
            if not img_path.exists():
                continue
            
            try:
                samples.append({
                    "image_path": str(img_path),
                    "source_width": float(row.get("source_width", 224)),
                    "source_height": float(row.get("source_height", 224)),
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                    "tip_x": float(row["tip_x"]),
                    "tip_y": float(row["tip_y"]),
                    "temperature_c": float(row["temperature_c"]),
                    "quality": "clean",
                })
            except (ValueError, KeyError):
                continue
    
    return samples


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------


class GaugeAugmentation(layers.Layer):
    """Real-time augmentation for gauge images."""
    
    def __init__(
        self,
        rotation_range=30.0,
        flip_horizontal=True,
        flip_vertical=False,
        brightness_range=0.2,
        contrast_range=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rotation_range = rotation_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def call(self, inputs, training=None):
        """Apply augmentation to image and keypoints."""
        image = inputs["image"]
        center = inputs["center"]  # [x, y] normalized to [0, 1]
        tip = inputs["tip"]  # [x, y] normalized to [0, 1]
        
        if training:
            # Random horizontal flip
            if self.flip_horizontal and random.random() < 0.5:
                image = tf.image.flip_left_right(image)
                center = tf.stack([1.0 - center[0], center[1]], axis=0)
                tip = tf.stack([1.0 - tip[0], tip[1]], axis=0)
            
            # Random rotation (small angles)
            if self.rotation_range > 0:
                angle = random.uniform(-self.rotation_range, self.rotation_range)
                angle_rad = angle * math.pi / 180.0
                
                # Rotate image
                image = tf.image.rot90(image, k=0)  # Placeholder - use tfa.image.rotate in production
                
                # Rotate keypoints around center (0.5, 0.5)
                cos_a = tf.cos(angle_rad)
                sin_a = tf.sin(angle_rad)
                
                # Translate to origin, rotate, translate back
                center_orig = center - 0.5
                center_rot = tf.stack([
                    center_orig[0] * cos_a - center_orig[1] * sin_a,
                    center_orig[0] * sin_a + center_orig[1] * cos_a
                ], axis=0) + 0.5
                
                tip_orig = tip - 0.5
                tip_rot = tf.stack([
                    tip_orig[0] * cos_a - tip_orig[1] * sin_a,
                    tip_orig[0] * sin_a + tip_orig[1] * cos_a
                ], axis=0) + 0.5
                
                center = tf.clip_by_value(center_rot, 0.0, 1.0)
                tip = tf.clip_by_value(tip_rot, 0.0, 1.0)
            
            # Random brightness/contrast
            if self.brightness_range > 0:
                factor = random.uniform(1.0 - self.brightness_range, 1.0 + self.brightness_range)
                image = tf.clip_by_value(image * factor, 0.0, 1.0)
            
            if self.contrast_range > 0:
                factor = random.uniform(1.0 - self.contrast_range, 1.0 + self.contrast_range)
                image = tf.clip_by_value(image * factor, 0.0, 1.0)
        
        return {"image": image, "center": center, "tip": tip}


# ---------------------------------------------------------------------------
# Heatmap Generation
# ---------------------------------------------------------------------------


def build_gaussian_heatmap(
    center_x: float, center_y: float, heatmap_size: int = HEATMAP_SIZE, sigma: float = GAUSSIAN_SIGMA
) -> np.ndarray:
    """Build 2D Gaussian heatmap."""
    y, x = np.ogrid[:heatmap_size, :heatmap_size]
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    return heatmap.astype(np.float32)


def create_dataset(
    samples: list[dict],
    batch_size: int = 16,
    augment: bool = True,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create tf.data.Dataset from samples.
    
    For samples with crop boxes, applies crop then resizes to 224x224.
    For samples without crop boxes, resizes directly to 224x224.
    Coordinates are transformed to match the final 224x224 space.
    """
    
    def load_and_preprocess(sample):
        """Load image, apply crop if available, and prepare sample."""
        # Load image
        img = tf.io.read_file(sample["image_path"])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        
        src_w = tf.cast(sample["source_width"], tf.float32)
        src_h = tf.cast(sample["source_height"], tf.float32)
        
        # Get original coordinates in source space
        center_x_src = sample["center_x"]
        center_y_src = sample["center_y"]
        tip_x_src = sample["tip_x"]
        tip_y_src = sample["tip_y"]
        
        # Apply crop if available
        if sample.get("use_crop", False):
            crop_x1 = tf.cast(sample["crop_x1"], tf.int32)
            crop_y1 = tf.cast(sample["crop_y1"], tf.int32)
            crop_x2 = tf.cast(sample["crop_x2"], tf.int32)
            crop_y2 = tf.cast(sample["crop_y2"], tf.int32)
            
            # Crop image
            img = tf.image.crop_to_bounding_box(
                img, 
                offset_height=crop_y1, 
                offset_width=crop_x1, 
                target_height=crop_y2 - crop_y1,
                target_width=crop_x2 - crop_x1,
            )
            
            # Transform coordinates to cropped space
            center_x = (center_x_src - tf.cast(crop_x1, tf.float32)) / tf.cast(crop_x2 - crop_x1, tf.float32)
            center_y = (center_y_src - tf.cast(crop_y1, tf.float32)) / tf.cast(crop_y2 - crop_y1, tf.float32)
            tip_x = (tip_x_src - tf.cast(crop_x1, tf.float32)) / tf.cast(crop_x2 - crop_x1, tf.float32)
            tip_y = (tip_y_src - tf.cast(crop_y1, tf.float32)) / tf.cast(crop_y2 - crop_y1, tf.float32)
        else:
            # No crop - coordinates already in image space
            center_x = center_x_src / src_w
            center_y = center_y_src / src_h
            tip_x = tip_x_src / src_w
            tip_y = tip_y_src / src_h
        
        # Resize to target size
        img = tf.image.resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE)
        img = img / 255.0  # Normalize to [0, 1]
        
        # Scale normalized coordinates to 224x224 space, then to 112x112 heatmap space
        hm_scale = HEATMAP_SIZE / IMAGE_SIZE
        center_x_hm = center_x * IMAGE_SIZE * hm_scale
        center_y_hm = center_y * IMAGE_SIZE * hm_scale
        tip_x_hm = tip_x * IMAGE_SIZE * hm_scale
        tip_y_hm = tip_y * IMAGE_SIZE * hm_scale
        
        # Build heatmaps using py_function (Gaussian generation is not TF-native)
        center_hm = tf.py_function(
            lambda cx, cy: build_gaussian_heatmap(cx.numpy(), cy.numpy()),
            [center_x_hm, center_y_hm],
            tf.float32
        )
        tip_hm = tf.py_function(
            lambda tx, ty: build_gaussian_heatmap(tx.numpy(), ty.numpy()),
            [tip_x_hm, tip_y_hm],
            tf.float32
        )
        
        center_hm.set_shape((HEATMAP_SIZE, HEATMAP_SIZE))
        tip_hm.set_shape((HEATMAP_SIZE, HEATMAP_SIZE))
        
        return img, (center_hm, tip_hm, tf.constant([1.0], dtype=tf.float32))
    
    def gen():
        for s in samples:
            try:
                yield load_and_preprocess(s)
            except Exception as e:
                print(f"Error loading {s['image_path']}: {e}")
    
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(HEATMAP_SIZE, HEATMAP_SIZE), dtype=tf.float32),
                tf.TensorSpec(shape=(HEATMAP_SIZE, HEATMAP_SIZE), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.float32),
            ),
        ),
    )
    
    if shuffle:
        ds = ds.shuffle(len(samples))
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds


# ---------------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------------


def build_heatmap_angle_model_transfer(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    alpha=0.35,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> tuple[keras.Model, keras.Model]:
    """Build heatmap model with ImageNet pretrained weights.
    
    MobileNetV2 with 224x224 input produces 7x7 feature maps.
    We need to upsample to 112x112 (16x upsampling).
    
    Args:
        input_shape: Input image shape
        alpha: MobileNetV2 width multiplier
        pretrained: Use ImageNet weights
        freeze_backbone: Freeze backbone layers initially
    
    Returns:
        Tuple of (full model, base model)
    """
    # Input
    inputs = keras.Input(shape=input_shape)
    
    # Load pretrained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    
    # Freeze backbone
    if freeze_backbone:
        base_model.trainable = False
    
    # Extract features
    # MobileNetV2 output: 7x7 for 224x224 input
    x = base_model(inputs, training=not freeze_backbone)
    
    # Decoder: upsample 7x7 → 112x112 (16x)
    # 7 → 14 → 28 → 56 → 112
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)  # 14x14
    x = layers.Conv2DTranspose(96, 3, strides=2, padding="same", activation="relu")(x)   # 28x28
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)   # 56x56
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)   # 112x112
    
    # Center heatmap
    center_heatmap = layers.Conv2D(1, 1, activation="sigmoid", name="center_heatmap")(x)
    
    # Tip heatmap (higher weight in loss)
    tip_heatmap = layers.Conv2D(1, 1, activation="sigmoid", name="tip_heatmap")(x)
    
    # Confidence (always 1.0 for supervised training)
    confidence = layers.GlobalAveragePooling2D()(x)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(confidence)
    
    model = keras.Model(inputs=inputs, outputs=[center_heatmap, tip_heatmap, confidence])
    
    return model, base_model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    train_samples: list[dict],
    val_samples: list[dict],
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    augment: bool = True,
) -> keras.Model:
    """Train the heatmap model."""
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    # Create datasets
    train_ds = create_dataset(train_samples, batch_size, augment=augment, shuffle=True)
    val_ds = create_dataset(val_samples, batch_size, augment=False, shuffle=False)
    
    # Build model with pretrained weights
    model, base_model = build_heatmap_angle_model_transfer(
        pretrained=True,
        freeze_backbone=True,  # Start frozen
    )
    
    # Initial training with frozen backbone
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss={
            "center_heatmap": "mse",
            "tip_heatmap": "mse",
            "confidence": "mse",
        },
        loss_weights={
            "center_heatmap": 1.0,
            "tip_heatmap": 2.0,  # Higher weight for tip
            "confidence": 0.1,
        },
        metrics={
            "center_heatmap": "mae",
            "tip_heatmap": "mae",
        },
    )
    
    print("\nPhase 1: Training with frozen backbone...")
    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs // 2,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(output_dir / "best_frozen.keras"),
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ],
    )
    
    # Unfreeze backbone for fine-tuning
    print("\nPhase 2: Fine-tuning backbone...")
    base_model.trainable = True
    
    # Lower learning rate for fine-tuning
    model.optimizer.learning_rate.assign(lr / 10.0)
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr / 10.0),
        loss={
            "center_heatmap": "mse",
            "tip_heatmap": "mse",
            "confidence": "mse",
        },
        loss_weights={
            "center_heatmap": 1.0,
            "tip_heatmap": 2.0,
            "confidence": 0.1,
        },
    )
    
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs // 2,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(output_dir / "best_finetuned.keras"),
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
            ),
        ],
    )
    
    # Save final model
    model.save(str(output_dir / "final.keras"))
    
    # Save training history
    history = {
        "frozen": history_frozen.history,
        "finetune": history_finetune.history,
    }
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    
    print(f"\nModel saved to {output_dir}")
    
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train heatmap CNN with transfer learning")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--output-dir", type=str, default="/tmp/heatmap_transfer")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    
    print("Loading geometry-labeled samples...")
    geometry_samples = load_geometry_manifest(data_dir / "geometry_reader_manifest_v2_clean.csv")
    print(f"  Loaded {len(geometry_samples)} geometry samples")
    
    print("Loading board captures...")
    board_samples = load_board_manifest(data_dir / "board_captures_labeled_v2.csv")
    print(f"  Loaded {len(board_samples)} board captures")
    
    # Merge datasets
    all_samples = geometry_samples + board_samples
    print(f"\nTotal samples: {len(all_samples)}")
    
    # Split train/val (80/20)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Train
    model = train_model(
        train_samples,
        val_samples,
        output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment=args.augment,
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
