"""Simple supervised training script for gauge reading using only image paths and values.

This script bypasses the CVAT annotation requirement and trains directly on
image paths and temperature values from a CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"


def load_manifest(manifest_path: Path) -> List[Tuple[str, float]]:
    """Load image paths and values from CSV manifest."""
    samples = []
    with open(manifest_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['image_path'].startswith('#'):
                continue
            img_path = row['image_path']
            if not Path(img_path).is_absolute():
                img_path = str(REPO_ROOT / img_path)
            samples.append((img_path, float(row['value'])))
    return samples


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> tf.Tensor:
    """Load and preprocess an image."""
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, [None, None, 3])
    
    # Resize with padding to preserve aspect ratio
    image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
    
    # Normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    return image


def create_dataset(
    samples: List[Tuple[str, float]],
    batch_size: int = 16,
    image_size: Tuple[int, int] = (224, 224),
    augment: bool = False,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from samples."""
    image_paths = [s[0] for s in samples]
    values = [s[1] for s in samples]
    
    # Normalize values to [-1, 1] range (assuming -30 to 50 range)
    min_val, max_val = -30.0, 50.0
    values_norm = [(v - min_val) / (max_val - min_val) * 2 - 1 for v in values]
    
    def load_sample(path, value):
        image = preprocess_image(path, image_size)
        return image, value
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, values_norm))
    dataset = dataset.map(
        lambda p, v: (preprocess_image(p, image_size), v),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if augment:
        # Add augmentation
        def augment_fn(image, value):
            image = tf.image.random_brightness(image, 0.1)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, value
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(input_shape: Tuple[int, int] = (224, 224, 3)) -> keras.Model:
    """Build MobileNetV2-based regression model."""
    # Use MobileNetV2 backbone
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=1.0,
    )
    
    # Freeze backbone initially
    backbone.trainable = False
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    x = backbone(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)  # Linear output for regression
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae'],
    )
    
    return model, backbone


def train_simple(
    train_manifest: Path,
    val_manifest: Path | None = None,
    epochs: int = 50,
    batch_size: int = 16,
    output_dir: Path = ML_ROOT / "artifacts" / "training" / "simple_supervised",
):
    """Train model using simple supervised learning on image paths and values."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"[TRAIN] Loading training data from {train_manifest}")
    train_samples = load_manifest(train_manifest)
    print(f"[TRAIN] Loaded {len(train_samples)} training samples")
    
    # Split if no validation manifest
    if val_manifest is None:
        train_samples, val_samples = train_test_split(
            train_samples, test_size=0.15, random_state=42
        )
        print(f"[TRAIN] Split: {len(train_samples)} train, {len(val_samples)} val")
    else:
        print(f"[TRAIN] Loading validation data from {val_manifest}")
        val_samples = load_manifest(val_manifest)
        print(f"[TRAIN] Loaded {len(val_samples)} validation samples")
    
    # Create datasets
    train_ds = create_dataset(train_samples, batch_size, augment=True)
    val_ds = create_dataset(val_samples, batch_size, augment=False)
    
    # Build model
    print("[TRAIN] Building model...")
    model, backbone = build_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "model_best.keras"),
            monitor='val_loss',
            save_best_only=True,
        ),
    ]
    
    # Phase 1: Train head only
    print("[TRAIN] Phase 1: Training head (backbone frozen)...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=min(10, epochs),
        callbacks=callbacks,
        verbose=1,
    )
    
    # Phase 2: Fine-tune entire model
    print("[TRAIN] Phase 2: Fine-tuning entire model...")
    backbone.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae'],
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=len(history1.history['loss']),
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save final model
    model.save(output_dir / "model_final.keras")
    print(f"[TRAIN] Model saved to {output_dir}")
    
    # Evaluate
    print("[TRAIN] Evaluating on validation set...")
    val_loss, val_mae = model.evaluate(val_ds, verbose=0)
    
    # Convert MAE back to temperature scale
    min_val, max_val = -30.0, 50.0
    val_mae_celsius = val_mae * (max_val - min_val) / 2
    print(f"[TRAIN] Validation MAE: {val_mae_celsius:.2f}°C")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train gauge reader with simple supervised learning")
    parser.add_argument("--train-manifest", type=Path, required=True, help="CSV with image_path,value columns")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Optional validation CSV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=ML_ROOT / "artifacts" / "training" / "simple_supervised")
    args = parser.parse_args()
    
    train_simple(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
