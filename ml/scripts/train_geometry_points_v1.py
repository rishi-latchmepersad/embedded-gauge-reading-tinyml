#!/usr/bin/env python3
"""
Training script for geometry points prediction model (v1).

This script trains a MobileNetV2-based model to predict normalized center and tip
coordinates from cropped gauge images.

Usage:
    poetry run python ml/scripts/train_geometry_points_v1.py

Output:
    ml/artifacts/training/geometry_points_v1/
        - model.keras (trained model)
        - history.csv (training history)
        - config.json (training configuration)
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Set TF environment variables early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image

# Import TensorFlow after setting env vars
import tensorflow as tf
from tensorflow import keras

from embedded_gauge_reading_tinyml.models_geometry import (
    build_mobilenetv2_geometry_points_v1,
    compile_geometry_model,
)
from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    SourceGeometryExample,
    create_jittered_crop,
    generate_jitter_params,
    JitterParams,
)


def _load_yuv422_as_rgb(image_path: Path, source_width: int, source_height: int) -> np.ndarray:
    """Decode a packed YUV422 board capture into a simple RGB array.

    We only need stable geometry training input here, so we keep the decoder
    intentionally small and convert the luma plane to three identical channels.
    """

    raw = image_path.read_bytes()
    expected = source_height * (source_width // 2) * 4
    if len(raw) < expected:
        raise ValueError(f"{image_path} is too small for {source_width}x{source_height} YUV422")
    yuyv = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(source_height, source_width // 2, 4)
    luma = np.empty((source_height, source_width), dtype=np.uint8)
    luma[:, 0::2] = yuyv[:, :, 0]
    luma[:, 1::2] = yuyv[:, :, 2]
    return np.repeat(luma[:, :, None], 3, axis=2)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    epochs: int = 80
    learning_rate: float = 1e-4
    input_size: int = 224
    dense_units: int = 96
    dropout_rate: float = 0.15
    alpha: float = 0.35
    backbone_frozen: bool = True
    seed: int = 42
    early_stopping_patience: int = 15


def load_clean_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load v2_clean manifest and filter to clean rows only."""
    rows = []
    
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            if row_dict.get("quality_flag", "clean") != "clean":
                continue
            rows.append(row_dict)
    
    return rows


def load_sample(
    row: Dict[str, Any],
    base_path: Path,
    input_size: int = 224,
    training: bool = True,
    rng: Optional[random.Random] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load one training sample from a manifest row.
    
    Args:
        row: Manifest row dictionary
        base_path: Base path for image loading
        input_size: Target input size
        training: Whether to apply jitter
        rng: Random number generator for jitter
        
    Returns:
        Tuple of (image_array, label_vector) or None
    """
    image_path = base_path / row["image_path"]
    
    if not image_path.exists():
        return None
    
    # Get crop coordinates
    x1 = float(row["loose_crop_x1"])
    y1 = float(row["loose_crop_y1"])
    x2 = float(row["loose_crop_x2"])
    y2 = float(row["loose_crop_y2"])
    
    # Get source dimensions
    source_w = int(row["source_width"])
    source_h = int(row["source_height"])

    try:
        if image_path.suffix.lower() == ".yuv422":
            image_array = _load_yuv422_as_rgb(image_path, source_w, source_h)
        else:
            image = Image.open(image_path).convert("RGB")
            image_array = np.asarray(image, dtype=np.uint8)
    except Exception:
        return None
    
    # Apply jitter for training
    if training and rng is not None:
        jitter: JitterParams = generate_jitter_params(rng)
        shift_x = jitter.shift_x
        shift_y = jitter.shift_y
        scale = jitter.scale
        aspect = jitter.aspect
        
        # Apply shift
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        x1_new = x1 + shift_x
        y1_new = y1 + shift_y
        x2_new = x1_new + crop_w * scale
        y2_new = y1_new + crop_h * aspect
        
        # Check bounds
        if x1_new < 0 or y1_new < 0 or x2_new > source_w or y2_new > source_h:
            # Fall back to identity crop
            x1_new, y1_new, x2_new, y2_new = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1_new, y1_new, x2_new, y2_new
    
    # Extract crop
    crop_box = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    crop = Image.fromarray(image_array).crop(crop_box)
    
    # Resize to input size
    crop_resized = crop.resize((input_size, input_size), Image.LANCZOS)
    
    # Convert to numpy array and normalize
    image_array = np.array(crop_resized, dtype=np.float32) / 255.0
    
    # Compute normalized coordinates
    crop_w = x2 - x1
    crop_h = y2 - y1
    
    if crop_w <= 0 or crop_h <= 0:
        return None
    
    center_x_norm = (float(row["center_x_source"]) - x1) / crop_w
    center_y_norm = (float(row["center_y_source"]) - y1) / crop_h
    tip_x_norm = (float(row["tip_x_source"]) - x1) / crop_w
    tip_y_norm = (float(row["tip_y_source"]) - y1) / crop_h
    
    # Validate coordinates
    if not (0 <= center_x_norm <= 1 and 0 <= center_y_norm <= 1 and
            0 <= tip_x_norm <= 1 and 0 <= tip_y_norm <= 1):
        return None
    
    # Create label vector
    label_vector = np.array([
        center_x_norm,
        center_y_norm,
        tip_x_norm,
        tip_y_norm,
        1.0,  # confidence
    ], dtype=np.float32)
    
    return image_array, label_vector


def create_dataset(
    rows: List[Dict[str, Any]],
    base_path: Path,
    config: TrainingConfig,
    training: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dataset from manifest rows.
    
    Args:
        rows: List of manifest row dictionaries
        base_path: Base path for image loading
        config: Training configuration
        training: Whether to apply jitter augmentation
        
    Returns:
        Tuple of (X, y) numpy arrays
    """
    images = []
    labels = []
    
    rng = random.Random(config.seed)
    crops_per_example = 3 if training else 1
    
    for i, row in enumerate(rows):
        example_rng = random.Random(config.seed + i)
        
        for _ in range(crops_per_example):
            sample = load_sample(row, base_path, config.input_size, training, example_rng)
            
            if sample is None:
                continue
            
            image_array, label_vector = sample
            images.append(image_array)
            labels.append(label_vector)
    
    if len(images) == 0:
        return np.array([]), np.array([])
    
    return np.array(images), np.array(labels)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train geometry points prediction model")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="CSV manifest path. Defaults to ml/data/geometry_reader_manifest_v2_clean.csv.",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Geometry Points Training (v1)")
    print("=" * 80)
    
    # Configuration
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
    
    # Paths
    base_path = Path(__file__).parent.parent.parent
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
    else:
        manifest_path = base_path / "ml" / "data" / "geometry_reader_manifest_v2_clean.csv"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_path / "ml" / "artifacts" / "training" / "geometry_points_v1"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nManifest: {manifest_path}")
    print(f"Output dir: {output_dir}")
    
    # Load clean manifest
    print("\nLoading clean manifest...")
    rows = load_clean_manifest(manifest_path)
    print(f"Loaded {len(rows)} clean rows")
    
    # Split by train/val/test
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_rows)}")
    print(f"  Val: {len(val_rows)}")
    print(f"  Test: {len(test_rows)}")
    
    # Create datasets
    print("\nCreating training dataset...")
    X_train, y_train = create_dataset(train_rows, base_path, config, training=True)
    print(f"Training samples: {len(X_train)}")
    
    print("\nCreating validation dataset...")
    X_val, y_val = create_dataset(val_rows, base_path, config, training=False)
    print(f"Validation samples: {len(X_val)}")
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("ERROR: No training or validation data loaded. Check image paths.")
        sys.exit(1)
    
    print(f"\nTraining data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    
    # Build model
    print("\nBuilding model...")
    model = build_mobilenetv2_geometry_points_v1(
        input_shape=(config.input_size, config.input_size, 3),
        alpha=config.alpha,
        backbone_frozen=config.backbone_frozen,
        dense_units=config.dense_units,
        dropout_rate=config.dropout_rate,
    )
    
    print(f"\nModel summary:")
    print(f"  Name: {model.name}")
    print(f"  Total parameters: {model.count_params():,}")
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Compile model
    print("\nCompiling model...")
    compile_geometry_model(model, learning_rate=config.learning_rate)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save model
    print("\nSaving model...")
    model.save(output_dir / "model.keras")
    print(f"Saved model to {output_dir / 'model.keras'}")
    
    # Save config
    print("\nSaving configuration...")
    config_dict = {
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "input_size": config.input_size,
        "dense_units": config.dense_units,
        "dropout_rate": config.dropout_rate,
        "alpha": config.alpha,
        "backbone_frozen": config.backbone_frozen,
        "seed": config.seed,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "test_examples": len(test_rows),
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Saved config to {output_dir / 'config.json'}")
    
    # Save history as CSV
    print("\nSaving history...")
    history_path = output_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        if history.history:
            fieldnames = list(history.history.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(history.history["loss"])):
                row = {k: v[i] for k, v in history.history.items()}
                writer.writerow(row)
    
    print(f"Saved history to {output_dir / 'history.csv'}")
    
    # Print final metrics
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    
    if history.history.get("val_loss"):
        best_epoch = np.argmin(history.history["val_loss"])
        print(f"Best epoch: {best_epoch + 1}")
        print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    
    print("\nFinal training metrics:")
    for metric_name, values in history.history.items():
        if values:
            print(f"  {metric_name}: {values[-1]:.6f}")
    
    return model, history


if __name__ == "__main__":
    main()
