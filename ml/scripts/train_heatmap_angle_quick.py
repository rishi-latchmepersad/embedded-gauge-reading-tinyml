"""Quick training script for heatmap angle CNN using board captures only.

This is a simplified training script that uses only the labeled board captures
which are all accessible in WSL. Use this for quick iteration.

Usage:
    poetry run python scripts/train_heatmap_angle_quick.py --epochs 20 --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from embedded_gauge_reading_tinyml.models_geometry import build_heatmap_angle_model

HEATMAP_SIZE = 112
IMAGE_SIZE = 224
GAUSSIAN_SIGMA = 8.0


def build_gaussian_heatmap(
    center_x: float, center_y: float, heatmap_size: int = HEATMAP_SIZE, sigma: float = GAUSSIAN_SIGMA
) -> np.ndarray:
    """Build 2D Gaussian heatmap."""
    y, x = np.ogrid[:heatmap_size, :heatmap_size]
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    return heatmap.astype(np.float32)


def load_board_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load board capture samples."""
    samples = []
    if not manifest_path.exists():
        return samples
    
    # Repo root is 2 levels up from scripts/
    repo_root = manifest_path.parent.parent.parent
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("image_path") or not row.get("center_x"):
                continue
            # Resolve path relative to repo root
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
                })
            except (ValueError, KeyError):
                continue
    return samples


def load_and_preprocess(sample: dict) -> tuple:
    """Load and preprocess one sample."""
    # Load image
    img = tf.io.read_file(sample["image_path"])
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE)
    
    # Transform coordinates from source image space to 224x224, then to 112x112 heatmap space
    src_w = sample["source_width"]
    src_h = sample["source_height"]
    
    # First normalize to [0, 1] in source space, then scale to 224x224
    center_x = (sample["center_x"] / src_w) * IMAGE_SIZE
    center_y = (sample["center_y"] / src_h) * IMAGE_SIZE
    tip_x = (sample["tip_x"] / src_w) * IMAGE_SIZE
    tip_y = (sample["tip_y"] / src_h) * IMAGE_SIZE
    
    # Scale to heatmap coordinates (112x112)
    hm_scale = HEATMAP_SIZE / IMAGE_SIZE
    center_x_hm = center_x * hm_scale
    center_y_hm = center_y * hm_scale
    tip_x_hm = tip_x * hm_scale
    tip_y_hm = tip_y * hm_scale
    
    # Build heatmaps
    center_hm = build_gaussian_heatmap(center_x_hm, center_y_hm)
    tip_hm = build_gaussian_heatmap(tip_x_hm, tip_y_hm)
    
    # Confidence
    conf = np.array([1.0], dtype=np.float32)
    
    return img, (center_hm, tip_hm, conf)


def create_dataset(samples: list[dict], batch_size: int = 8, shuffle: bool = True) -> tf.data.Dataset:
    """Create dataset from samples."""
    def gen():
        for s in samples:
            try:
                yield load_and_preprocess(s)
            except Exception as e:
                print(f"Error: {e}")
    
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
        ds = ds.shuffle(len(samples)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="/tmp/heatmap_angle_quick")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    samples = load_board_manifest(data_dir / "board_captures_labeled_v2.csv")
    print(f"Loaded {len(samples)} board captures")
    
    if len(samples) < 10:
        print("Not enough samples!")
        return
    
    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    n_val = max(1, int(len(samples) * 0.15))
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Create datasets
    train_ds = create_dataset(train_samples, args.batch_size, shuffle=True)
    val_ds = create_dataset(val_samples, args.batch_size, shuffle=False)
    
    # Build model
    model = build_heatmap_angle_model(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        alpha=0.35,
        backbone_frozen=False,
        heatmap_size=HEATMAP_SIZE,
    )
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss={
            "center_heatmap": "mse",
            "tip_heatmap": keras.losses.MSE,
            "confidence": keras.losses.binary_crossentropy,
        },
        loss_weights={
            "center_heatmap": 1.0,
            "tip_heatmap": 2.0,
            "confidence": 0.1,
        },
    )
    
    # Train
    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(output_dir / "best.keras"),
                save_best_only=True,
                monitor="val_loss",
                mode="min",
            ),
            keras.callbacks.CSVLogger(str(output_dir / "training.log")),
        ],
    )
    
    # Save
    model.save(output_dir / "final.keras")
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
