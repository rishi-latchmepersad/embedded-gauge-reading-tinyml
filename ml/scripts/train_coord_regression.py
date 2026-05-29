#!/usr/bin/env python3
"""Train simple coordinate regression model for gauge reading."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops"
OUTPUT_DIR = Path("/tmp/coord_regression")
INPUT_SIZE = 224


def load_data(crops_dir: Path):
    """Load crops and coordinates."""
    meta = json.load(open(crops_dir / "metadata.json"))
    train_meta = [m for m in meta if m["split"] == "train"]
    val_meta = [m for m in meta if m["split"] == "val"]
    
    def load_split(samples):
        imgs = []
        coords = []  # [center_x, center_y, tip_x, tip_y] normalized
        for s in samples:
            img = Image.open(crops_dir / s["image_path"]).convert("RGB")
            img_arr = np.asarray(img, dtype=np.float32) / 255.0
            if img_arr.shape[-1] == 1:
                img_arr = np.repeat(img_arr, 3, -1)
            imgs.append(img_arr)
            coords.append([
                s["center_x_norm"], s["center_y_norm"],
                s["tip_x_norm"], s["tip_y_norm"]
            ])
        return np.array(imgs, dtype=np.float32), np.array(coords, dtype=np.float32)
    
    train_imgs, train_coords = load_split(train_meta)
    val_imgs, val_coords = load_split(val_meta)
    print(f"Loaded {len(train_imgs)} train, {len(val_imgs)} val samples")
    return train_imgs, train_coords, val_imgs, val_coords


def build_tiny_model(input_shape=(224, 224, 3)):
    """Build a tiny CNN for coordinate regression."""
    inputs = keras.Input(shape=input_shape)
    
    # Simple CNN - no pretrained backbone
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)  # 224x224
    x = layers.MaxPooling2D(2)(x)  # 112x112
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)  # 56x56
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)  # 28x28
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)  # 14x14
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)  # 7x7
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)  # 4 normalized coordinates
    
    model = keras.Model(inputs, outputs)
    return model


def train(epochs=100, batch_size=16, lr=1e-3):
    """Train the coordinate regression model."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_imgs, train_coords, val_imgs, val_coords = load_data(CROPS_DIR)
    
    model = build_tiny_model((INPUT_SIZE, INPUT_SIZE, 3))
    print(f"Model params: {model.count_params():,}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        ),
    ]
    
    history = model.fit(
        train_imgs, train_coords,
        validation_data=(val_imgs, val_coords),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    model.save(str(OUTPUT_DIR / "final.keras"))
    
    # Save history
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    
    # Evaluate
    val_loss, val_mae = model.evaluate(val_imgs, val_coords, verbose=0)
    print(f"\nFinal val MSE: {val_loss:.6f}, val MAE: {val_mae:.6f}")
    
    # Convert coordinate MAE to angle error
    # Each coordinate error of 0.01 = 1.12 pixels on 112x112
    # Angle error depends on position, but roughly 1 pixel ~ 1-2 degrees
    print(f"Approx angle error: {val_mae * 112 * 1.5:.2f} degrees")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
