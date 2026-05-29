#!/usr/bin/env python3
"""Direct temperature regression - simpler objective."""

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
OUTPUT_DIR = Path("/tmp/direct_temp_regression")
INPUT_SIZE = 224

# Temperature normalization
MIN_TEMP = -30.0
MAX_TEMP = 50.0
TEMP_RANGE = MAX_TEMP - MIN_TEMP


def load_data(crops_dir: Path):
    """Load crops and temperatures."""
    meta = json.load(open(crops_dir / "metadata.json"))
    train_meta = [m for m in meta if m["split"] == "train"]
    val_meta = [m for m in meta if m["split"] == "val"]
    
    def load_split(samples):
        imgs = []
        temps = []
        is_board = []
        for s in samples:
            img = Image.open(crops_dir / s["image_path"]).convert("RGB")
            img_arr = np.asarray(img, dtype=np.float32) / 255.0
            if img_arr.shape[-1] == 1:
                img_arr = np.repeat(img_arr, 3, -1)
            imgs.append(img_arr)
            # Normalize temperature to [0, 1]
            temp_norm = (s["temperature_c"] - MIN_TEMP) / TEMP_RANGE
            temps.append(temp_norm)
            is_board.append('capture_' in s['image_path'])
        return np.array(imgs, dtype=np.float32), np.array(temps, dtype=np.float32), np.array(is_board)
    
    train_imgs, train_temps, train_board = load_split(train_meta)
    val_imgs, val_temps, val_board = load_split(val_meta)
    print(f"Loaded {len(train_imgs)} train ({train_board.sum()} board), {len(val_imgs)} val ({val_board.sum()} board)")
    return train_imgs, train_temps, train_board, val_imgs, val_temps, val_board


def build_model():
    """Build CNN for direct temperature regression."""
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    
    # Deeper CNN
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Normalized temperature
    
    model = keras.Model(inputs, outputs)
    return model


def train(epochs=200, batch_size=16, lr=1e-3, board_weight=10.0):
    """Train direct temperature regression."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_imgs, train_temps, train_board, val_imgs, val_temps, val_board = load_data(CROPS_DIR)
    
    # Heavy board weighting
    sample_weights = np.where(train_board, board_weight, 1.0)
    print(f"Sample weights: board={board_weight}, phone=1.0")
    
    model = build_model()
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
            monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=40, restore_best_weights=True
        ),
    ]
    
    history = model.fit(
        train_imgs, train_temps,
        sample_weight=sample_weights,
        validation_data=(val_imgs, val_temps),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    model.save(str(OUTPUT_DIR / "final.keras"))
    
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    
    # Evaluate
    val_loss, val_mae_norm = model.evaluate(val_imgs, val_temps, verbose=0)
    val_mae_temp = val_mae_norm * TEMP_RANGE
    print(f"\nFinal val MAE: {val_mae_temp:.2f}°C")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--board-weight", type=float, default=10.0)
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        board_weight=args.board_weight,
    )
