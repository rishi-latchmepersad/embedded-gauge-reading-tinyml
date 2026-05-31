#!/usr/bin/env python3
"""Train with EfficientNet-B0 from scratch + heavy board weighting."""

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
OUTPUT_DIR = Path("/tmp/coord_efficientnet_scratch")
INPUT_SIZE = 224


def load_data(crops_dir: Path):
    """Load crops and coordinates."""
    meta = json.load(open(crops_dir / "metadata.json"))
    train_meta = [m for m in meta if m["split"] == "train"]
    val_meta = [m for m in meta if m["split"] == "val"]
    
    def load_split(samples):
        imgs = []
        coords = []
        is_board = []
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
            is_board.append('capture_' in s["image_path"])
        return np.array(imgs, dtype=np.float32), np.array(coords, dtype=np.float32), np.array(is_board)
    
    train_imgs, train_coords, train_board = load_split(train_meta)
    val_imgs, val_coords, val_board = load_split(val_meta)
    print(f"Loaded {len(train_imgs)} train ({train_board.sum()} board), {len(val_imgs)} val ({val_board.sum()} board)")
    return train_imgs, train_coords, train_board, val_imgs, val_coords


def build_efficientnet():
    """Build EfficientNet-B0 from scratch (no pretrained weights)."""
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    
    # EfficientNet-B0 without pretrained weights
    backbone = keras.applications.EfficientNetB0(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_top=False,
        weights=None,  # No pretrained weights
        pooling='avg',
    )
    
    x = backbone(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model


def train(board_weight=5.0, epochs=150, batch_size=16, lr=1e-3):
    """Train with weighted loss for board captures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_imgs, train_coords, train_board, val_imgs, val_coords, val_board = load_data(CROPS_DIR)
    
    # Create sample weights - board samples weighted higher
    sample_weights = np.where(train_board, board_weight, 1.0)
    print(f"Sample weights: board={board_weight}, phone=1.0")
    
    model = build_efficientnet()
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
            monitor="val_loss", patience=30, restore_best_weights=True
        ),
    ]
    
    history = model.fit(
        train_imgs, train_coords,
        sample_weight=sample_weights,
        validation_data=(val_imgs, val_coords),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    model.save(str(OUTPUT_DIR / "final.keras"))
    
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    
    val_loss, val_mae = model.evaluate(val_imgs, val_coords, verbose=0)
    print(f"\nFinal val MSE: {val_loss:.6f}, val MAE: {val_mae:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-weight", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train(
        board_weight=args.board_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
