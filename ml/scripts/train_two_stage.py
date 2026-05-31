#!/usr/bin/env python3
"""Two-stage training: train on all, then fine-tune on board only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops"
OUTPUT_DIR = Path("/tmp/coord_two_stage")
INPUT_SIZE = 224


def load_data(crops_dir: Path):
    """Load crops and coordinates."""
    meta = json.load(open(crops_dir / "metadata.json"))
    train_meta = [m for m in meta if m["split"] == "train"]
    val_meta = [m for m in meta if m["split"] == "val"]
    
    def load_split(samples):
        imgs = []
        coords = []
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
    
    # Separate board captures
    train_board_idx = [i for i, m in enumerate(train_meta) if 'capture_' in m['image_path']]
    train_board_imgs = train_imgs[train_board_idx]
    train_board_coords = train_coords[train_board_idx]
    
    print(f"Loaded {len(train_imgs)} train, {len(train_board_imgs)} board only, {len(val_imgs)} val")
    return train_imgs, train_coords, train_board_imgs, train_board_coords, val_imgs, val_coords


def build_model():
    """Build simple CNN."""
    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    
    x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(4, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model


def train_two_stage():
    """Stage 1: Train on all data. Stage 2: Fine-tune on board only."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_imgs, train_coords, train_board_imgs, train_board_coords, val_imgs, val_coords = load_data(CROPS_DIR)
    
    # Stage 1: Train on all data
    print("\n" + "="*80)
    print("STAGE 1: Training on all data")
    print("="*80)
    
    model = build_model()
    print(f"Model params: {model.count_params():,}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    history1 = model.fit(
        train_imgs, train_coords,
        validation_data=(val_imgs, val_coords),
        epochs=50,
        batch_size=16,
        verbose=1,
    )
    
    # Stage 2: Fine-tune on board captures only with lower LR
    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning on board captures only")
    print("="*80)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True
        ),
    ]
    
    # Oversample board captures
    n_repeat = len(train_imgs) // len(train_board_imgs)
    board_imgs_repeated = np.repeat(train_board_imgs, n_repeat, axis=0)
    board_coords_repeated = np.repeat(train_board_coords, n_repeat, axis=0)
    
    history2 = model.fit(
        board_imgs_repeated, board_coords_repeated,
        validation_data=(val_imgs, val_coords),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )
    
    model.save(str(OUTPUT_DIR / "final.keras"))
    
    # Save histories
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump({
            'stage1': {k: [float(v) for v in vals] for k, vals in history1.history.items()},
            'stage2': {k: [float(v) for v in vals] for k, vals in history2.history.items()},
        }, f, indent=2)
    
    val_loss, val_mae = model.evaluate(val_imgs, val_coords, verbose=0)
    print(f"\nFinal val MSE: {val_loss:.6f}, val MAE: {val_mae:.6f}")


if __name__ == "__main__":
    train_two_stage()
