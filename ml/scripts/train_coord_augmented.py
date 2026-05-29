#!/usr/bin/env python3
"""Train coordinate regression with aggressive augmentation for board captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops"
OUTPUT_DIR = Path("/tmp/coord_regression_augmented")
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
    print(f"Loaded {len(train_imgs)} train ({train_board.sum()} board), {len(val_imgs)} val ({val_board.sum()} board) samples")
    return train_imgs, train_coords, train_board, val_imgs, val_coords, val_board


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply aggressive augmentation to simulate board capture characteristics."""
    # Convert to PIL for augmentation
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    
    # Random brightness (board captures have different lighting)
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.7, 1.3)
        img_pil = ImageEnhance.Brightness(img_pil).enhance(factor)
    
    # Random contrast (board captures have different contrast)
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        img_pil = ImageEnhance.Contrast(img_pil).enhance(factor)
    
    # Random sharpness (board captures may be less sharp)
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.5, 1.5)
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(factor)
    
    # Add Gaussian noise
    if np.random.random() > 0.5:
        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        noise = np.random.normal(0, 0.05, img_arr.shape).astype(np.float32)
        img_arr = np.clip(img_arr + noise, 0, 1)
        img_pil = Image.fromarray((img_arr * 255).astype(np.uint8))
    
    # Random grayscale (some board captures are grayscale)
    if np.random.random() > 0.8:
        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        gray = 0.299 * img_arr[..., 0] + 0.587 * img_arr[..., 1] + 0.114 * img_arr[..., 2]
        img_arr = np.stack([gray] * 3, -1)
        img_pil = Image.fromarray((img_arr * 255).astype(np.uint8))
    
    return np.asarray(img_pil, dtype=np.float32) / 255.0


def build_cnn_model(input_shape=(224, 224, 3)):
    """Build simple CNN without pretrained weights."""
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model


def train_with_augmentation(epochs=150, batch_size=16, lr=1e-3):
    """Train with on-the-fly augmentation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_imgs, train_coords, train_board, val_imgs, val_coords, val_board = load_data(CROPS_DIR)
    
    # Create augmented training data
    print("Generating augmented training data...")
    aug_train_imgs = []
    aug_train_coords = []
    
    # Original data
    aug_train_imgs.append(train_imgs)
    aug_train_coords.append(train_coords)
    
    # 2x augmented version of each sample
    for _ in range(2):
        aug_imgs = np.array([augment_image(img) for img in train_imgs])
        aug_train_imgs.append(aug_imgs)
        aug_train_coords.append(train_coords)
    
    # Oversample board captures 3x more
    board_indices = np.where(train_board)[0]
    print(f"Oversampling {len(board_indices)} board captures 3x...")
    for _ in range(3):
        board_imgs = np.array([augment_image(train_imgs[i]) for i in board_indices])
        board_coords = train_coords[board_indices]
        aug_train_imgs.append(board_imgs)
        aug_train_coords.append(board_coords)
    
    aug_train_imgs = np.concatenate(aug_train_imgs, axis=0)
    aug_train_coords = np.concatenate(aug_train_coords, axis=0)
    print(f"Augmented training set: {len(aug_train_imgs)} samples")
    
    model = build_cnn_model((INPUT_SIZE, INPUT_SIZE, 3))
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
        aug_train_imgs, aug_train_coords,
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
    
    # Evaluate on board captures separately
    if val_board.sum() > 0:
        board_val_imgs = val_imgs[val_board]
        board_val_coords = val_coords[val_board]
        board_loss, board_mae = model.evaluate(board_val_imgs, board_val_coords, verbose=0)
        print(f"Board captures val MSE: {board_loss:.6f}, val MAE: {board_mae:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_with_augmentation(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
