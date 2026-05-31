#!/usr/bin/env python3
"""Train MobileNetV2 coordinate regression with transfer learning."""

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

from embedded_gauge_reading_tinyml.models_coord import (
    build_mobilenetv2_coord_regression,
    build_two_phase_coord_model,
)

CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops"
OUTPUT_DIR = Path("/tmp/coord_regression_mobilenet")
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
    print(f"Loaded {len(train_imgs)} train, {len(val_imgs)} val samples")
    return train_imgs, train_coords, val_imgs, val_coords


def train_two_phase(
    epochs_phase1: int = 30,
    epochs_phase2: int = 50,
    batch_size: int = 16,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-5,
):
    """Two-phase transfer learning training."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_imgs, train_coords, val_imgs, val_coords = load_data(CROPS_DIR)
    
    # Phase 1: Train head with frozen backbone
    print("\n" + "=" * 80)
    print("PHASE 1: Training head with frozen backbone")
    print("=" * 80)
    
    model = build_two_phase_coord_model((INPUT_SIZE, INPUT_SIZE, 3))
    print(f"Model params: {model.count_params():,}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_phase1),
        loss='mse',
        metrics=['mae']
    )
    
    callbacks_phase1 = [
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best_phase1.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]
    
    history_phase1 = model.fit(
        train_imgs, train_coords,
        validation_data=(val_imgs, val_coords),
        epochs=epochs_phase1,
        batch_size=batch_size,
        callbacks=callbacks_phase1,
        verbose=1,
    )
    
    # Load best weights from phase 1
    model = keras.models.load_model(str(OUTPUT_DIR / "best_phase1.keras"))
    
    # Phase 2: Unfreeze backbone and fine-tune
    print("\n" + "=" * 80)
    print("PHASE 2: Fine-tuning backbone")
    print("=" * 80)
    
    # Unfreeze backbone
    model.layers[1].trainable = True
    
    # Recompile with lower LR
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_phase2),
        loss='mse',
        metrics=['mae']
    )
    
    callbacks_phase2 = [
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best_phase2.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
    ]
    
    history_phase2 = model.fit(
        train_imgs, train_coords,
        validation_data=(val_imgs, val_coords),
        epochs=epochs_phase2,
        batch_size=batch_size,
        callbacks=callbacks_phase2,
        verbose=1,
    )
    
    # Save final model
    model.save(str(OUTPUT_DIR / "final.keras"))
    
    # Save histories
    with open(OUTPUT_DIR / "history_phase1.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history_phase1.history.items()}, f, indent=2)
    
    with open(OUTPUT_DIR / "history_phase2.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history_phase2.history.items()}, f, indent=2)
    
    # Evaluate
    val_loss, val_mae = model.evaluate(val_imgs, val_coords, verbose=0)
    print(f"\nFinal val MSE: {val_loss:.6f}, val MAE: {val_mae:.6f}")
    print(f"Approx angle error: {val_mae * 224 * 1.5:.2f} degrees")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs1", type=int, default=30, help="Phase 1 epochs")
    parser.add_argument("--epochs2", type=int, default=50, help="Phase 2 epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr1", type=float, default=1e-3, help="Phase 1 LR")
    parser.add_argument("--lr2", type=float, default=1e-5, help="Phase 2 LR")
    args = parser.parse_args()
    
    train_two_phase(
        epochs_phase1=args.epochs1,
        epochs_phase2=args.epochs2,
        batch_size=args.batch_size,
        lr_phase1=args.lr1,
        lr_phase2=args.lr2,
    )
