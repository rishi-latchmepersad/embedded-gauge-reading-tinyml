#!/usr/bin/env python3
"""
Training script for heatmap-based geometry prediction model (v1).

This script trains a MobileNetV2-based model to predict center and tip
heatmaps from cropped gauge images. Coordinates are extracted via
softargmax during inference.

Usage:
    poetry run python ml/scripts/train_geometry_heatmap_v1.py

Output:
    ml/artifacts/training/geometry_heatmap_v1/
        - model.keras
        - history.csv
        - config.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    load_geometry_manifest,
    SourceGeometryExample,
    create_jittered_crop,
    generate_jitter_params,
    JitterParams,
)
from embedded_gauge_reading_tinyml.models_geometry import (
    build_mobilenetv2_geometry_heatmap_v1
)
from embedded_gauge_reading_tinyml.heatmap_utils import (
    generate_center_tip_heatmaps,
    HeatmapConfig,
)


def create_heatmap_dataset(
    examples: List[SourceGeometryExample],
    base_path: Path,
    input_size: int = 224,
    heatmap_size: int = 56,
    sigma_pixels: float = 2.5,
    max_samples: int | None = None,
    seed: int = 42,
) -> Tuple[np.array, List[Dict]]:
    """Create heatmap dataset from examples."""
    rng = np.random.default_rng(seed)
    
    X = []
    y = []
    
    heatmap_config = HeatmapConfig(
        heatmap_height=heatmap_size,
        heatmap_width=heatmap_size,
        sigma_pixels=sigma_pixels,
    )
    
    for i, ex in enumerate(examples):
        if max_samples and i >= max_samples:
            break
        
        # Generate jitter params
        jitter = generate_jitter_params(rng=rng)
        
        # Create jittered crop
        crop = create_jittered_crop(ex, jitter)
        
        if not crop.accepted:
            continue
        
        # Load image
        image_path = base_path / crop.source_image_path
        if not image_path.exists():
            continue
            
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
            
        # Extract crop
        crop_box = (crop.crop_x1, crop.crop_y1, crop.crop_x2, crop.crop_y2)
        crop_image = image.crop(crop_box)
        crop_image = crop_image.resize((input_size, input_size))
        
        image_array = np.array(crop_image, dtype=np.float32) / 255.0
        
        # Generate heatmaps
        center_hm, tip_hm = generate_center_tip_heatmaps(
            center_x_norm=crop.center_x_normalized,
            center_y_norm=crop.center_y_normalized,
            tip_x_norm=crop.tip_x_normalized,
            tip_y_norm=crop.tip_y_normalized,
            config=heatmap_config,
        )
        
        X.append(image_array)
        y.append({
            'center_heatmap': center_hm,
            'tip_heatmap': tip_hm,
            'confidence': 1.0,
        })
    
    # Convert y list to dict of arrays
    y_dict = {
        'center_heatmap': np.array([item['center_heatmap'] for item in y]),
        'tip_heatmap': np.array([item['tip_heatmap'] for item in y]),
        'confidence': np.array([item['confidence'] for item in y]),
    }
    return np.array(X), y_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/geometry_reader_manifest_v2_clean.csv')
    parser.add_argument('--output-dir', default='artifacts/training/geometry_heatmap_v1')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--sigma-pixels', type=float, default=2.5)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Manifest: {args.manifest}")
    print(f"Output dir: {output_dir}")
    
    # Load manifest
    print("\nLoading clean manifest...")
    all_examples = load_geometry_manifest(args.manifest)
    clean_examples = [ex for ex in all_examples if ex.quality_flag == 'clean']
    print(f"Loaded {len(clean_examples)} clean rows")
    
    # Split examples
    train_examples = [ex for ex in clean_examples if ex.split == 'train']
    val_examples = [ex for ex in clean_examples if ex.split == 'val']
    test_examples = [ex for ex in clean_examples if ex.split == 'test']
    
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")
    print(f"  Test: {len(test_examples)}")
    
    base_path = Path(__file__).parent.parent.parent
    
    # Create datasets
    print("\nCreating training dataset...")
    X_train, y_train = create_heatmap_dataset(
        train_examples, base_path, sigma_pixels=args.sigma_pixels,
        seed=42,
    )
    print(f"Training samples: {len(X_train)}")
    
    print("\nCreating validation dataset...")
    X_val, y_val = create_heatmap_dataset(
        val_examples, base_path, sigma_pixels=args.sigma_pixels,
        seed=42,
    )
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_mobilenetv2_geometry_heatmap_v1(learning_rate=args.learning_rate)
    print(f"Model name: {model.name}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )
    
    reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(output_dir / 'checkpoints/epoch-{epoch}.keras'),
        save_best_only=False,
    )
    
    # Train
    print("\\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\\n")
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop, reduce_lr_on_plateau, model_checkpoint],
    )
    
    # Save model
    print("\nSaving model...")
    model.save(str(output_dir / 'model.keras'))
    print(f"Saved model to {output_dir / 'model.keras'}")
    
    # Save config
    print("\nSaving configuration...")
    config = {
        'manifest': args.manifest,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'sigma_pixels': args.sigma_pixels,
        'input_size': 224,
        'heatmap_size': 56,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {output_dir / 'config.json'}")
    
    # Save history
    print("\nSaving history...")
    history_data = history.history
    with open(output_dir / "history.csv", "w") as f:
        f.write("epoch,loss,center_heatmap_loss,tip_heatmap_loss,confidence_loss,val_loss,val_center_heatmap_mae,val_tip_heatmap_mae,lr\n")
        num_epochs = len(history_data.get("loss", []))
        for j in range(num_epochs):
            loss = history_data.get("loss", [0]*num_epochs)[j]
            chm_loss = history_data.get("center_heatmap_loss", [0]*num_epochs)[j]
            thm_loss = history_data.get("tip_heatmap_loss", [0]*num_epochs)[j]
            conf_loss = history_data.get("confidence_loss", [0]*num_epochs)[j]
            val_loss = history_data.get("val_loss", [0]*num_epochs)[j]
            val_chm_mae = history_data.get("val_center_heatmap_mae", [0]*num_epochs)[j]
            val_thm_mae = history_data.get("val_tip_heatmap_mae", [0]*num_epochs)[j]
            lr = history_data.get("learning_rate", [0]*num_epochs)[j]
            f.write(f"{j + 1},{loss},{chm_loss},{thm_loss},{conf_loss},{val_loss},{val_chm_mae},{val_thm_mae},{lr}\n")
    
    print("\\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80 + "\\n")
    print(f"Best epoch: {early_stop.best_epoch}")
    print(f"Best validation loss: {early_stop.best}")


if __name__ == '__main__':
    main()
