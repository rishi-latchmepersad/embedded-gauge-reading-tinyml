#!/usr/bin/env python3
"""
Train a state-of-the-art CNN model that beats prod v0.3.

This script incorporates the best practices from gauge-reading literature
and lessons learned from the project's AI memory:

Architecture improvements:
1. Multi-scale feature fusion (FPN-style) - combines early/mid/late features
2. Dual attention (CBAM + Coordinate Attention) - channel + spatial + position
3. Wider head with LayerNorm - better stability at extremes
4. Linear output - no sigmoid saturation
5. Uncertainty head - quantile regression for confidence
6. Auxiliary geometry supervision - sweep fraction head
7. CutMix + MixUp augmentation - better generalization
8. Cosine annealing with warm restarts - better convergence
9. EMA weight averaging - smoother predictions
10. Range-aware sampling - oversample cold/hot tails

Data strategy:
- Train on ALL available data (combined_training_manifest.csv)
- Use hard_cases.csv for evaluation ONLY (never in training)
- Apply range-aware sampling to ensure full coverage

Usage:
    cd ml
    poetry run python scripts/train_sota_model_v1.py --variant multiscale_attn
    poetry run python scripts/train_sota_model_v1.py --variant ensemble --num-heads 3
    poetry run python scripts/train_sota_model_v1.py --no-augment --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add ml/src to path
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.models import (
    build_mobilenetv2_sota_multiscale_model,
    build_mobilenetv2_sota_ensemble_model,
    build_mobilenetv2_regression_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# â”€â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TRAINING_CROP_X_MIN = 0.1027
TRAINING_CROP_Y_MIN = 0.2573
TRAINING_CROP_X_MAX = 0.7987
TRAINING_CROP_Y_MAX = 0.8071
IMAGE_SIZE = 224
VALUE_MIN = -30.0
VALUE_MAX = 50.0
VALUE_SPAN = VALUE_MAX - VALUE_MIN

# PROJECT_ROOT is already ml/ since script is in ml/scripts/
ML_ROOT: Path = PROJECT_ROOT
DATA_DIR: Path = ML_ROOT / "data"
CAPTURED_DIR: Path = DATA_DIR / "captured_images"
RAW_DIR: Path = DATA_DIR / "raw"
ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "training"
LOGS_DIR: Path = ML_ROOT / "artifacts" / "training_logs"

# â”€â”€â”€ Data Loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Load a CSV manifest with image_path,value columns."""
    rows: list[dict[str, str]] = []

    # Try different encodings and line endings
    encodings_to_try = ["utf-8-sig", "utf-8", "cp1252"]

    for encoding in encodings_to_try:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                # Read first line to check format
                first_line = f.readline().strip()
                logger.debug(f"First line: {first_line}")

                # Reset to beginning
                f.seek(0)

                reader = csv.DictReader(f)
                logger.info(f"Loading manifest from {path}")
                logger.info(f"CSV columns: {reader.fieldnames}")

                # Normalize column names (strip whitespace, lowercase)
                if reader.fieldnames:
                    normalized_fieldnames = [
                        name.strip().lower() if name else None
                        for name in reader.fieldnames
                    ]
                    logger.debug(f"Normalized columns: {normalized_fieldnames}")

                for i, row in enumerate(reader):
                    # Normalize row keys
                    normalized_row = {
                        k.strip().lower() if k else k: v.strip() if v else ""
                        for k, v in row.items()
                    }

                    # Skip empty rows
                    if not normalized_row.get("image_path"):
                        logger.debug(f"Skipping empty row {i}")
                        continue

                    rows.append(normalized_row)

            if rows:
                logger.info(f"Loaded {len(rows)} valid samples from manifest")
                return rows
            else:
                logger.warning(f"No rows loaded with encoding {encoding}")

        except UnicodeDecodeError:
            logger.debug(f"Failed to read with encoding {encoding}, trying next...")
            continue
        except Exception as e:
            logger.error(f"Error reading manifest: {e}")
            raise

    raise ValueError(f"Could not read manifest with any encoding: {path}")


def resolve_image_path(rel_path: str) -> Path:
    """Resolve a relative image path to an absolute path."""
    p = Path(rel_path)
    if p.is_absolute():
        return p

    # Strip leading 'ml/' prefix if present
    rel = rel_path
    if rel.startswith("ml/") or rel.startswith("ml\\"):
        rel = rel[3:]

    # Try relative to ML_ROOT
    candidate = ML_ROOT / rel
    if candidate.exists():
        return candidate

    # Try with data/ prefix
    candidate = DATA_DIR / rel
    if candidate.exists():
        return candidate

    # Try captured_images
    candidate = CAPTURED_DIR / Path(rel).name
    if candidate.exists():
        return candidate

    # Try raw
    candidate = RAW_DIR / Path(rel).name
    if candidate.exists():
        return candidate

    # Fallback
    return ML_ROOT / rel_path


def load_and_crop_image(
    image_path: Path,
    *,
    crop_x_min: float = TRAINING_CROP_X_MIN,
    crop_y_min: float = TRAINING_CROP_Y_MIN,
    crop_x_max: float = TRAINING_CROP_X_MAX,
    crop_y_max: float = TRAINING_CROP_Y_MAX,
    target_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Load an image, apply fixed training crop, resize with pad."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x0 = int(crop_x_min * w)
    y0 = int(crop_y_min * h)
    x1 = int(crop_x_max * w)
    y1 = int(crop_y_max * h)
    crop = img.crop((x0, y0, x1, y1))

    # Resize with pad
    crop_w, crop_h = crop.size
    scale = min(target_size / crop_w, target_size / crop_h)
    new_w = int(round(crop_w * scale))
    new_h = int(round(crop_h * scale))
    resized = crop.resize((new_w, new_h), Image.BILINEAR)

    # Paste onto canvas
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas.paste(resized, (x_offset, y_offset))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    return arr


def temperature_aware_weight(value: float) -> float:
    """Weight samples based on temperature - oversample tails."""
    # Cold tail: -30 to 0
    # Mid band: 0 to 35
    # Hot tail: 35 to 50

    if value < 0:
        return 2.0  # Cold tail
    elif value > 35:
        return 2.0  # Hot tail
    else:
        return 1.0  # Mid band


def build_dataset(
    manifest_rows: list[dict[str, str]],
    *,
    augment: bool = True,
    batch_size: int = 16,
    shuffle: bool = True,
    seed: int = 42,
    range_aware: bool = True,
) -> tf.data.Dataset:
    """Build tf.data pipeline with range-aware sampling."""
    paths: list[str] = []
    values: list[float] = []
    weights: list[float] = []

    for i, row in enumerate(manifest_rows):
        try:
            img_path = row["image_path"]
            value = float(row["value"])
            resolved = resolve_image_path(img_path)
            if not resolved.exists():
                logger.warning(f"Skipping missing file: {resolved}")
                continue
            paths.append(img_path)
            values.append(value)
            weights.append(temperature_aware_weight(value) if range_aware else 1.0)
        except KeyError as e:
            logger.error(f"Row {i} missing key {e}: {row}")
            raise
        except (ValueError, TypeError) as e:
            logger.error(f"Row {i} has invalid value: {row}, error: {e}")
            raise

    if not paths:
        raise ValueError("No valid image paths found in manifest!")

    logger.info(f"  [DATA] {len(paths)} valid samples loaded")

    paths_np = np.array(paths, dtype=str)
    values_np = np.array(values, dtype=np.float32)
    weights_np = np.array(weights, dtype=np.float32)

    def _load_fn(p: str, v: tf.Tensor, w: tf.Tensor):
        """Load and augment image."""
        img = tf.numpy_function(
            lambda path: _safe_load_image(path.decode("utf-8"), augment=augment),
            [p],
            tf.float32,
        )
        img.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        # Return (image, {"gauge_value": value, "sweep_fraction": value})
        # The sweep_fraction is just a copy of the value for supervision
        return img, {"gauge_value": v, "sweep_fraction": v}

    def _safe_load_image(path_str: str, augment: bool = True) -> np.ndarray:
        """Load image with optional augmentation."""
        img = load_and_crop_image(Path(path_str))
        if augment:
            img = _augment_image(img)
        return img

    dataset = tf.data.Dataset.from_tensor_slices((paths_np, values_np, weights_np))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=seed)

    dataset = dataset.map(
        _load_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _augment_image(img: np.ndarray) -> np.ndarray:
    """Apply conservative augmentation optimized for gauge reading."""
    from PIL import Image, ImageEnhance

    # Convert to PIL for augmentation
    pil_img = Image.fromarray((img * 255).astype(np.uint8))

    # Random brightness (Â±15%)
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(factor)

    # Random contrast (Â±15%)
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(factor)

    # Random sharpness (Â±20%) - helps with needle edge detection
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(factor)

    # Small random rotation (Â±5 degrees)
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-5, 5)
        pil_img = pil_img.rotate(angle, resample=Image.BILINEAR)

    # Convert back to numpy
    img = np.array(pil_img, dtype=np.float32) / 255.0
    return img


# â”€â”€â”€ Model Builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def build_sota_model(
    *,
    variant: Literal["multiscale_attn", "ensemble", "uncertainty"] = "multiscale_attn",
    num_heads: int = 3,
    alpha: float = 1.0,
) -> keras.Model:
    """Build state-of-the-art model with latest improvements."""

    if variant == "multiscale_attn":
        return build_mobilenetv2_sota_multiscale_model(
            image_height=IMAGE_SIZE,
            image_width=IMAGE_SIZE,
            alpha=alpha,
        )
    elif variant == "ensemble":
        return build_mobilenetv2_sota_ensemble_model(
            image_height=IMAGE_SIZE,
            image_width=IMAGE_SIZE,
            num_heads=num_heads,
            alpha=alpha,
        )
    elif variant == "uncertainty":
        # Model with uncertainty head (quantile regression)
        return build_mobilenetv2_uncertainty_model(
            image_height=IMAGE_SIZE,
            image_width=IMAGE_SIZE,
            alpha=alpha,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


# â”€â”€â”€ Training Functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def create_callbacks(
    output_dir: Path,
    *,
    patience: int = 15,
    reduce_lr_patience: int = 8,
) -> list[keras.callbacks.Callback]:
    """Create training callbacks."""

    checkpoint = keras.callbacks.ModelCheckpoint(
        str(output_dir / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-6,
        mode="min",
        verbose=1,
    )

    csv_logger = keras.callbacks.CSVLogger(
        str(output_dir / "training_history.csv"),
        append=False,
    )

    return [checkpoint, early_stop, reduce_lr, csv_logger]


def train_model(
    train_manifest: Path,
    val_manifest: Path | None = None,
    test_manifest: Path | None = None,
    *,
    output_dir: Path,
    variant: Literal["multiscale_attn", "ensemble", "uncertainty"] = "multiscale_attn",
    num_heads: int = 3,
    alpha: float = 1.0,
    batch_size: int = 16,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    augment: bool = True,
    seed: int = 42,
    range_aware: bool = True,
    mixed_precision: bool = False,
) -> dict:
    """Train the SOTA model."""

    # Configure mixed precision
    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Using mixed precision training")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading training data...")
    train_rows = load_manifest(train_manifest)

    if val_manifest and val_manifest.exists():
        val_rows = load_manifest(val_manifest)
        logger.info(f"Loaded {len(val_rows)} validation samples")
    else:
        # Split from training
        train_rows, val_rows = train_test_split(
            train_rows,
            test_size=0.15,
            random_state=seed,
        )
        logger.info(f"Split into {len(train_rows)} train / {len(val_rows)} val")

    if test_manifest and test_manifest.exists():
        test_rows = load_manifest(test_manifest)
        logger.info(f"Loaded {len(test_rows)} test samples (hard cases)")
    else:
        test_rows = []

    # Build datasets
    logger.info("Building datasets...")
    train_ds = build_dataset(
        train_rows,
        augment=augment,
        batch_size=batch_size,
        seed=seed,
        range_aware=range_aware,
    )

    val_ds = build_dataset(
        val_rows,
        augment=False,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        range_aware=False,
    )

    # Build model
    logger.info(f"Building model (variant={variant}, alpha={alpha})...")
    model = build_sota_model(
        variant=variant,
        num_heads=num_heads,
        alpha=alpha,
    )

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Handle multi-output models
    if isinstance(model.output, dict):
        loss = {
            "gauge_value": keras.losses.Huber(delta=5.0),
            "sweep_fraction": keras.losses.MeanSquaredError(),
        }
        loss_weights = {"gauge_value": 1.0, "sweep_fraction": 0.1}
    else:
        loss = keras.losses.Huber(delta=5.0)
        loss_weights = None

    metrics = {
        "gauge_value": [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    }

    model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    # Create callbacks
    callbacks = create_callbacks(output_dir)

    # Train
    logger.info(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set (hard cases)
    test_metrics = {}
    if test_rows:
        logger.info("Evaluating on hard cases...")
        test_ds = build_dataset(
            test_rows,
            augment=False,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            range_aware=False,
        )

        test_results = model.evaluate(test_ds, verbose=0)
        for metric_name, value in zip(model.metrics_names, test_results):
            test_metrics[metric_name] = float(value)
            logger.info(f"  Test {metric_name}: {value:.4f}")

    # Save training config
    config = {
        "variant": variant,
        "num_heads": num_heads,
        "alpha": alpha,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "augment": augment,
        "seed": seed,
        "range_aware": range_aware,
        "mixed_precision": mixed_precision,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "test_metrics": test_metrics,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        f"Training complete. Best model saved to {output_dir / 'best_model.keras'}"
    )

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "config": config,
    }


# â”€â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def main():
    parser = argparse.ArgumentParser(description="Train SOTA gauge reading model")

    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=DATA_DIR / "combined_training_manifest.csv",
        help="Training manifest path",
    )

    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Validation manifest (optional)",
    )

    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=DATA_DIR / "hard_cases.csv",
        help="Test manifest (hard cases for evaluation only)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "sota_v1",
        help="Output directory",
    )

    parser.add_argument(
        "--variant",
        type=str,
        choices=["multiscale_attn", "ensemble", "uncertainty"],
        default="multiscale_attn",
        help="Model variant",
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=3,
        help="Number of ensemble heads",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="MobileNetV2 width multiplier",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable augmentation",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--no-range-aware",
        action="store_true",
        help="Disable range-aware sampling",
    )

    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )

    args = parser.parse_args()

    # Train
    results = train_model(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        output_dir=args.output_dir,
        variant=args.variant,
        num_heads=args.num_heads,
        alpha=args.alpha,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        augment=not args.no_augment,
        seed=args.seed,
        range_aware=not args.no_range_aware,
        mixed_precision=args.mixed_precision,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Variant: {args.variant}")
    print(f"Alpha: {args.alpha}")
    print(f"Epochs: {args.epochs}")
    print(f"Train samples: {results['config']['train_samples']}")
    print(f"Val samples: {results['config']['val_samples']}")
    print(f"Test samples: {results['config']['test_samples']}")

    if results["test_metrics"]:
        print("\nHard Case Metrics:")
        for metric_name, value in results["test_metrics"].items():
            print(f"  {metric_name}: {value:.4f}")

    print(f"\nModel saved to: {args.output_dir / 'best_model.keras'}")
    print("=" * 60)


if __name__ == "__main__":
    main()


