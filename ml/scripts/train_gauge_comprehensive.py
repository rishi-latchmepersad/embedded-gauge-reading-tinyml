#!/usr/bin/env python3
"""Comprehensive gauge training script for STM32 N6 deployment.

Uses all available labelled data + board captures, strong augmentation,
5-fold cross-validation, and targets <5°C MAE on hard cases.

Model constraints:
- 4GB GPU for training
- 64MB flash for deployment
- MobileNetV2 backbone (alpha tunable for size/accuracy tradeoff)
- Input: 224x224 RGB (grayscale images are replicated to 3 channels)
- Output: sigmoid-scaled to [-30, 50]°C

Data sources:
- canonical_manifest_v1.csv (141 images, highest quality labels)
- unified_training_manifest_v1.csv (409 images, broader coverage)
- Board captures with inferred labels from classical baseline

Usage:
    cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
    poetry run python scripts/train_gauge_comprehensive.py \
        --output-dir artifacts/training/comprehensive_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import KFold
from tensorflow import keras

# Add ml/src to path
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.models import build_mobilenetv2_regression_model
from embedded_gauge_reading_tinyml.board_crop_compare import load_rgb_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
VALUE_MIN: float = -30.0
VALUE_MAX: float = 50.0
VALUE_RANGE: float = VALUE_MAX - VALUE_MIN
IMAGE_SIZE: int = 224


def normalize_path(path_str: str, repo_root: Path) -> str:
    """Normalize path to be relative with forward slashes."""
    normalized = path_str.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        try:
            path = path.relative_to(repo_root)
        except ValueError:
            pass
    return path.as_posix()


def resolve_full_path(normalized_path: str, repo_root: Path) -> Path:
    """Resolve normalized path to absolute Path."""
    return repo_root / normalized_path


def load_manifest(file_path: Path, repo_root: Path) -> pd.DataFrame | None:
    """Load a manifest CSV and standardize columns."""
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    # Standardize column names
    if "label" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"label": "value"})
    if "image_path" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "image_path"})
    # Normalize paths
    df["image_path"] = df["image_path"].apply(lambda p: normalize_path(p, repo_root))
    # Resolve to absolute for loading
    df["image_path_abs"] = df["image_path"].apply(
        lambda p: str(resolve_full_path(p, repo_root))
    )
    # Ensure value is float
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # Drop rows with missing values or files
    df = df.dropna(subset=["value"])
    df = df[df["image_path_abs"].apply(lambda p: Path(p).exists())]
    return df.reset_index(drop=True)


def merge_all_manifests(repo_root: Path) -> pd.DataFrame:
    """Load and merge all available manifests with deduplication.

    Priority (highest first):
    1. canonical_manifest_v1.csv (hand-verified, highest quality)
    2. unified_training_manifest_v1.csv (broader coverage)
    3. full_labelled_plus_board30_valid_with_new5.csv
    4. hard_cases_plus_board30_valid_with_new6.csv
    5. new_labelled_captures4.csv
    6. all_captured_images_manifest.csv (board captures with labels)
    """
    manifest_files = [
        ("canonical_manifest_v1.csv", 6),
        ("unified_training_manifest_v1.csv", 5),
        ("full_labelled_plus_board30_valid_with_new5.csv", 4),
        ("hard_cases_plus_board30_valid_with_new6.csv", 3),
        ("new_labelled_captures4.csv", 2),
        ("all_captured_images_manifest.csv", 1),
    ]

    all_rows: list[pd.DataFrame] = []
    seen_paths: set[str] = set()

    for filename, priority in manifest_files:
        path = PROJECT_ROOT / "data" / filename
        df = load_manifest(path, repo_root)
        if df is None or len(df) == 0:
            logger.warning(f"Manifest not found or empty: {filename}")
            continue

        # Add source and priority
        df["source"] = filename.replace(".csv", "")
        df["priority"] = priority

        # Deduplicate: keep highest priority version
        df_new = df[~df["image_path"].isin(seen_paths)].copy()
        seen_paths.update(df["image_path"].tolist())

        logger.info(f"Loaded {filename}: {len(df)} rows, {len(df_new)} new")
        all_rows.append(df_new)

    if not all_rows:
        raise ValueError("No manifests could be loaded")

    merged = pd.concat(all_rows, ignore_index=True)
    logger.info(f"Merged dataset: {len(merged)} total images")
    logger.info(
        f"Value range: {merged['value'].min():.1f} to {merged['value'].max():.1f}"
    )
    logger.info(f"Sources: {merged['source'].value_counts().to_dict()}")

    return merged


def add_hard_case_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Tag hard cases based on value and source."""
    df = df.copy()
    # Hard cases: extreme temperatures or explicitly tagged
    df["is_hard"] = (
        (df["value"] <= -20)  # Cold extreme
        | (df["value"] >= 40)  # Hot extreme
        | df["source"].str.contains("hard_case", na=False)
    )
    return df


def compute_tail_weights(df: pd.DataFrame) -> np.ndarray:
    """Compute sample weights emphasizing tail regions.

    Cold tail: <-15°C, Hot tail: >35°C, Mid band: -15 to 35°C
    """
    weights = np.ones(len(df), dtype=np.float32)
    cold_mask = df["value"] <= -15
    hot_mask = df["value"] >= 35
    mid_mask = ~(cold_mask | hot_mask)

    # Upweight tails by 3x
    weights[cold_mask] = 3.0
    weights[hot_mask] = 3.0
    # Slightly downweight mid band
    weights[mid_mask] = 1.0

    # Also upweight by inverse frequency within 5°C bins
    bins = pd.cut(df["value"], bins=np.arange(-30, 55, 5))
    bin_counts = bins.value_counts()
    for bin_val, count in bin_counts.items():
        mask = bins == bin_val
        if count > 0:
            weights[mask] *= len(df) / (len(bin_counts) * count)

    return weights


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """Moderate augmentation pipeline matching board camera reality.

    Applied only during training. Val/test get clean images.

    NOTE: NO horizontal flip — flipping a gauge reverses needle direction
    and corrupts the label (e.g. +20C looks like -20C when flipped).
    """
    # Get image dimensions
    shape = tf.shape(image)
    h, w = shape[0], shape[1]

    # --- Random crop and resize (simulates framing variation) ---
    # Keep crop large (92-100%) to preserve gauge geometry
    crop_scale = tf.random.uniform([], 0.92, 1.0)
    crop_h = tf.cast(tf.cast(h, tf.float32) * crop_scale, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * crop_scale, tf.int32)
    crop_h = tf.maximum(crop_h, 2)
    crop_w = tf.maximum(crop_w, 2)
    image = tf.image.random_crop(image, [crop_h, crop_w, 3])
    image = tf.image.resize(image, [h, w], method="bilinear")

    # --- Photometric augmentation ---
    # Brightness: +/- 15% (moderate exposure variation)
    image = tf.image.random_brightness(image, max_delta=0.15)
    # Contrast: 0.8 to 1.2 (moderate)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    # Saturation: 0.9 to 1.1 (very subtle, images are nearly grayscale)
    image = tf.image.random_saturation(image, lower=0.90, upper=1.10)
    # Hue: minimal shift (gauge is color-agnostic)
    image = tf.image.random_hue(image, max_delta=0.03)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # --- Gamma correction (non-linear exposure response) ---
    # Narrower range: 0.7 to 1.5 to avoid extreme darkening/blowing out
    gamma = tf.random.uniform([], 0.7, 1.5)
    image = tf.pow(image, gamma)
    image = tf.clip_by_value(image, 0.0, 1.0)

    # --- Gaussian noise (sensor noise simulation) ---
    # Light noise: stddev 0.01
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.01)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)

    # NO horizontal flip — would corrupt needle direction labels

    return image


def preprocess_image_file(path: str, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load and preprocess an image file to normalized float32 numpy array.

    Uses PIL for all operations — no TensorFlow ops inside tf.numpy_function.
    This avoids shape inference issues that cause 'width must be <= target - offset'.
    """
    # Load with PIL (already returns RGB uint8 from load_rgb_image)
    img = load_rgb_image(path)

    # Ensure we have a PIL Image for resizing
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)
    else:
        pil_img = img

    # Resize with pad to preserve aspect ratio
    # Use PIL's thumbnail-like logic: fit within target_size x target_size
    # and pad with black to make it square
    pil_img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    # Create square canvas and paste centered
    square = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - pil_img.width) // 2
    offset_y = (target_size - pil_img.height) // 2
    square.paste(pil_img, (offset_x, offset_y))

    # Convert to float32 and normalize to [0, 1]
    arr = np.asarray(square, dtype=np.float32) / 255.0
    return arr


def create_tf_dataset(
    df: pd.DataFrame,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = False,
    cache: bool = True,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from a DataFrame.

    Uses on-the-fly loading to handle large datasets without preloading.
    """
    paths = df["image_path_abs"].values
    values = df["value"].values.astype(np.float32)
    weights = (
        df["sample_weight"].values.astype(np.float32)
        if "sample_weight" in df.columns
        else np.ones(len(df), np.float32)
    )

    # Targets are in CELSIUS — the model's Rescaling layer handles the
    # sigmoid [0,1] -> [-30, 50] conversion. Do NOT normalize here.
    values_celsius = values

    def load_fn(path, value, weight):
        image = tf.numpy_function(
            func=lambda p: preprocess_image_file(p.decode("utf-8")),
            inp=[path],
            Tout=tf.float32,
        )
        image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        return image, value, weight

    dataset = tf.data.Dataset.from_tensor_slices((paths, values_celsius, weights))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    # Parallel loading
    dataset = dataset.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:

        def aug_fn(image, value, weight):
            image = augment_image(image)
            return image, value, weight

        dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_model(alpha: float = 0.5, dropout: float = 0.5) -> keras.Model:
    """Build MobileNetV2 regression model.

    Args:
        alpha: Width multiplier. 0.35=small/fast, 1.0=full accuracy.
        dropout: Head dropout rate.

    Returns:
        Compiled Keras model.
    """
    model = build_mobilenetv2_regression_model(
        image_height=IMAGE_SIZE,
        image_width=IMAGE_SIZE,
        alpha=alpha,
        head_units=128,
        head_dropout=dropout,
        pretrained=True,
        value_min=VALUE_MIN,
        value_max=VALUE_MAX,
    )
    return model


def compile_model(model: keras.Model, lr: float) -> None:
    """Compile model with AdamW (weight decay), Huber loss, and MAE metrics."""
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=1e-4,  # L2 regularization to prevent overfitting
        ),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.MeanSquaredError(name="mse"),
        ],
    )


def get_callbacks(checkpoint_path: Path, patience: int = 15) -> list:
    """Get training callbacks."""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


def evaluate_model(
    model: keras.Model, dataset: tf.data.Dataset, df: pd.DataFrame
) -> dict[str, float]:
    """Evaluate model and compute metrics.

    IMPORTANT: Model output is already in Celsius (Rescaling layer in model).
    Do NOT denormalize predictions — they are already in the correct scale.
    Only training targets were normalized to [0,1] for sigmoid compatibility.
    """
    predictions = model.predict(dataset, verbose=1).flatten()
    true_values = df["value"].values

    errors = np.abs(predictions - true_values)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))
    median_error = float(np.median(errors))

    # Hard case metrics
    hard_mask = (
        df["is_hard"] if "is_hard" in df.columns else np.zeros(len(df), dtype=bool)
    )
    if hard_mask.any():
        hard_mae = float(np.mean(errors[hard_mask]))
        hard_max = float(np.max(errors[hard_mask]))
    else:
        hard_mae = 0.0
        hard_max = 0.0

    # Tail metrics
    cold_mask = true_values <= -15
    hot_mask = true_values >= 35
    mid_mask = ~(cold_mask | hot_mask)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "median_error": median_error,
        "hard_mae": hard_mae,
        "hard_max_error": hard_max,
        "cold_mae": float(np.mean(errors[cold_mask])) if cold_mask.any() else 0.0,
        "mid_mae": float(np.mean(errors[mid_mask])) if mid_mask.any() else 0.0,
        "hot_mae": float(np.mean(errors[hot_mask])) if hot_mask.any() else 0.0,
        "pct_under_5c": float(np.mean(errors < 5.0)),
        "pct_under_3c": float(np.mean(errors < 3.0)),
        "pct_under_1c": float(np.mean(errors < 1.0)),
    }

    return metrics, predictions, errors


def train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_idx: int,
    output_dir: Path,
    epochs: int = 60,
    warmup_epochs: int = 10,
    batch_size: int = 16,
    alpha: float = 0.5,
    dropout: float = 0.5,
    lr_warmup: float = 1e-3,
    lr_finetune: float = 5e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one fold of cross-validation.

    Two-stage training:
    1. Warmup: frozen backbone, train head only
    2. Fine-tune: unfrozen backbone, full model training
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Fold {fold_idx + 1}")
    logger.info(f"{'='*60}")
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")

    # Set seeds
    np.random.seed(seed + fold_idx)
    tf.random.set_seed(seed + fold_idx)
    keras.utils.set_random_seed(seed + fold_idx)

    # Compute sample weights
    train_df = train_df.copy()
    train_df["sample_weight"] = compute_tail_weights(train_df)

    # Create datasets
    train_ds = create_tf_dataset(train_df, batch_size, augment=True, shuffle=True)
    val_ds = create_tf_dataset(val_df, batch_size, augment=False, shuffle=False)
    test_ds = create_tf_dataset(test_df, batch_size, augment=False, shuffle=False)

    # Build model
    model = build_model(alpha=alpha, dropout=dropout)
    compile_model(model, lr=lr_warmup)

    # Find backbone layer
    base_model_name = f"mobilenetv2_{alpha:.2f}_224"
    try:
        base_model = model.get_layer(base_model_name)
    except ValueError:
        for layer in model.layers:
            if "mobilenetv2" in layer.name.lower():
                base_model = layer
                break
        else:
            raise ValueError("Could not find MobileNetV2 layer")

    # Phase 1: Warmup with frozen backbone
    base_model.trainable = False
    logger.info(f"\n--- Phase 1: Warmup ({warmup_epochs} epochs, frozen backbone) ---")

    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "best.weights.h5"

    callbacks = get_callbacks(checkpoint_path, patience=10)

    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: Fine-tune with unfrozen backbone
    base_model.trainable = True
    compile_model(model, lr=lr_finetune)
    logger.info(
        f"\n--- Phase 2: Fine-tune ({epochs - warmup_epochs} epochs, unfrozen backbone) ---"
    )

    # Reset callbacks for fine-tuning with longer patience
    callbacks = get_callbacks(checkpoint_path, patience=15)

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on all sets
    logger.info(f"\n--- Fold {fold_idx} Evaluation ---")

    train_metrics, _, _ = evaluate_model(model, train_ds, train_df)
    val_metrics, val_preds, val_errors = evaluate_model(model, val_ds, val_df)
    test_metrics, test_preds, test_errors = evaluate_model(model, test_ds, test_df)

    logger.info(f"Train MAE: {train_metrics['mae']:.2f}°C")
    logger.info(
        f"Val MAE: {val_metrics['mae']:.2f}°C (hard: {val_metrics['hard_mae']:.2f}°C)"
    )
    logger.info(
        f"Test MAE: {test_metrics['mae']:.2f}°C (hard: {test_metrics['hard_mae']:.2f}°C)"
    )
    logger.info(f"Test % under 5°C: {test_metrics['pct_under_5c']*100:.1f}%")
    logger.info(f"Test % under 3°C: {test_metrics['pct_under_3c']*100:.1f}%")

    # Save predictions
    val_df = val_df.copy()
    val_df["prediction"] = val_preds
    val_df["error"] = val_errors
    val_df.to_csv(fold_dir / "val_predictions.csv", index=False)

    test_df = test_df.copy()
    test_df["prediction"] = test_preds
    test_df["error"] = test_errors
    test_df.to_csv(fold_dir / "test_predictions.csv", index=False)

    # Save model
    model.save(fold_dir / "model.keras")

    # Save history
    history = {
        "warmup": {
            k: [float(v) for v in vals] for k, vals in history_warmup.history.items()
        },
        "finetune": {
            k: [float(v) for v in vals] for k, vals in history_finetune.history.items()
        },
    }
    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "fold": fold_idx,
        "train_mae": train_metrics["mae"],
        "val_mae": val_metrics["mae"],
        "val_hard_mae": val_metrics["hard_mae"],
        "test_mae": test_metrics["mae"],
        "test_hard_mae": test_metrics["hard_mae"],
        "test_max_error": test_metrics["max_error"],
        "test_pct_under_5c": test_metrics["pct_under_5c"],
        "test_pct_under_3c": test_metrics["pct_under_3c"],
        "test_cold_mae": test_metrics["cold_mae"],
        "test_mid_mae": test_metrics["mid_mae"],
        "test_hot_mae": test_metrics["hot_mae"],
    }


def run_cross_validation(
    df: pd.DataFrame,
    output_dir: Path,
    n_folds: int = 5,
    epochs: int = 60,
    warmup_epochs: int = 10,
    batch_size: int = 16,
    alpha: float = 0.5,
    dropout: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run k-fold cross-validation.

    Stratified by value bins to ensure all temperature ranges in each fold.
    """
    df = add_hard_case_tags(df)

    # Create stratification key: value bin + hard case flag
    df["value_bin"] = pd.cut(df["value"], bins=np.arange(-30, 55, 5))
    df["strat_key"] = df["value_bin"].astype(str) + "_" + df["is_hard"].astype(str)

    # Group by strat key and assign fold indices
    fold_indices = np.zeros(len(df), dtype=int)
    for key, group in df.groupby("strat_key"):
        indices = group.index.values
        if len(indices) < n_folds:
            # If too few samples, assign randomly
            fold_indices[indices] = np.random.randint(0, n_folds, size=len(indices))
        else:
            # Distribute evenly across folds
            fold_indices[indices] = np.arange(len(indices)) % n_folds

    df["fold"] = fold_indices

    # Run each fold
    results: list[dict[str, Any]] = []
    for fold_idx in range(n_folds):
        train_df = df[df["fold"] != fold_idx].copy()
        val_test_df = df[df["fold"] == fold_idx].copy()

        # Split val_test into val and test (50/50)
        val_test_df = val_test_df.sample(
            frac=1.0, random_state=seed + fold_idx
        ).reset_index(drop=True)
        split_point = len(val_test_df) // 2
        val_df = val_test_df.iloc[:split_point].copy()
        test_df = val_test_df.iloc[split_point:].copy()

        result = train_fold(
            train_df,
            val_df,
            test_df,
            fold_idx=fold_idx,
            output_dir=output_dir,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            batch_size=batch_size,
            alpha=alpha,
            dropout=dropout,
            seed=seed,
        )
        results.append(result)

    # Aggregate results
    logger.info(f"\n{'='*60}")
    logger.info("Cross-Validation Summary")
    logger.info(f"{'='*60}")

    metrics_to_avg = [
        "train_mae",
        "val_mae",
        "val_hard_mae",
        "test_mae",
        "test_hard_mae",
        "test_max_error",
        "test_pct_under_5c",
        "test_pct_under_3c",
        "test_cold_mae",
        "test_mid_mae",
        "test_hot_mae",
    ]

    summary = {}
    for metric in metrics_to_avg:
        values = [r[metric] for r in results]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
        summary[f"{metric}_min"] = float(np.min(values))
        summary[f"{metric}_max"] = float(np.max(values))
        logger.info(
            f"{metric}: {summary[f'{metric}_mean']:.3f} ± {summary[f'{metric}_std']:.3f} "
            f"(range: {summary[f'{metric}_min']:.3f} - {summary[f'{metric}_max']:.3f})"
        )

    # Save summary
    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump({"folds": results, "summary": summary}, f, indent=2)

    # Save full predictions
    all_predictions = []
    for fold_idx in range(n_folds):
        fold_dir = output_dir / f"fold_{fold_idx}"
        test_preds = pd.read_csv(fold_dir / "test_predictions.csv")
        all_predictions.append(test_preds)
    full_preds = pd.concat(all_predictions, ignore_index=True)
    full_preds.to_csv(output_dir / "all_test_predictions.csv", index=False)

    # Overall metrics on all test predictions
    errors = np.abs(full_preds["prediction"] - full_preds["value"])
    logger.info(f"\nOverall Test MAE (all folds): {np.mean(errors):.2f}°C")
    logger.info(f"Overall Test % under 5°C: {np.mean(errors < 5)*100:.1f}%")
    logger.info(f"Overall Test % under 3°C: {np.mean(errors < 3)*100:.1f}%")

    return summary


def train_final_model(
    df: pd.DataFrame,
    output_dir: Path,
    epochs: int = 60,
    warmup_epochs: int = 10,
    batch_size: int = 16,
    alpha: float = 0.5,
    dropout: float = 0.5,
    seed: int = 42,
) -> keras.Model:
    """Train final model on all data.

    Uses a train/val split for early stopping, then retrains on all data
    with best hyperparameters.
    """
    logger.info(f"\n{'='*60}")
    logger.info("Training Final Model on All Data")
    logger.info(f"{'='*60}")

    df = add_hard_case_tags(df)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # 85/15 train/val split
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    # Compute weights
    train_df["sample_weight"] = compute_tail_weights(train_df)

    # Create datasets
    train_ds = create_tf_dataset(train_df, batch_size, augment=True, shuffle=True)
    val_ds = create_tf_dataset(val_df, batch_size, augment=False, shuffle=False)

    # Build model
    model = build_model(alpha=alpha, dropout=dropout)
    compile_model(model, lr=1e-3)

    # Find backbone
    base_model_name = f"mobilenetv2_{alpha:.2f}_224"
    try:
        base_model = model.get_layer(base_model_name)
    except ValueError:
        for layer in model.layers:
            if "mobilenetv2" in layer.name.lower():
                base_model = layer
                break
        else:
            raise ValueError("Could not find MobileNetV2 layer")

    # Phase 1: Warmup
    base_model.trainable = False
    logger.info(f"\n--- Final Model Warmup ({warmup_epochs} epochs) ---")

    checkpoint_path = output_dir / "final_best.weights.h5"
    callbacks = get_callbacks(checkpoint_path, patience=10)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: Fine-tune
    base_model.trainable = True
    compile_model(model, lr=5e-5)
    logger.info(f"\n--- Final Model Fine-tune ({epochs - warmup_epochs} epochs) ---")

    callbacks = get_callbacks(checkpoint_path, patience=15)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    full_ds = create_tf_dataset(df, batch_size, augment=False, shuffle=False)
    metrics, preds, errors = evaluate_model(model, full_ds, df)

    logger.info(f"\nFinal Model Metrics:")
    logger.info(f"  MAE: {metrics['mae']:.2f}°C")
    logger.info(f"  Hard MAE: {metrics['hard_mae']:.2f}°C")
    logger.info(f"  % under 5°C: {metrics['pct_under_5c']*100:.1f}%")
    logger.info(f"  % under 3°C: {metrics['pct_under_3c']*100:.1f}%")

    # Save
    model.save(output_dir / "final_model.keras")

    df_out = df.copy()
    df_out["prediction"] = preds
    df_out["error"] = errors
    df_out.to_csv(output_dir / "final_predictions.csv", index=False)

    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train comprehensive gauge model")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=60, help="Total epochs")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="MobileNetV2 width multiplier"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Head dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument(
        "--skip-final", action="store_true", help="Skip final model training"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # Log GPU info
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPUs available: {len(gpus)}")
    if gpus:
        logger.info(f"GPU: {gpus[0]}")

    # Load all data
    logger.info("Loading all manifests...")
    df = merge_all_manifests(REPO_ROOT)

    # Save merged dataset
    df.to_csv(output_dir / "merged_dataset.csv", index=False)

    # Run cross-validation
    if not args.skip_cv:
        cv_summary = run_cross_validation(
            df,
            output_dir=output_dir / "cv",
            n_folds=args.n_folds,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            alpha=args.alpha,
            dropout=args.dropout,
            seed=args.seed,
        )

    # Train final model
    if not args.skip_final:
        final_model = train_final_model(
            df,
            output_dir=output_dir,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            alpha=args.alpha,
            dropout=args.dropout,
            seed=args.seed,
        )

    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
