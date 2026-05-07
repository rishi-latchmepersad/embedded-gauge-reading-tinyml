#!/usr/bin/env python3
"""Comprehensive gauge training v2 — hybrid approach.

Uses canonical baseline's proven training pipeline (preload images, conservative
augmentation, Adam optimizer) with comprehensive data merging and 5-fold CV.

Key differences from v1:
- Preloads images into memory (no tf.numpy_function issues)
- Uses canonical baseline's proven augmentation (no flip, moderate photometric)
- Adam optimizer (not AdamW) with conservative LRs: 1e-4 warmup / 1e-5 fine-tune
- Batch size 8 (not 16) for stable gradients on small datasets
- Alpha 1.0, dropout 0.2 (proven defaults from canonical baseline)

Usage:
    cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
    poetry run python scripts/train_gauge_comprehensive_v2.py \
        --output-dir artifacts/training/comprehensive_v3
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
    df["image_path_resolved"] = df["image_path"].apply(
        lambda p: str(resolve_full_path(p, repo_root))
    )
    # Ensure value is float
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # Drop rows with missing values or files
    df = df.dropna(subset=["value"])
    df = df[df["image_path_resolved"].apply(lambda p: Path(p).exists())]
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
    logger.info(f"Value range: {merged['value'].min():.1f} to {merged['value'].max():.1f}")
    logger.info(f"Sources: {merged['source'].value_counts().to_dict()}")

    return merged


def add_hard_case_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Tag hard cases based on value and source."""
    df = df.copy()
    df["is_hard"] = (
        (df["value"] <= -20)
        | (df["value"] >= 40)
        | df["source"].str.contains("hard_case", na=False)
    )
    return df


def create_value_bins(df: pd.DataFrame, bin_size: float = 5.0) -> pd.DataFrame:
    """Create value bins for weighted sampling."""
    df = df.copy()
    min_val = df["value"].min()
    max_val = df["value"].max()
    bin_edges = np.arange(
        np.floor(min_val / bin_size) * bin_size,
        np.ceil(max_val / bin_size) * bin_size + bin_size,
        bin_size,
    )
    df["value_bin"] = pd.cut(df["value"], bins=bin_edges, include_lowest=True)
    return df


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Compute sample weights inversely proportional to bin frequency.

    Weights are capped at 5.0 to prevent extreme values from destabilizing training.
    """
    bin_counts = df["value_bin"].value_counts()
    total = len(df)
    num_bins = len(bin_counts)

    bin_weights = {}
    for bin_val, count in bin_counts.items():
        if count > 0:
            # Inverse frequency weighting, capped at 5.0
            weight = min(total / (num_bins * count), 5.0)
            bin_weights[bin_val] = weight
        else:
            bin_weights[bin_val] = 0.0

    weights = df["value_bin"].map(bin_weights).values
    return weights.astype(np.float32)


def preprocess_image(image_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and preprocess an image for training.

    Uses tf.image.resize_with_pad for aspect-ratio-preserving resize.
    """
    img = load_rgb_image(image_path)
    image_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
    resized = tf.image.resize_with_pad(
        tf.cast(image_tensor, tf.float32),
        target_size[0],
        target_size[1],
        method="bilinear",
    )
    img_resized = np.clip(np.rint(resized.numpy()), 0.0, 255.0).astype(np.uint8)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized


def create_dataset(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool = False,
    use_weights: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from a DataFrame.

    Preloads all images into memory for faster, more reliable training.
    """
    logger.info(f"Loading {len(df)} images into memory...")
    images = []
    values = []

    for idx, row in df.iterrows():
        try:
            img = preprocess_image(row["image_path_resolved"], (224, 224))
            images.append(img)
            values.append(float(row["value"]))
        except Exception as e:
            logger.warning(f"Failed to load {row['image_path_resolved']}: {e}")
            continue

    images = np.array(images, dtype=np.float32)
    values = np.array(values, dtype=np.float32)

    logger.info(f"Successfully loaded {len(images)} images")

    if use_weights and "sample_weight" in df.columns:
        weights = df["sample_weight"].values[: len(images)].astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((images, values, weights))
        dataset = dataset.map(lambda x, y, w: (x, y, w))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, values))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(images), reshuffle_each_iteration=True
        )

    if augment:
        def _augment_strong(image: tf.Tensor) -> tf.Tensor:
            """Apply strong augmentation matching board camera reality.

            NO horizontal flip — would reverse needle direction and corrupt labels.
            """
            image_shape = tf.shape(image)
            image_h = image_shape[0]
            image_w = image_shape[1]

            # --- Crop jitter: simulate slight dial position variations ---
            scale = tf.random.uniform([], minval=0.90, maxval=1.0, dtype=tf.float32)
            crop_h = tf.maximum(
                2, tf.cast(tf.cast(image_h, tf.float32) * scale, tf.int32)
            )
            crop_w = tf.maximum(
                2, tf.cast(tf.cast(image_w, tf.float32) * scale, tf.int32)
            )
            image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
            image = tf.image.resize(image, [image_h, image_w])

            # --- Brightness / exposure ---
            image = tf.image.random_brightness(image, max_delta=0.20)
            # --- Contrast ---
            image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
            # --- Saturation ---
            image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
            image = tf.clip_by_value(image, 0.0, 1.0)

            # --- Gamma augmentation ---
            gamma_dark = tf.random.uniform([], minval=1.0, maxval=2.2, dtype=tf.float32)
            gamma_bright = tf.random.uniform(
                [], minval=0.6, maxval=1.0, dtype=tf.float32
            )
            apply_dark = tf.random.uniform([]) < 0.25
            apply_bright = tf.random.uniform([]) < 0.25
            gamma = tf.where(
                apply_dark, gamma_dark, tf.where(apply_bright, gamma_bright, 1.0)
            )
            image = tf.pow(image, gamma)
            image = tf.clip_by_value(image, 0.0, 1.0)

            # --- Glare blobs ---
            BLOB_RES = 32
            mask = tf.zeros([BLOB_RES, BLOB_RES, 1], dtype=tf.float32)
            for _ in range(3):
                active = tf.cast(tf.random.uniform([]) < 0.5, tf.float32)
                cx = tf.random.uniform([], 2, BLOB_RES - 2, dtype=tf.float32)
                cy = tf.random.uniform([], 2, BLOB_RES - 2, dtype=tf.float32)
                brightness = tf.random.uniform([], 0.5, 1.0) * active
                cy_i = tf.cast(tf.round(cy), tf.int32)
                cx_i = tf.cast(tf.round(cx), tf.int32)
                idx = tf.stack([cy_i, cx_i, 0])
                spot = tf.tensor_scatter_nd_update(
                    tf.zeros([BLOB_RES, BLOB_RES, 1], dtype=tf.float32),
                    [idx],
                    [brightness],
                )
                mask = mask + spot
            mask = tf.image.resize(mask, [image_h, image_w], method="bicubic")
            mask = tf.clip_by_value(mask, 0.0, 1.0)
            image = image + mask * (1.0 - image)
            image = tf.clip_by_value(image, 0.0, 1.0)

            # --- Gaussian noise ---
            image = image + tf.random.normal(tf.shape(image), stddev=0.015)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image

        if use_weights:
            dataset = dataset.map(
                lambda x, y, w: (_augment_strong(x), y, w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            dataset = dataset.map(
                lambda x, y: (_augment_strong(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_and_compile_model(
    alpha: float = 1.0,
    dropout: float = 0.2,
    lr: float = 1e-4,
) -> tuple[keras.Model, keras.Model]:
    """Build and compile MobileNetV2 regression model.

    Returns:
        Tuple of (model, backbone_layer).
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

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae", "mse"],
    )

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

    return model, base_model


def get_callbacks(checkpoint_path: Path) -> list:
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
            patience=15,
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
    ]


def evaluate_model(
    model: keras.Model, dataset: tf.data.Dataset, df: pd.DataFrame
) -> dict[str, float]:
    """Evaluate model and compute metrics."""
    predictions = model.predict(dataset, verbose=1).flatten()
    true_values = df["value"].values

    errors = np.abs(predictions - true_values)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))
    median_error = float(np.median(errors))

    hard_mask = (
        df["is_hard"] if "is_hard" in df.columns else np.zeros(len(df), dtype=bool)
    )
    if hard_mask.any():
        hard_mae = float(np.mean(errors[hard_mask]))
        hard_max = float(np.max(errors[hard_mask]))
    else:
        hard_mae = 0.0
        hard_max = 0.0

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
    epochs: int = 40,
    warmup_epochs: int = 8,
    batch_size: int = 8,
    alpha: float = 1.0,
    dropout: float = 0.2,
    lr_warmup: float = 1e-4,
    lr_finetune: float = 1e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one fold using canonical baseline pipeline."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Fold {fold_idx + 1}")
    logger.info(f"{'='*60}")
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")

    np.random.seed(seed + fold_idx)
    tf.random.set_seed(seed + fold_idx)
    keras.utils.set_random_seed(seed + fold_idx)

    # Compute sample weights
    train_df = train_df.copy()
    train_df = create_value_bins(train_df)
    train_df["sample_weight"] = compute_sample_weights(train_df)

    # Create datasets
    train_ds = create_dataset(
        train_df, batch_size, shuffle=True, use_weights=True, augment=True
    )
    val_ds = create_dataset(val_df, batch_size, shuffle=False, use_weights=False, augment=False)
    test_ds = create_dataset(test_df, batch_size, shuffle=False, use_weights=False, augment=False)

    # Build model
    model, base_model = build_and_compile_model(alpha=alpha, dropout=dropout, lr=5e-5)

    # Phase 1: Warmup with frozen backbone
    base_model.trainable = False
    logger.info(f"\n--- Phase 1: Warmup ({warmup_epochs} epochs, frozen backbone) ---")

    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "best.weights.h5"

    callbacks = get_callbacks(checkpoint_path)

    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: Fine-tune with unfrozen backbone
    base_model.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-6),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae", "mse"],
    )
    logger.info(
        f"\n--- Phase 2: Fine-tune ({epochs - warmup_epochs} epochs, unfrozen backbone) ---"
    )

    callbacks = get_callbacks(checkpoint_path)

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
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
            k: [float(v) for v in vals]
            for k, vals in history_warmup.history.items()
        },
        "finetune": {
            k: [float(v) for v in vals]
            for k, vals in history_finetune.history.items()
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
    epochs: int = 40,
    warmup_epochs: int = 8,
    batch_size: int = 8,
    alpha: float = 1.0,
    dropout: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Run k-fold cross-validation with stratification by value bins."""
    df = add_hard_case_tags(df)

    # Create stratification key
    df["value_bin"] = pd.cut(df["value"], bins=np.arange(-30, 55, 5))
    df["strat_key"] = df["value_bin"].astype(str) + "_" + df["is_hard"].astype(str)

    # Assign fold indices
    fold_indices = np.zeros(len(df), dtype=int)
    for key, group in df.groupby("strat_key"):
        indices = group.index.values
        if len(indices) < n_folds:
            fold_indices[indices] = np.random.randint(0, n_folds, size=len(indices))
        else:
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

    errors = np.abs(full_preds["prediction"] - full_preds["value"])
    logger.info(f"\nOverall Test MAE (all folds): {np.mean(errors):.2f}°C")
    logger.info(f"Overall Test % under 5°C: {np.mean(errors < 5)*100:.1f}%")
    logger.info(f"Overall Test % under 3°C: {np.mean(errors < 3)*100:.1f}%")

    return summary


def train_final_model(
    df: pd.DataFrame,
    output_dir: Path,
    epochs: int = 40,
    warmup_epochs: int = 8,
    batch_size: int = 8,
    alpha: float = 1.0,
    dropout: float = 0.2,
    seed: int = 42,
) -> keras.Model:
    """Train final model on all data with train/val split."""
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
    train_df = create_value_bins(train_df)
    train_df["sample_weight"] = compute_sample_weights(train_df)

    # Create datasets
    train_ds = create_dataset(
        train_df, batch_size, shuffle=True, use_weights=True, augment=True
    )
    val_ds = create_dataset(val_df, batch_size, shuffle=False, use_weights=False, augment=False)

    # Build model
    model, base_model = build_and_compile_model(alpha=alpha, dropout=dropout, lr=5e-5)

    # Phase 1: Warmup
    base_model.trainable = False
    logger.info(f"\n--- Final Model Warmup ({warmup_epochs} epochs) ---")

    checkpoint_path = output_dir / "final_best.weights.h5"
    callbacks = get_callbacks(checkpoint_path)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: Fine-tune
    base_model.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-6),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae", "mse"],
    )
    logger.info(f"\n--- Final Model Fine-tune ({epochs - warmup_epochs} epochs) ---")

    callbacks = get_callbacks(checkpoint_path)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    full_ds = create_dataset(df, batch_size, shuffle=False, use_weights=False, augment=False)
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
    parser = argparse.ArgumentParser(description="Train comprehensive gauge model v2")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=40, help="Total epochs")
    parser.add_argument("--warmup-epochs", type=int, default=8, help="Warmup epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--alpha", type=float, default=1.0, help="MobileNetV2 width multiplier")
    parser.add_argument("--dropout", type=float, default=0.2, help="Head dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument("--skip-final", action="store_true", help="Skip final model training")
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

    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPUs available: {len(gpus)}")
    if gpus:
        logger.info(f"GPU: {gpus[0]}")

    # Load all data
    logger.info("Loading all manifests...")
    df = merge_all_manifests(REPO_ROOT)
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
