"""Train the polar needle-segmentation model.

This script trains a CNN that reads analog gauges by:
  1. Projecting the image into polar coordinates.
  2. Segmenting the needle in polar space.
  3. Converting the detected needle angle to temperature.

The model is trained with both mask supervision (when available) and
temperature supervision, making it robust even with limited labeled data.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Add ml/src to path.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.models import build_mobilenetv2_tiny_regression_model
from embedded_gauge_reading_tinyml.polar_model import (
    PolarAngleToTemperature,
    build_polar_needle_segmentation_model,
    build_polar_tiny_model,
)
from embedded_gauge_reading_tinyml.polar_projection import (
    augment_polar_image,
    polar_project_image_path,
)
from embedded_gauge_reading_tinyml.board_crop_compare import load_rgb_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
VALUE_MIN = -30.0
VALUE_MAX = 50.0


def normalize_path(path_str: str, repo_root: Path) -> str:
    normalized = path_str.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        try:
            path = path.relative_to(repo_root)
        except ValueError:
            pass
    return path.as_posix()


def resolve_full_path(normalized_path: str, repo_root: Path) -> Path:
    return repo_root / normalized_path


def load_manifest(file_path: Path, repo_root: Path) -> pd.DataFrame | None:
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    if "label" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"label": "value"})
    if "image_path" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "image_path"})
    df["image_path"] = df["image_path"].apply(lambda p: normalize_path(p, repo_root))
    df["image_path_resolved"] = df["image_path"].apply(
        lambda p: str(resolve_full_path(p, repo_root))
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["image_path_resolved"].apply(lambda p: Path(p).exists())]
    return df.reset_index(drop=True)


def merge_all_manifests(repo_root: Path) -> pd.DataFrame:
    """Merge all available manifests with priority-based deduplication."""
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
        df["source"] = filename.replace(".csv", "")
        df["priority"] = priority
        df_new = df[~df["image_path"].isin(seen_paths)].copy()
        seen_paths.update(df["image_path"].tolist())
        logger.info(f"Loaded {filename}: {len(df)} rows, {len(df_new)} new")
        all_rows.append(df_new)

    # Also load polar mask manifest if available.
    polar_mask_path = PROJECT_ROOT / "artifacts" / "polar_masks" / "manifest.csv"
    if polar_mask_path.exists():
        df_polar = pd.read_csv(polar_mask_path)
        if len(df_polar) > 0:
            # Resolve paths relative to repo root (paths in manifest already include ml/artifacts prefix).
            df_polar["image_path_resolved"] = df_polar["image_path"].apply(
                lambda p: str(REPO_ROOT / p)
            )
            df_polar["mask_path_resolved"] = df_polar["mask_path"].apply(
                lambda p: str(REPO_ROOT / p)
            )
            df_polar["value"] = pd.to_numeric(df_polar["value"], errors="coerce")
            df_polar = df_polar.dropna(subset=["value"])
            df_polar["source"] = "polar_masks"
            df_polar["priority"] = 7  # Highest priority for labeled masks.
            # Only add rows whose original image isn't already in the dataset.
            df_new = df_polar[~df_polar["original_path"].isin(seen_paths)].copy()
            seen_paths.update(df_polar["original_path"].tolist())
            logger.info(
                f"Loaded polar_masks manifest: {len(df_polar)} rows, {len(df_new)} new"
            )
            all_rows.append(df_new)

    if not all_rows:
        raise ValueError("No manifests could be loaded")

    merged = pd.concat(all_rows, ignore_index=True)
    logger.info(f"Merged dataset: {len(merged)} total images")
    logger.info(
        f"Value range: {merged['value'].min():.1f} to {merged['value'].max():.1f}"
    )
    return merged


def create_value_bins(df: pd.DataFrame, bin_size: float = 10.0) -> pd.DataFrame:
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
    bin_counts = df["value_bin"].value_counts()
    total = len(df)
    num_bins = len(bin_counts)

    bin_weights = {}
    for bin_val, count in bin_counts.items():
        if count > 0:
            weight = min(total / (num_bins * count), 5.0)
            bin_weights[bin_val] = weight
        else:
            bin_weights[bin_val] = 0.0

    weights = df["value_bin"].map(bin_weights).values
    return weights.astype(np.float32)


def preprocess_polar_image(image_path: str, polar_size: int = 224) -> np.ndarray:
    """Load an image and project it into polar coordinates.

    The polar projection turns the circular gauge face into a rectangular
    angle-versus-radius image, making the needle a vertical line feature.
    """
    import cv2
    from PIL import Image

    path = Path(image_path)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    center = (float(width) * 0.5, float(height) * 0.5)
    max_radius = float(min(height, width)) * 0.5

    # OpenCV warpPolar: angle across horizontal, radius down vertical.
    polar = cv2.warpPolar(
        img,
        (polar_size, polar_size),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS,
    )
    if polar.shape[0] != polar_size or polar.shape[1] != polar_size:
        polar = cv2.resize(polar, (polar_size, polar_size), interpolation=cv2.INTER_LINEAR)

    return polar.astype(np.float32) / 255.0


def create_polar_dataset(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool = False,
    use_weights: bool = False,
    augment: bool = False,
    has_masks: bool = False,
    polar_size: int = 224,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset for polar needle training.

    Precomputes polar projections for all images and loads them into memory.
    """
    logger.info(f"Loading {len(df)} polar projections into memory...")
    polar_images = []
    values = []
    masks = [] if has_masks else None

    for idx, row in df.iterrows():
        try:
            polar_img = preprocess_polar_image(row["image_path_resolved"], polar_size)
            polar_images.append(polar_img)
            values.append(float(row["value"]))

            if has_masks and "mask_path" in row and pd.notna(row["mask_path"]):
                mask_path = str(resolve_full_path(row["mask_path"], REPO_ROOT))
                if Path(mask_path).exists():
                    import cv2
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = cv2.resize(mask, (polar_size, polar_size))
                        mask = mask.astype(np.float32) / 255.0
                        # Ensure shape is (H, W, 1)
                        if len(mask.shape) == 2:
                            mask = mask[..., np.newaxis]
                        masks.append(mask)
                    else:
                        masks.append(np.zeros((polar_size, polar_size, 1), dtype=np.float32))
                else:
                    masks.append(np.zeros((polar_size, polar_size, 1), dtype=np.float32))
            elif has_masks:
                masks.append(np.zeros((polar_size, polar_size, 1), dtype=np.float32))
        except Exception as e:
            logger.warning(f"Failed to load {row['image_path_resolved']}: {e}")
            continue

    polar_images = np.array(polar_images, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    logger.info(f"Successfully loaded {len(polar_images)} polar projections")

    if has_masks:
        masks_arr = np.array(masks, dtype=np.float32)
        if use_weights and "sample_weight" in df.columns:
            weights = df["sample_weight"].values[: len(polar_images)].astype(np.float32)
            dataset = tf.data.Dataset.from_tensor_slices((polar_images, values, masks_arr, weights))
            dataset = dataset.map(
                lambda x, y, m, w: (
                    x,
                    {"gauge_value": y, "needle_mask": m},  # Dict order matches model.output_names
                    {"gauge_value": w, "needle_mask": w},
                )
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices((polar_images, values, masks_arr))
            dataset = dataset.map(
                lambda x, y, m: (
                    x,
                    {"gauge_value": y, "needle_mask": m},  # Dict order matches model.output_names
                )
            )
    else:
        if use_weights and "sample_weight" in df.columns:
            weights = df["sample_weight"].values[: len(polar_images)].astype(np.float32)
            dataset = tf.data.Dataset.from_tensor_slices((polar_images, values, weights))
            dataset = dataset.map(
                lambda x, y, w: (
                    x,
                    {"gauge_value": y},
                    {"gauge_value": w},
                )
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices((polar_images, values))
            dataset = dataset.map(lambda x, y: (x, {"gauge_value": y}))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(polar_images), reshuffle_each_iteration=True
        )

    if augment:
        # Simplified augmentation: only photometric, no geometric transforms
        # that could conflict with dict targets in tf.data pipeline.
        # Apply augmentation via a lambda with *rest so it works for both
        # 2-tuple (img, target) and 3-tuple (img, target, weights) datasets.
        def _augment_image(img):
            img = tf.image.random_brightness(img, max_delta=0.10)
            img = tf.image.random_contrast(img, lower=0.88, upper=1.12)
            img = tf.clip_by_value(img, 0.0, 1.0)
            img = img + tf.random.normal(tf.shape(img), stddev=0.010)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img

        dataset = dataset.map(
            lambda img, *rest: (_augment_image(img),) + rest,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def train_polar_model(
    df: pd.DataFrame,
    output_dir: Path,
    epochs: int = 60,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    mask_loss_weight: float = 0.5,
    seed: int = 42,
    tiny: bool = False,
) -> dict[str, Any]:
    """Train the polar needle-segmentation model.

    Args:
        df: DataFrame with image_path and value columns.
        output_dir: Where to save model artifacts.
        epochs: Total training epochs.
        batch_size: Batch size.
        learning_rate: Initial learning rate.
        mask_loss_weight: Weight for mask loss vs temperature loss.
        seed: Random seed.
        tiny: If True, use the tiny model variant.

    Returns:
        Dictionary with training results and metrics.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    # Create stratified split.
    broad_bins = (df["value"] / 10).astype(int)
    bin_counts = broad_bins.value_counts()
    if (bin_counts < 2).any():
        logger.warning("Some bins have <2 samples, using random split")
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)
    else:
        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=seed, stratify=broad_bins
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=seed,
            stratify=(temp_df["value"] / 10).astype(int),
        )

    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")

    # Compute sample weights.
    train_df = create_value_bins(train_df, bin_size=10.0)
    train_df["sample_weight"] = compute_sample_weights(train_df)
    logger.info(
        f"Sample weight range: {train_df['sample_weight'].min():.3f} to {train_df['sample_weight'].max():.3f}"
    )

    # Check if masks are available.
    has_masks = "mask_path_resolved" in df.columns and df["mask_path_resolved"].notna().any()
    if has_masks:
        logger.info(f"Masks available for {df['mask_path_resolved'].notna().sum()} samples")
    else:
        logger.info("No masks available, training with temperature supervision only")

    # Create datasets.
    train_ds = create_polar_dataset(
        train_df, batch_size, shuffle=True, use_weights=True, augment=True, has_masks=has_masks, polar_size=IMAGE_SIZE
    )
    val_ds = create_polar_dataset(
        val_df, batch_size, shuffle=False, use_weights=False, augment=False, has_masks=has_masks, polar_size=IMAGE_SIZE
    )
    test_ds = create_polar_dataset(
        test_df, batch_size, shuffle=False, use_weights=False, augment=False, has_masks=has_masks, polar_size=IMAGE_SIZE
    )

    # Build model.
    if tiny:
        logger.info("Building polar TINY model (base_filters=16, depth=3)...")
        model = build_polar_tiny_model(
            polar_size=IMAGE_SIZE,
        )
    else:
        logger.info("Building polar needle-segmentation model...")
        model = build_polar_needle_segmentation_model(
            polar_size=IMAGE_SIZE,
            base_filters=32,
            depth=4,
            dropout_rate=0.1,
        )

    # Count parameters.
    total_params = model.count_params()
    trainable_params = sum(
        [keras.backend.count_params(w) for w in model.trainable_weights]
    )
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    # Compile with combined loss.
    if mask_loss_weight > 0.0:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                "gauge_value": keras.losses.MeanSquaredError(),
                "needle_mask": keras.losses.BinaryCrossentropy(),
            },
            loss_weights={
                "gauge_value": 1.0,
                "needle_mask": mask_loss_weight,
            },
            metrics={
                "gauge_value": ["mae"],
            },
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                "gauge_value": keras.losses.MeanSquaredError(),
            },
            metrics={
                "gauge_value": ["mae"],
            },
        )

    # Callbacks.
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_weights.weights.h5"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_gauge_value_mae",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_gauge_value_mae",
            mode="min",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_gauge_value_mae",
            mode="min",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Train.
    logger.info(f"\n=== Training polar model ({epochs} epochs) ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate.
    logger.info("\n=== Evaluating on test set ===")
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    logger.info(f"Test metrics: {test_metrics}")

    # Save model.
    model.save(output_dir / "model.keras")
    logger.info(f"Saved model to {output_dir / 'model.keras'}")

    # Save history.
    with open(output_dir / "history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )

    # Save predictions.
    predictions = model.predict(test_ds, verbose=1)
    gauge_preds = predictions["gauge_value"].flatten()
    test_df = test_df.copy()
    test_df["prediction"] = gauge_preds
    test_df["abs_error"] = np.abs(test_df["prediction"] - test_df["value"])
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Detailed metrics.
    errors = test_df["abs_error"].values
    hard_mask = (test_df["value"] <= -20) | (test_df["value"] >= 40)
    hard_errors = errors[hard_mask] if hard_mask.any() else np.array([])

    metrics = {
        "test_mae": float(np.mean(errors)),
        "test_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "test_max_error": float(np.max(errors)),
        "test_median_error": float(np.median(errors)),
        "test_std": float(np.std(errors)),
        "test_pct_under_5c": float(np.mean(errors < 5.0) * 100),
        "test_hard_mae": float(np.mean(hard_errors)) if len(hard_errors) > 0 else None,
        "test_hard_max": float(np.max(hard_errors)) if len(hard_errors) > 0 else None,
        "predicted_std": float(np.std(gauge_preds)),
        "correlation": float(np.corrcoef(test_df["value"], gauge_preds)[0, 1])
        if len(gauge_preds) > 1
        else 0.0,
    }

    logger.info("\n=== Final Metrics ===")
    for key, val in metrics.items():
        if val is not None:
            logger.info(f"  {key}: {val:.4f}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "test_df": test_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train polar needle-segmentation model for gauge reading."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training" / "polar_needle_model",
        help="Output directory for model artifacts.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV to use instead of merging all.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--mask-loss-weight",
        type=float,
        default=0.0,
        help="Weight for mask loss (0 = temperature only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use the tiny model variant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.manifest is not None:
        df = load_manifest(args.manifest, REPO_ROOT)
        if df is None:
            raise ValueError(f"Manifest not found: {args.manifest}")
    else:
        df = merge_all_manifests(REPO_ROOT)

    logger.info(f"Training with {len(df)} samples")

    train_polar_model(
        df=df,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mask_loss_weight=args.mask_loss_weight,
        seed=args.seed,
        tiny=args.tiny,
    )


if __name__ == "__main__":
    main()
