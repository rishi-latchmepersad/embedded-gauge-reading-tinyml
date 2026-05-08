#!/usr/bin/env python3
"""Train tiny MobileNetV2 (alpha=0.35) with minimal head to prevent overfitting.

The full MobileNetV2 (alpha=1.0) with 256-unit head has ~3.5M parameters,
which massively overfits on 141 images. The tiny model has ~1.2M parameters
and is designed for STM32N6 deployment.

Usage:
    cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
    poetry run python scripts/train_tiny_model.py \
        --output-dir artifacts/training/tiny_model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Add ml/src to path
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.models import build_mobilenetv2_tiny_regression_model
from embedded_gauge_reading_tinyml.board_crop_compare import load_rgb_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_SIZE = 224


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


def load_scalar_manifest(file_path: Path, repo_root: Path) -> pd.DataFrame:
    """Load a prebuilt scalar manifest and normalize path columns."""
    df = pd.read_csv(file_path)
    if "image_path" not in df.columns:
        raise ValueError(f"Manifest is missing image_path column: {file_path}")
    if "value" not in df.columns:
        raise ValueError(f"Manifest is missing value column: {file_path}")

    df = df.copy()
    df["image_path"] = df["image_path"].apply(
        lambda p: normalize_path(str(p), repo_root)
    )
    if "resolved_path" in df.columns:
        df["image_path_resolved"] = df["resolved_path"].astype(str)
    else:
        df["image_path_resolved"] = df["image_path"].apply(
            lambda p: str(resolve_full_path(p, repo_root))
        )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["image_path_resolved"].apply(lambda p: Path(p).exists())]
    return df.reset_index(drop=True)


def merge_all_manifests(repo_root: Path) -> pd.DataFrame:
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

    if not all_rows:
        raise ValueError("No manifests could be loaded")

    merged = pd.concat(all_rows, ignore_index=True)
    logger.info(f"Merged dataset: {len(merged)} total images")
    logger.info(
        f"Value range: {merged['value'].min():.1f} to {merged['value'].max():.1f}"
    )
    return merged


def load_training_dataframe(
    repo_root: Path,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Load the training dataframe from a manifest or the legacy merge path."""
    if manifest_path is not None:
        logger.info(f"Loading scalar manifest: {manifest_path}")
        return load_scalar_manifest(manifest_path, repo_root)
    logger.info("Loading all manifests using the legacy merge path...")
    return merge_all_manifests(repo_root)


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


def preprocess_image(
    image_path: str, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
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

        def _augment(image: tf.Tensor) -> tf.Tensor:
            image_shape = tf.shape(image)
            image_h = image_shape[0]
            image_w = image_shape[1]

            scale = tf.random.uniform([], minval=0.90, maxval=1.0, dtype=tf.float32)
            crop_h = tf.maximum(
                2, tf.cast(tf.cast(image_h, tf.float32) * scale, tf.int32)
            )
            crop_w = tf.maximum(
                2, tf.cast(tf.cast(image_w, tf.float32) * scale, tf.int32)
            )
            image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
            image = tf.image.resize(image, [image_h, image_w])

            image = tf.image.random_brightness(image, max_delta=0.20)
            image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
            image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
            image = tf.clip_by_value(image, 0.0, 1.0)

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

            image = image + tf.random.normal(tf.shape(image), stddev=0.015)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image

        if use_weights:
            dataset = dataset.map(
                lambda x, y, w: (_augment(x), y, w),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            dataset = dataset.map(
                lambda x, y: (_augment(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def train_tiny_model(
    df: pd.DataFrame,
    output_dir: Path,
    epochs: int = 40,
    warmup_epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    fine_tune_lr: float = 1e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train tiny MobileNetV2 with a frozen warmup and fine-tuning phase."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    # Create split: 70/15/15
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

    # Compute sample weights
    train_df = create_value_bins(train_df, bin_size=10.0)
    train_df["sample_weight"] = compute_sample_weights(train_df)
    logger.info(
        f"Sample weight range: {train_df['sample_weight'].min():.3f} to {train_df['sample_weight'].max():.3f}"
    )

    # Create datasets
    train_ds = create_dataset(
        train_df, batch_size, shuffle=True, use_weights=True, augment=True
    )
    val_ds = create_dataset(
        val_df, batch_size, shuffle=False, use_weights=False, augment=False
    )
    test_ds = create_dataset(
        test_df, batch_size, shuffle=False, use_weights=False, augment=False
    )

    # Build TINY model
    logger.info("Building tiny MobileNetV2 (alpha=0.35, head=64, dropout=0.15)...")
    model = build_mobilenetv2_tiny_regression_model(
        image_height=IMAGE_SIZE,
        image_width=IMAGE_SIZE,
        pretrained=True,
        backbone_trainable=False,  # Frozen
    )

    # Find the backbone so we can unfreeze it after the head has warmed up.
    try:
        backbone = model.get_layer("mobilenetv2_0.35_224")
    except ValueError:
        backbone = next(
            layer for layer in model.layers if "mobilenetv2" in layer.name.lower()
        )

    # Compile with MSE
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=["mae", "mse"],
    )

    # Count parameters
    total_params = model.count_params()
    trainable_params = sum(
        [keras.backend.count_params(w) for w in model.trainable_weights]
    )
    non_trainable_params = total_params - trainable_params
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    logger.info(f"Non-trainable params: {non_trainable_params:,}")

    # Callbacks
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_weights.weights.h5"

    callbacks = [
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

    warmup_epochs = max(0, min(warmup_epochs, epochs))
    if warmup_epochs > 0:
        # Warm up the regression head before letting the backbone adapt.
        logger.info(
            f"\n=== Training tiny model warmup ({warmup_epochs} epochs, frozen backbone) ==="
        )
        history_warmup = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            callbacks=[],
            verbose=1,
        )
    else:
        history_warmup = keras.callbacks.History()
        history_warmup.history = {}

    # Fine-tune end-to-end with a lower learning rate so the pretrained
    # backbone can adapt without blowing up on the larger manifest.
    backbone.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss=keras.losses.MeanSquaredError(),
        metrics=["mae", "mse"],
    )
    logger.info(
        f"\n=== Training tiny model fine-tune ({epochs - warmup_epochs} epochs, unfrozen backbone) ==="
    )
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Merge history from both phases for later analysis.
    merged_history: dict[str, list[float]] = {}
    for key in set(history_warmup.history) | set(history_finetune.history):
        warmup_vals = [float(v) for v in history_warmup.history.get(key, [])]
        finetune_vals = [float(v) for v in history_finetune.history.get(key, [])]
        merged_history[key] = warmup_vals + finetune_vals
    history_finetune.history = merged_history
    history = history_finetune

    # Evaluate
    logger.info("\n=== Evaluating on test set ===")
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    logger.info(f"Test metrics: {test_metrics}")

    # Save model
    model.save(output_dir / "model.keras")
    logger.info(f"Saved model to {output_dir / 'model.keras'}")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )

    # Save predictions
    predictions = model.predict(test_ds, verbose=1).flatten()
    test_df = test_df.copy()
    test_df["prediction"] = predictions
    test_df["abs_error"] = np.abs(test_df["prediction"] - test_df["value"])
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Detailed metrics
    errors = test_df["abs_error"].values
    hard_mask = (test_df["value"] <= -20) | (test_df["value"] >= 40)
    metrics = {
        "test_mae": float(np.mean(errors)),
        "test_rmse": float(np.sqrt(np.mean(errors**2))),
        "test_max_error": float(np.max(errors)),
        "test_median_error": float(np.median(errors)),
        "test_hard_mae": float(np.mean(errors[hard_mask])) if hard_mask.any() else 0.0,
        "test_pct_under_5c": float(np.mean(errors < 5.0)),
        "test_pct_under_3c": float(np.mean(errors < 3.0)),
        "test_pct_under_1c": float(np.mean(errors < 1.0)),
        "predicted_mean": float(np.mean(predictions)),
        "predicted_std": float(np.std(predictions)),
        "true_mean": float(np.mean(test_df["value"].values)),
        "true_std": float(np.std(test_df["value"].values)),
        "correlation": float(np.corrcoef(test_df["value"], predictions)[0, 1]),
    }

    logger.info(f"\nDetailed Test Metrics:")
    for k, v in metrics.items():
        if "pct" in k:
            logger.info(f"  {k}: {v*100:.1f}%")
        else:
            logger.info(f"  {k}: {v:.2f}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {"model": model, "metrics": metrics, "output_dir": output_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny MobileNetV2 model")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional prebuilt scalar manifest CSV to train from.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Total epochs")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=8,
        help="Warmup epochs with the backbone frozen.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-5,
        help="Learning rate used for the fine-tuning phase.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPUs available: {len(gpus)}")

    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    df = load_training_dataframe(REPO_ROOT, manifest_path=manifest_path)
    df.to_csv(output_dir / "merged_dataset.csv", index=False)

    train_tiny_model(
        df,
        output_dir=output_dir,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        fine_tune_lr=args.fine_tune_lr,
        seed=args.seed,
    )

    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
