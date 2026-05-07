"""Train CNN baseline on canonical data with two-stage training.

This script trains a MobileNetV2-based CNN on the canonical manifest data
using fixed train/val/test splits with two-stage training:
1. Warmup phase: backbone frozen, train head only
2. Fine-tune phase: backbone unfrozen, train end-to-end

Usage:
    python train_canonical_baseline.py --epochs 40 --warmup-epochs 8
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

# Add ml/src to path
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import keras
import tensorflow as tf
from sklearn.utils.class_weight import compute_sample_weight

from embedded_gauge_reading_tinyml.models import build_mobilenetv2_regression_model
from embedded_gauge_reading_tinyml.board_crop_compare import (
    load_rgb_image,
    resize_with_pad_rgb,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_split(split_path: Path, repo_root: Path) -> pd.DataFrame:
    """Load a split CSV and resolve image paths.

    Args:
        split_path: Path to the split CSV file.
        repo_root: Repository root for resolving relative paths.

    Returns:
        DataFrame with resolved image paths.
    """
    df = pd.read_csv(split_path)

    # Resolve image paths
    def resolve_path(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str(repo_root / path)

    df["image_path_resolved"] = df["image_path"].apply(resolve_path)

    return df


def create_value_bins(df: pd.DataFrame, bin_size: float = 5.0) -> pd.DataFrame:
    """Create value bins for weighted sampling.

    Args:
        df: Input DataFrame with 'value' column.
        bin_size: Size of each temperature bin in degrees C.

    Returns:
        DataFrame with added 'value_bin' column.
    """
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

    Args:
        df: DataFrame with 'value_bin' column.

    Returns:
        Array of sample weights.
    """
    bin_counts = df["value_bin"].value_counts()
    total = len(df)
    num_bins = len(bin_counts)

    # Compute weights: inverse frequency normalized by number of bins
    bin_weights = {}
    for bin_val, count in bin_counts.items():
        if count > 0:
            bin_weights[bin_val] = total / (num_bins * count)
        else:
            bin_weights[bin_val] = 0.0

    weights = df["value_bin"].map(bin_weights).values
    return weights.astype(np.float32)


def preprocess_image(
    image_path: str, target_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Load and preprocess an image for training.

    Args:
        image_path: Path to the image file.
        target_size: Target (height, width) for resizing.

    Returns:
        Preprocessed image as numpy array.
    """
    img = load_rgb_image(image_path)
    # Use tf.image.resize_with_pad directly for simple resize
    image_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
    resized = tf.image.resize_with_pad(
        tf.cast(image_tensor, tf.float32),
        target_size[0],  # height
        target_size[1],  # width
        method="bilinear",
    )
    img_resized = np.clip(np.rint(resized.numpy()), 0.0, 255.0).astype(np.uint8)
    # Normalize to [0, 1]
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

    Args:
        df: DataFrame with image paths and values.
        batch_size: Batch size for the dataset.
        shuffle: Whether to shuffle the dataset.
        use_weights: Whether to use sample weights.
        augment: Whether to apply train-time image augmentation.

    Returns:
        TensorFlow Dataset.
    """
    # Preload all images into memory for faster training
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

    # Create dataset from preloaded arrays
    if use_weights and "sample_weight" in df.columns:
        weights = df["sample_weight"].values[: len(images)].astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((images, values, weights))
        # Keep Keras' expected tuple structure: (x, y, sample_weight).
        # Returning ((x, y), w) makes Keras treat (x, y) as model inputs.
        dataset = dataset.map(lambda x, y, w: (x, y, w))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, values))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(images), reshuffle_each_iteration=True
        )

    if augment:
        # Strong photometric + geometric augmentation for train only.
        # Val/test stay clean. This is the biggest lever for small datasets.
        def _augment_strong(image: tf.Tensor) -> tf.Tensor:
            """Apply strong augmentation matching board camera reality."""
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


def train_canonical_baseline(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    repo_root: Path,
    output_dir: Path,
    epochs: int = 40,
    warmup_epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    fine_tune_lr: float = 1e-5,
    image_size: tuple[int, int] = (224, 224),
    use_weighted_sampling: bool = True,
    alpha: float = 1.0,
    dropout: float = 0.2,
    frozen_backbone: bool = False,
    use_augmentation: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a CNN baseline on canonical data with two-stage training.

    Args:
        train_csv: Path to training split CSV.
        val_csv: Path to validation split CSV.
        test_csv: Path to test split CSV.
        repo_root: Repository root directory.
        output_dir: Directory to save model artifacts.
        epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs with frozen backbone.
        batch_size: Batch size for training.
        learning_rate: Learning rate for warmup phase.
        fine_tune_lr: Learning rate for fine-tuning phase.
        image_size: Input image size (height, width).
        use_weighted_sampling: Whether to use weighted sampling by value bin.
        use_augmentation: Whether to apply train-time image augmentation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with training results and metrics.
    """
    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    # Load splits
    logger.info("Loading data splits...")
    train_df = load_split(train_csv, repo_root)
    val_df = load_split(val_csv, repo_root)
    test_df = load_split(test_csv, repo_root)

    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")

    # Compute sample weights for training data
    if use_weighted_sampling:
        logger.info("Computing sample weights by value bin frequency...")
        train_df = create_value_bins(train_df)
        train_df["sample_weight"] = compute_sample_weights(train_df)
        logger.info(
            f"Sample weight range: {train_df['sample_weight'].min():.3f} to {train_df['sample_weight'].max():.3f}"
        )

    # Create datasets
    logger.info("Creating TensorFlow datasets...")
    train_dataset = create_dataset(
        train_df,
        batch_size,
        shuffle=True,
        use_weights=use_weighted_sampling,
        augment=use_augmentation,
    )
    val_dataset = create_dataset(
        val_df, batch_size, shuffle=False, use_weights=False, augment=False
    )
    test_dataset = create_dataset(
        test_df, batch_size, shuffle=False, use_weights=False, augment=False
    )

    # Build model
    logger.info(
        f"Building MobileNetV2 regression model (alpha={alpha}, dropout={dropout})..."
    )
    model = build_mobilenetv2_regression_model(
        image_height=image_size[0],
        image_width=image_size[1],
        alpha=alpha,
        head_units=128,
        head_dropout=dropout,
        pretrained=True,
        value_min=-30.0,
        value_max=50.0,
    )

    # Compile model with Huber loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae", "mse"],
    )

    # Freeze backbone
    # MobileNetV2 layer name format: "mobilenetv2_{alpha:.2f}_224"
    base_model_name = f"mobilenetv2_{alpha:.2f}_224"
    try:
        base_model = model.get_layer(base_model_name)
    except ValueError:
        # Fallback: find the MobileNetV2 layer by class name
        for layer in model.layers:
            if "mobilenetv2" in layer.name.lower():
                base_model = layer
                base_model_name = layer.name
                break
        else:
            raise ValueError(
                f"Could not find MobileNetV2 layer in model. Layers: {[l.name for l in model.layers]}"
            )

    base_model.trainable = False

    # Shared callbacks for both phases
    checkpoint_path = output_dir / "best_weights.weights.h5"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    if frozen_backbone:
        # Train with frozen backbone for all epochs
        logger.info(f"\n=== Training with FROZEN backbone ({epochs} epochs) ===")
        logger.info(f"Model: MobileNetV2 alpha={alpha}, backbone frozen")

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Combine history for consistency
        history_warmup = history
        history_finetune = None
    else:
        # Phase 1: Warmup with frozen backbone
        logger.info(
            f"\n=== Phase 1: Warmup ({warmup_epochs} epochs with frozen backbone) ==="
        )

        # Train head only with callbacks (checkpoint best so far)
        history_warmup = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=warmup_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Phase 2: Fine-tuning with unfrozen backbone
        fine_tune_epochs = epochs - warmup_epochs
        logger.info(
            f"\n=== Phase 2: Fine-tuning ({fine_tune_epochs} epochs with unfrozen backbone) ==="
        )

        # Unfreeze backbone
        base_model.trainable = True

        # Recompile with lower learning rate for fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss=keras.losses.Huber(delta=1.0),
            metrics=["mae", "mse"],
        )

        # Continue training with callbacks
        history_finetune = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            initial_epoch=warmup_epochs,
            callbacks=callbacks,
            verbose=1,
        )

    # Evaluate on test set
    logger.info("\n=== Evaluating on test set ===")
    test_metrics = model.evaluate(test_dataset, return_dict=True, verbose=1)
    logger.info(f"Test metrics: {test_metrics}")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "canonical_baseline_model.keras"
    model.save(model_path)
    logger.info(f"Saved model to {model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    if history_finetune is not None:
        history_combined = {
            "warmup": {
                k: [float(v) for v in vals]
                for k, vals in history_warmup.history.items()
            },
            "finetune": {
                k: [float(v) for v in vals]
                for k, vals in history_finetune.history.items()
            },
        }
    else:
        history_combined = {
            "train": {
                k: [float(v) for v in vals]
                for k, vals in history_warmup.history.items()
            },
        }
    with open(history_path, "w") as f:
        json.dump(history_combined, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    # Save test predictions for analysis
    logger.info("Generating test predictions...")
    predictions = model.predict(test_dataset, verbose=1)
    test_df["prediction"] = predictions.flatten()
    test_df["abs_error"] = np.abs(test_df["prediction"] - test_df["value"])

    predictions_path = output_dir / "test_predictions.csv"
    test_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved test predictions to {predictions_path}")

    return {
        "model": model,
        "test_metrics": test_metrics,
        "test_df": test_df,
        "output_dir": output_dir,
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train CNN baseline on canonical data with two-stage training."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Path to training split CSV (default: ml/data/splits/canonical_split_v1_train.csv)",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="Path to validation split CSV (default: ml/data/splits/canonical_split_v1_val.csv)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Path to test split CSV (default: ml/data/splits/canonical_split_v1_test.csv)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (default: auto-detected)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for artifacts (default: ml/artifacts/canonical_baseline/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Total number of training epochs (default: 40)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=8,
        help="Number of warmup epochs with frozen backbone (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for warmup phase (default: 1e-4)",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-5,
        help="Learning rate for fine-tuning phase (default: 1e-5)",
    )
    parser.add_argument(
        "--no-weighted-sampling",
        action="store_true",
        help="Disable weighted sampling by value bin frequency",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="MobileNetV2 width multiplier (default: 1.0, use 0.35 for tiny)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for classification head (default: 0.2)",
    )
    parser.add_argument(
        "--frozen-backbone",
        action="store_true",
        help="Keep backbone frozen for entire training (no fine-tuning)",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable train-time image augmentation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Auto-detect paths
    if args.repo_root is None:
        script_dir = Path(__file__).resolve().parent
        args.repo_root = script_dir.parent.parent
    else:
        args.repo_root = args.repo_root.resolve()

    if args.train_csv is None:
        args.train_csv = (
            args.repo_root / "ml" / "data" / "splits" / "canonical_split_v1_train.csv"
        )
    if args.val_csv is None:
        args.val_csv = (
            args.repo_root / "ml" / "data" / "splits" / "canonical_split_v1_val.csv"
        )
    if args.test_csv is None:
        args.test_csv = (
            args.repo_root / "ml" / "data" / "splits" / "canonical_split_v1_test.csv"
        )
    if args.output_dir is None:
        args.output_dir = args.repo_root / "ml" / "artifacts" / "canonical_baseline"

    logger.info(f"Repository root: {args.repo_root}")
    logger.info(f"Train CSV: {args.train_csv}")
    logger.info(f"Val CSV: {args.val_csv}")
    logger.info(f"Test CSV: {args.test_csv}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs} (warmup: {args.warmup_epochs})")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(
        f"Learning rates: warmup={args.learning_rate}, fine-tune={args.fine_tune_lr}"
    )
    logger.info(f"Weighted sampling: {not args.no_weighted_sampling}")
    logger.info(f"MobileNetV2 alpha: {args.alpha}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Frozen backbone: {args.frozen_backbone}")
    logger.info(f"Train augmentation: {not args.no_augmentation}")

    # Run training
    try:
        results = train_canonical_baseline(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            fine_tune_lr=args.fine_tune_lr,
            use_weighted_sampling=not args.no_weighted_sampling,
            alpha=args.alpha,
            dropout=args.dropout,
            frozen_backbone=args.frozen_backbone,
            use_augmentation=not args.no_augmentation,
            seed=args.seed,
        )

        print("\n=== Training Complete ===")
        print(f"Test MAE: {results['test_metrics'].get('mae', 'N/A'):.4f}")
        print(f"Test MSE: {results['test_metrics'].get('mse', 'N/A'):.4f}")
        print(f"Model saved to: {results['output_dir']}")

        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
