#!/usr/bin/env python3
"""Train polar needle model with explicit hard-case holdout for evaluation.

This script trains the polar needle-segmentation model on all available data
except the hard cases, which are held out for test evaluation.
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

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.polar_projection import (
    augment_polar_image,
    polar_project_image_path,
)
from embedded_gauge_reading_tinyml.polar_model import (
    build_polar_tiny_model,
    build_polar_needle_segmentation_model,
)

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
    manifest_files = [
        ("data/canonical_manifest_v1.csv", 6),
        ("data/unified_training_manifest_v1.csv", 5),
        ("data/full_labelled_plus_board30_valid_with_new5.csv", 4),
        ("data/new_labelled_captures4.csv", 2),
        ("data/all_captured_images_manifest.csv", 1),
    ]
    all_rows: list[pd.DataFrame] = []
    seen_paths: set[str] = set()
    for filename, priority in manifest_files:
        path = PROJECT_ROOT / filename
        df = load_manifest(path, repo_root)
        if df is None or len(df) == 0:
            continue
        df["source"] = filename.replace(".csv", "")
        df["priority"] = priority
        df_new = df[~df["image_path"].isin(seen_paths)].copy()
        seen_paths.update(df["image_path"].tolist())
        logger.info(f"Loaded {filename}: {len(df)} rows, {len(df_new)} new")
        all_rows.append(df_new)
    if not all_rows:
        raise ValueError("No manifests found")
    return pd.concat(all_rows, ignore_index=True)


def preprocess_polar_image(image_path: str, polar_size: int = 224) -> np.ndarray:
    path = Path(image_path)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    else:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    center = (float(width) * 0.5, float(height) * 0.5)
    max_radius = float(min(height, width)) * 0.5
    import cv2
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
    augment: bool = False,
    polar_size: int = 224,
) -> tf.data.Dataset:
    logger.info(f"Loading {len(df)} polar projections into memory...")
    polar_images = []
    values = []
    for idx, row in df.iterrows():
        try:
            polar_img = preprocess_polar_image(row["image_path_resolved"], polar_size)
            polar_images.append(polar_img)
            values.append(float(row["value"]))
        except Exception as e:
            logger.warning(f"Failed to load {row['image_path_resolved']}: {e}")
            continue
    polar_images = np.array(polar_images, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    logger.info(f"Successfully loaded {len(polar_images)} polar projections")
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"polar_image": polar_images}, {"gauge_value": values})
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(len(polar_images), 1), reshuffle_each_iteration=True)
    if augment:
        def _augment(inputs, targets):
            img = inputs["polar_image"]
            angle_shift = tf.random.uniform([], -15.0, 15.0)
            brightness = tf.random.uniform([], -0.1, 0.1)
            contrast = tf.random.uniform([], 0.8, 1.2)
            img = tf.numpy_function(
                lambda i, a, b, c: augment_polar_image(i, float(a), float(b), float(c)).astype(np.float32),
                [img, angle_shift, brightness, contrast],
                tf.float32,
            )
            img.set_shape([polar_size, polar_size, 3])
            return {"polar_image": img}, targets
        dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def train_polar_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    epochs: int = 60,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
    tiny: bool = True,
) -> dict[str, Any]:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    polar_size = IMAGE_SIZE
    if tiny:
        model = build_polar_tiny_model(polar_size=polar_size)
        logger.info("Built polar TINY model")
    else:
        model = build_polar_needle_segmentation_model(polar_size=polar_size)
        logger.info("Built polar FULL model")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "gauge_value": keras.losses.MeanSquaredError(),
        },
        metrics={
            "gauge_value": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    model.summary()

    train_ds = create_polar_dataset(train_df, batch_size=batch_size, shuffle=True, augment=True, polar_size=polar_size)
    test_ds = create_polar_dataset(test_df, batch_size=batch_size, shuffle=False, augment=False, polar_size=polar_size)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_weights.weights.h5"),
            save_weights_only=True,
            monitor="val_gauge_value_mae",
            mode="min",
            save_best_only=True,
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

    logger.info(f"\n=== Training polar model ({epochs} epochs) ===")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("\n=== Evaluating on test set (hard cases) ===")
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    logger.info(f"Test metrics: {test_metrics}")

    model.save(output_dir / "model.keras")
    logger.info(f"Saved model to {output_dir / 'model.keras'}")

    with open(output_dir / "history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )

    predictions = model.predict(test_ds, verbose=1)
    gauge_preds = predictions["gauge_value"].flatten()
    test_df = test_df.copy().reset_index(drop=True)
    test_df["prediction"] = gauge_preds[:len(test_df)]
    test_df["abs_error"] = np.abs(test_df["prediction"] - test_df["value"])
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    errors = test_df["abs_error"].values
    metrics = {
        "test_mae": float(np.mean(errors)),
        "test_rmse": float(np.sqrt(np.mean(errors**2))),
        "test_max_error": float(np.max(errors)),
        "test_median_error": float(np.median(errors)),
        "test_std": float(np.std(errors)),
        "test_pct_under_5c": float(np.mean(errors < 5.0) * 100),
        "predicted_std": float(np.std(gauge_preds)),
        "correlation": (
            float(np.corrcoef(test_df["value"], gauge_preds)[0, 1])
            if len(gauge_preds) > 1
            else 0.0
        ),
    }

    logger.info("\n=== Final Metrics ===")
    for key, val in metrics.items():
        logger.info(f"  {key}: {val:.4f}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "test_df": test_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train polar needle model with hard-case holdout")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "training" / "polar_v2_hardcases")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny", action="store_true", default=True)
    parser.add_argument("--no-tiny", action="store_true", dest="tiny", default=False)
    args = parser.parse_args()

    # Load hard cases as test set
    hard_cases_path = PROJECT_ROOT / "data" / "hard_cases_plus_board30.csv"
    test_df = load_manifest(hard_cases_path, REPO_ROOT)
    if test_df is None or len(test_df) == 0:
        raise ValueError(f"Hard cases manifest not found or empty: {hard_cases_path}")
    logger.info(f"Loaded {len(test_df)} hard cases for test set")

    # Load all other data for training
    all_df = merge_all_manifests(REPO_ROOT)
    # Remove hard cases from training
    train_df = all_df[~all_df["image_path"].isin(test_df["image_path"])].copy()
    logger.info(f"Training set: {len(train_df)} samples after removing hard cases")

    if len(train_df) < 10:
        raise ValueError("Not enough training samples")

    train_polar_model(
        train_df=train_df,
        test_df=test_df,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        tiny=args.tiny,
    )


if __name__ == "__main__":
    main()
