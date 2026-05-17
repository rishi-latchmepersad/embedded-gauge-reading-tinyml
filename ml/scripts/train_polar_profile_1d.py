#!/usr/bin/env python3
"""Train a 1D polar-profile model for needle detection.

Key insight: in polar space, the needle becomes a vertical line. Summing vertically
gives a 1D angular profile where the needle creates a peak. A small 1D CNN can
learn to detect this peak robustly.

Architecture:
1. Input: raw RGB image
2. Polar projection (using image center as approximate dial center)
3. Vertical sum to get 1D profile
4. 1D convolutions to enhance needle peak / suppress noise
5. Softmax over profile positions
6. Expected angle -> temperature via known calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import cv2
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

from embedded_gauge_reading_tinyml.dataset import load_dataset, Sample
from embedded_gauge_reading_tinyml.gauge.processing import (
    load_gauge_specs,
    needle_angle_clockwise_rad,
    needle_value,
    GaugeSpec,
)
from embedded_gauge_reading_tinyml.polar_projection import polar_project_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

POLAR_SIZE = 224
VALUE_MIN = -30.0
VALUE_MAX = 50.0
MIN_ANGLE_DEG = 135.0
SWEEP_DEG = 270.0


def image_to_polar_profile(image: np.ndarray, polar_size: int = 224) -> np.ndarray:
    """Convert image to polar projection and sum vertically to get 1D profile."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    max_radius = min(h, w) * 0.45
    polar = polar_project_image(image, center_xy=center, max_radius=max_radius, polar_size=polar_size)
    # Sum over vertical dimension (radius) to get angular profile
    profile = np.sum(polar, axis=(0, 2))  # shape (polar_size,)
    # Normalize
    profile = profile / (np.max(profile) + 1e-8)
    return profile.astype(np.float32)


def create_needle_profile_label(angle_deg: float, polar_size: int = 224, sigma: float = 3.0) -> np.ndarray:
    """Create a Gaussian peak label at the needle angle position."""
    center_x = (angle_deg / 360.0) * polar_size
    xx = np.arange(polar_size, dtype=np.float32)
    profile = np.exp(-((xx - center_x) ** 2) / (2.0 * sigma ** 2))
    return profile.astype(np.float32)


class ExpectedValueLayer(keras.layers.Layer):
    """Compute expected value of a discrete distribution."""
    def call(self, inputs):
        probs, values = inputs
        return tf.reduce_sum(probs * values[tf.newaxis, :], axis=-1, keepdims=True)


def build_polar_profile_model(polar_size: int = 224, num_filters: int = 32) -> keras.Model:
    """Build a 1D CNN that detects needle peaks in polar profiles.

    Input: 1D profile of length polar_size
    Output: predicted angle in degrees (converted to temperature internally)
    """
    inputs = keras.Input(shape=(polar_size,), name="polar_profile")
    x = inputs

    # 1D convolutions to enhance peaks and suppress noise
    x = keras.layers.Reshape((polar_size, 1))(x)
    x = keras.layers.Conv1D(num_filters, kernel_size=5, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(num_filters, kernel_size=5, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    x = keras.layers.Conv1D(num_filters * 2, kernel_size=5, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(num_filters * 2, kernel_size=5, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)

    # Upsample back to original resolution
    x = keras.layers.Conv1DTranspose(num_filters, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv1DTranspose(num_filters, kernel_size=5, strides=2, padding="same", activation="relu")(x)

    # Predict profile logits (peak at needle position)
    logits = keras.layers.Conv1D(1, kernel_size=1, padding="same", name="profile_logits")(x)
    logits = keras.layers.Reshape((polar_size,))(logits)
    probs = keras.layers.Softmax(name="profile_probs")(logits)

    # Convert to temperature
    # Precompute temperature for each bin
    angles_deg = tf.range(polar_size, dtype=tf.float32) * (360.0 / polar_size)
    # Map angle to gauge value
    shifted = tf.math.floormod(angles_deg - MIN_ANGLE_DEG, 360.0)
    fractions = tf.clip_by_value(shifted / SWEEP_DEG, 0.0, 1.0)
    values = VALUE_MIN + fractions * (VALUE_MAX - VALUE_MIN)

    # Expected temperature
    gauge_value = ExpectedValueLayer(name="gauge_value")([probs, values])

    model = keras.Model(inputs=inputs, outputs=gauge_value, name="polar_profile_1d")
    return model


def preprocess_sample(sample: Sample, spec: GaugeSpec, augment: bool = False) -> tuple[np.ndarray, float, float]:
    """Load image, generate polar profile, compute angle and temp labels."""
    from PIL import Image
    img = np.asarray(Image.open(sample.image_path).convert("RGB"), dtype=np.uint8)

    # For augmentation, apply small rotation
    if augment:
        angle_jitter = np.random.uniform(-5, 5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_jitter, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    profile = image_to_polar_profile(img, POLAR_SIZE)
    temp = needle_value(sample, spec, strict=False)
    angle_deg = math.degrees(needle_angle_clockwise_rad(sample)) % 360.0
    return profile, temp, angle_deg


def create_training_data(samples: list[Sample], spec: GaugeSpec, augment: bool = False):
    profiles = []
    temps = []
    for s in samples:
        try:
            profile, temp, _ = preprocess_sample(s, spec, augment=augment)
            profiles.append(profile)
            temps.append(temp)
        except Exception as exc:
            logger.warning(f"Skip {s.image_path}: {exc}")
    return np.array(profiles, dtype=np.float32), np.array(temps, dtype=np.float32)


def preprocess_unlabelled(image_path: str, polar_size: int = 224) -> np.ndarray:
    from PIL import Image
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    return image_to_polar_profile(img, polar_size)


def train_polar_profile_model(
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = load_gauge_specs()
    spec = specs["littlegood_home_temp_gauge_c"]

    samples = load_dataset()
    logger.info(f"Loaded {len(samples)} CVAT samples")

    train_samples, val_samples = train_test_split(samples, test_size=0.15, random_state=seed)
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_profiles, train_temps = create_training_data(train_samples, spec, augment=True)
    val_profiles, val_temps = create_training_data(val_samples, spec, augment=False)

    logger.info(f"Train profiles: {train_profiles.shape}, temp range [{train_temps.min():.1f}, {train_temps.max():.1f}]")

    model = build_polar_profile_model(polar_size=POLAR_SIZE, num_filters=32)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_weights.weights.h5"),
            save_weights_only=True,
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mae",
            mode="min",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    logger.info(f"\n=== Training 1D polar profile model ({epochs} epochs) ===")
    history = model.fit(
        train_profiles, train_temps,
        validation_data=(val_profiles, val_temps),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on hard cases
    hard_path = PROJECT_ROOT / "data" / "hard_cases_plus_board30.csv"
    if hard_path.exists():
        logger.info(f"\n=== Evaluating on hard cases ===")
        df = pd.read_csv(hard_path)
        if "label" in df.columns and "value" not in df.columns:
            df = df.rename(columns={"label": "value"})
        if "image_path" not in df.columns and "path" in df.columns:
            df = df.rename(columns={"path": "image_path"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        profiles = []
        for _, row in df.iterrows():
            try:
                img_path = str(REPO_ROOT / row["image_path"])
                profile = preprocess_unlabelled(img_path, POLAR_SIZE)
                profiles.append(profile)
            except Exception as exc:
                logger.warning(f"Skip hard case: {exc}")
                profiles.append(np.zeros(POLAR_SIZE, dtype=np.float32))

        profiles = np.array(profiles, dtype=np.float32)
        preds = model.predict(profiles, verbose=1, batch_size=8).flatten()

        df = df.copy().reset_index(drop=True)
        df["prediction"] = preds
        df["abs_error"] = np.abs(df["prediction"] - df["value"])
        df.to_csv(output_dir / "hard_case_predictions.csv", index=False)

        errors = df["abs_error"].values
        metrics = {
            "hard_mae": float(np.mean(errors)),
            "hard_rmse": float(np.sqrt(np.mean(errors**2))),
            "hard_max_error": float(np.max(errors)),
            "hard_pct_under_5c": float(np.mean(errors < 5.0) * 100),
        }
        logger.info("\n=== Hard Case Metrics ===")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        with open(output_dir / "hard_case_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    model.save(output_dir / "model.keras")
    logger.info(f"Saved model to {output_dir / 'model.keras'}")

    with open(output_dir / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    return {"model": model, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1D polar profile needle detector")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "training" / "polar_profile_1d_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_polar_profile_model(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
