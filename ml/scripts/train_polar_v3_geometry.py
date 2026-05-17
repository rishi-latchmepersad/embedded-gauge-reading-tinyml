#!/usr/bin/env python3
"""Train polar needle-segmentation model with EXPLICIT geometry supervision.

This is the key improvement over previous polar attempts:
- We use the 352 CVAT samples that have center + tip + dial labels.
- We compute exact needle angles and generate exact needle masks in polar space.
- The UNet learns to segment the needle with strong pixel-level supervision.
- Temperature is derived from the predicted mask angle via the known calibration.

For unlabelled images (board captures), we use image center as approximate center.
The model is trained with center jitter to be robust to small center errors.
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
from embedded_gauge_reading_tinyml.polar_projection import (
    augment_polar_image,
    polar_project_image,
)
from embedded_gauge_reading_tinyml.polar_model import (
    build_polar_needle_segmentation_model,
    build_polar_tiny_model,
    PolarAngleToTemperature,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

POLAR_SIZE = 224
VALUE_MIN = -30.0
VALUE_MAX = 50.0
MIN_ANGLE_DEG = 135.0
SWEEP_DEG = 270.0


def preprocess_image_for_polar(
    image_path: str,
    center_xy: tuple[float, float],
    dial_radius: float,
    target_size: int = 224,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Load image, crop around dial, resize, return resized image + scaled center + max_radius.

    The crop is a square 2.5x dial_radius centered on the dial center.
    After resize to target_size, center is at (target_size/2, target_size/2).
    """
    path = Path(image_path)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    cx, cy = center_xy
    crop_half = int(1.25 * dial_radius)  # 2.5x total diameter -> 1.25x radius each side

    # Ensure crop is within bounds
    x1 = max(0, int(cx) - crop_half)
    y1 = max(0, int(cy) - crop_half)
    x2 = min(w, int(cx) + crop_half)
    y2 = min(h, int(cy) + crop_half)

    # Make square crop
    crop_size = min(x2 - x1, y2 - y1)
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop for {image_path}")

    # Resize to target_size x target_size
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # New center in resized image
    new_cx = target_size / 2.0
    new_cy = target_size / 2.0

    # Scale max_radius: original dial_radius mapped to resized space
    scale = target_size / crop_size if crop_size > 0 else 1.0
    new_radius = dial_radius * scale * 1.2  # slight margin

    return resized, (new_cx, new_cy), new_radius


def angle_deg_from_sample(sample: Sample, spec: GaugeSpec) -> float:
    """Compute needle angle in degrees [0, 360) from CVAT labels."""
    raw_angle = needle_angle_clockwise_rad(sample)
    # Convert to degrees, normalize to [0, 360)
    angle_deg = math.degrees(raw_angle) % 360.0
    return angle_deg


def create_needle_mask_polar(
    polar_image: np.ndarray,
    needle_angle_deg: float,
    mask_sigma: float = 2.0,
) -> np.ndarray:
    """Create soft Gaussian needle mask in polar space."""
    height, width = polar_image.shape[:2]
    center_x = (needle_angle_deg / 360.0) * float(width)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    dist_sq = (xx - center_x) ** 2
    mask = np.exp(-dist_sq / (2.0 * mask_sigma**2))
    return mask[..., np.newaxis].astype(np.float32)


def generate_training_pair(
    sample: Sample,
    spec: GaugeSpec,
    center_jitter_px: float = 0.0,
    radius_scale_range: tuple[float, float] = (0.9, 1.1),
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate one training pair: (polar_image, needle_mask, temperature).

    Args:
        sample: CVAT sample with geometry labels.
        spec: Gauge calibration spec.
        center_jitter_px: Random pixel offset to apply to center for robustness.
        radius_scale_range: Random scale on max_radius.

    Returns:
        polar_image: (POLAR_SIZE, POLAR_SIZE, 3) float32
        needle_mask: (POLAR_SIZE, POLAR_SIZE, 1) float32
        temperature: scalar float
    """
    cx = sample.center.x + np.random.uniform(-center_jitter_px, center_jitter_px)
    cy = sample.center.y + np.random.uniform(-center_jitter_px, center_jitter_px)
    dial_radius = max(sample.dial.rx, sample.dial.ry)

    img, (new_cx, new_cy), new_radius = preprocess_image_for_polar(
        str(sample.image_path),
        (cx, cy),
        dial_radius,
        target_size=POLAR_SIZE,
    )

    # Random radius scaling
    radius_scale = np.random.uniform(*radius_scale_range)
    new_radius *= radius_scale

    # Polar project
    polar = polar_project_image(img, center_xy=(new_cx, new_cy), max_radius=new_radius, polar_size=POLAR_SIZE)

    # Needle angle
    angle_deg = angle_deg_from_sample(sample, spec)
    # Adjust angle if we did horizontal flip (not doing here, but keep for symmetry)
    # For now, just generate mask
    mask = create_needle_mask_polar(polar, angle_deg, mask_sigma=2.0)

    # Temperature label
    temp = needle_value(sample, spec, strict=False)

    return polar, mask, temp


def augment_training_pair(
    polar: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply augmentation in polar space. Horizontal shift = gauge rotation.

    Returns updated polar image, mask, and new angle.
    """
    # Horizontal shift amount (in pixels)
    shift_px = np.random.randint(-15, 16)  # ~+/-24 degrees max

    if shift_px != 0:
        polar = np.roll(polar, shift_px, axis=1)
        mask = np.roll(mask, shift_px, axis=1)
        # Update angle: shift right (positive) means angle increases
        angle_deg = (angle_deg + (shift_px / POLAR_SIZE) * 360.0) % 360.0

    # Photometric augmentations
    brightness = np.random.uniform(-0.1, 0.1)
    contrast = np.random.uniform(0.8, 1.2)
    polar = np.clip((polar + brightness - 0.5) * contrast + 0.5, 0.0, 1.0)

    # Random blur
    if np.random.rand() < 0.3:
        blur_sigma = np.random.uniform(0.5, 1.5)
        ksize = max(3, int(blur_sigma * 2) * 2 + 1)
        polar = cv2.GaussianBlur(polar, (ksize, ksize), blur_sigma)

    # Random noise
    if np.random.rand() < 0.3:
        noise_std = np.random.uniform(0.01, 0.03)
        polar = np.clip(polar + np.random.normal(0.0, noise_std, polar.shape), 0.0, 1.0)

    return polar, mask, angle_deg


def create_tf_dataset(
    samples: list[Sample],
    spec: GaugeSpec,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    """Create TensorFlow dataset from CVAT samples."""
    # Pre-generate all samples
    polar_images = []
    masks = []
    temps = []
    angles = []

    logger.info(f"Generating polar projections for {len(samples)} samples...")
    for sample in samples:
        try:
            polar, mask, temp = generate_training_pair(
                sample, spec, center_jitter_px=8.0 if augment else 0.0
            )
            angle_deg = angle_deg_from_sample(sample, spec)
            if augment:
                polar, mask, angle_deg = augment_training_pair(polar, mask, angle_deg)
            polar_images.append(polar)
            masks.append(mask)
            temps.append(temp)
            angles.append(angle_deg)
        except Exception as exc:
            logger.warning(f"Skip {sample.image_path}: {exc}")

    polar_images = np.array(polar_images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    temps = np.array(temps, dtype=np.float32)
    angles = np.array(angles, dtype=np.float32)

    logger.info(f"Successfully generated {len(polar_images)} training pairs")

    ds = tf.data.Dataset.from_tensor_slices((
        polar_images,
        {
            "needle_mask": masks,
            "gauge_value": temps,
        }
    ))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(polar_images), 500))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_hard_cases_df(repo_root: Path) -> pd.DataFrame | None:
    """Load hard cases manifest."""
    hard_path = PROJECT_ROOT / "data" / "hard_cases_plus_board30.csv"
    if not hard_path.exists():
        return None
    df = pd.read_csv(hard_path)
    if "label" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"label": "value"})
    if "image_path" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "image_path"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def preprocess_unlabelled_for_polar(image_path: str, polar_size: int = 224) -> np.ndarray:
    """Preprocess an unlabelled image (board capture) for polar projection.

    Uses image center as approximate dial center.
    """
    path = Path(image_path)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    # Resize to square if needed
    if h != polar_size or w != polar_size:
        img = cv2.resize(img, (polar_size, polar_size), interpolation=cv2.INTER_LINEAR)

    # Use image center
    center = (polar_size / 2.0, polar_size / 2.0)
    max_radius = polar_size * 0.45  # ~100 pixels for 224

    polar = polar_project_image(img, center_xy=center, max_radius=max_radius, polar_size=polar_size)
    return polar


def train_polar_geometry_supervised(
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
    tiny: bool = True,
    mask_loss_weight: float = 1.0,
    temp_loss_weight: float = 0.1,
) -> dict[str, Any]:
    """Train polar needle model with explicit mask supervision."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gauge spec
    specs = load_gauge_specs()
    spec = specs["littlegood_home_temp_gauge_c"]

    # Load CVAT samples
    samples = load_dataset()
    logger.info(f"Loaded {len(samples)} CVAT samples with geometry labels")

    # Split train/val
    train_samples, val_samples = train_test_split(
        samples, test_size=0.15, random_state=seed, stratify=None
    )
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Build datasets
    train_ds = create_tf_dataset(train_samples, spec, batch_size=batch_size, shuffle=True, augment=True)
    val_ds = create_tf_dataset(val_samples, spec, batch_size=batch_size, shuffle=False, augment=False)

    # Build model
    if tiny:
        model = build_polar_tiny_model(
            polar_size=POLAR_SIZE,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            min_angle_deg=MIN_ANGLE_DEG,
            sweep_deg=SWEEP_DEG,
        )
    else:
        model = build_polar_needle_segmentation_model(
            polar_size=POLAR_SIZE,
            base_filters=32,
            depth=4,
            dropout_rate=0.1,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            min_angle_deg=MIN_ANGLE_DEG,
            sweep_deg=SWEEP_DEG,
        )

    # Custom loss: mask BCE + temp MSE
    def mask_bce(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

    def temp_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "needle_mask": mask_bce,
            "gauge_value": temp_mse,
        },
        loss_weights={
            "needle_mask": mask_loss_weight,
            "gauge_value": temp_loss_weight,
        },
        metrics={
            "gauge_value": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    model.summary()

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
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_gauge_value_mae",
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    logger.info(f"\n=== Training polar geometry-supervised model ({epochs} epochs) ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on hard cases
    hard_df = load_hard_cases_df(REPO_ROOT)
    if hard_df is not None and len(hard_df) > 0:
        logger.info(f"\n=== Evaluating on {len(hard_df)} hard cases ===")
        polar_images = []
        for _, row in hard_df.iterrows():
            try:
                # Resolve path relative to repo root
                img_path = REPO_ROOT / row["image_path"]
                polar = preprocess_unlabelled_for_polar(str(img_path), POLAR_SIZE)
                polar_images.append(polar)
            except Exception as exc:
                logger.warning(f"Skip hard case {row['image_path']}: {exc}")
                polar_images.append(np.zeros((POLAR_SIZE, POLAR_SIZE, 3), dtype=np.float32))

        polar_images = np.array(polar_images, dtype=np.float32)
        predictions = model.predict(polar_images, verbose=1, batch_size=8)
        gauge_preds = predictions["gauge_value"].flatten()

        hard_df = hard_df.copy().reset_index(drop=True)
        hard_df["prediction"] = gauge_preds[:len(hard_df)]
        hard_df["abs_error"] = np.abs(hard_df["prediction"] - hard_df["value"])
        hard_df.to_csv(output_dir / "hard_case_predictions.csv", index=False)

        errors = hard_df["abs_error"].values
        metrics = {
            "hard_mae": float(np.mean(errors)),
            "hard_rmse": float(np.sqrt(np.mean(errors**2))),
            "hard_max_error": float(np.max(errors)),
            "hard_median_error": float(np.median(errors)),
            "hard_pct_under_5c": float(np.mean(errors < 5.0) * 100),
            "predicted_std": float(np.std(gauge_preds)),
            "correlation": (
                float(np.corrcoef(hard_df["value"], gauge_preds)[0, 1])
                if len(gauge_preds) > 1
                else 0.0
            ),
        }
        logger.info("\n=== Hard Case Metrics ===")
        for key, val in metrics.items():
            logger.info(f"  {key}: {val:.4f}")

        with open(output_dir / "hard_case_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        logger.warning("No hard cases found for evaluation")
        metrics = {}

    # Save model
    model.save(output_dir / "model.keras")
    logger.info(f"Saved model to {output_dir / 'model.keras'}")

    with open(output_dir / "history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train polar needle model with geometry supervision")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "training" / "polar_v3_geometry")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny", action="store_true", default=True)
    parser.add_argument("--no-tiny", action="store_true", dest="tiny", default=False)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--temp-loss-weight", type=float, default=0.1)
    args = parser.parse_args()

    train_polar_geometry_supervised(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        tiny=args.tiny,
        mask_loss_weight=args.mask_loss_weight,
        temp_loss_weight=args.temp_loss_weight,
    )


if __name__ == "__main__":
    main()



