#!/usr/bin/env python3
"""Train a hybrid model that predicts both temperature (scalar) and needle angle (classification).

The key insight: the scalar head provides strong baseline performance, while the angle
classification head acts as a geometric regularizer that forces the backbone to learn
needle-orientation-sensitive features. At inference, we can ensemble both predictions.

The angle labels are computed exactly from CVAT center+tip geometry.
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
from embedded_gauge_reading_tinyml.board_crop_compare import (
    estimate_board_crop_from_rgb,
    load_rgb_image,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.presets import DEFAULT_CROP_PAD_RATIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
VALUE_MIN = -30.0
VALUE_MAX = 50.0
SWEEP_DEG = 270.0
NUM_ANGLE_BINS = 270  # 1 degree per bin


def angle_to_bin_index(angle_deg: float, spec: GaugeSpec) -> int:
    """Map needle angle in image coords to a bin index [0, NUM_ANGLE_BINS-1].

    The bins cover the gauge's calibrated sweep.
    """
    # Normalize angle into the actual calibrated sweep instead of assuming 270 degrees.
    shifted = (angle_deg - math.degrees(spec.min_angle_rad)) % 360.0
    sweep_deg = math.degrees(spec.sweep_rad)
    fraction = min(max(shifted / sweep_deg, 0.0), 1.0)
    bin_idx = int(fraction * (NUM_ANGLE_BINS - 1))
    return min(max(bin_idx, 0), NUM_ANGLE_BINS - 1)


def bin_index_to_temperature(bin_idx: int) -> float:
    """Convert bin index back to temperature."""
    fraction = bin_idx / (NUM_ANGLE_BINS - 1)
    span = VALUE_MAX - VALUE_MIN
    return VALUE_MIN + fraction * span


def _compute_dial_crop_box(sample: Sample, pad_ratio: float = DEFAULT_CROP_PAD_RATIO) -> tuple[float, float, float, float]:
    """Build the same dial-centered crop box used by the main training stack."""
    pad_x = sample.dial.rx * pad_ratio
    pad_y = sample.dial.ry * pad_ratio
    return (
        sample.dial.cx - sample.dial.rx - pad_x,
        sample.dial.cy - sample.dial.ry - pad_y,
        sample.dial.cx + sample.dial.rx + pad_x,
        sample.dial.cy + sample.dial.ry + pad_y,
    )


def preprocess_image(sample: Sample, target_size: int = 224) -> np.ndarray:
    """Load a labeled sample, crop around the dial, and resize to the model input."""
    img = load_rgb_image(str(sample.image_path))
    if img is None:
        raise ValueError(f"Unable to load image: {sample.image_path}")
    crop_box_xyxy = _compute_dial_crop_box(sample)
    img_resized = resize_with_pad_rgb(img, crop_box_xyxy, image_size=target_size)
    return img_resized.astype(np.float32) / 255.0


def preprocess_eval_image(image_path: str, target_size: int = 224) -> np.ndarray:
    """Load an inference image, use the board crop heuristic, and normalize it."""
    img = load_rgb_image(image_path)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")

    board_estimate = estimate_board_crop_from_rgb(img)
    if board_estimate is None:
        crop_box_xyxy = (0.0, 0.0, float(img.shape[1]), float(img.shape[0]))
    else:
        crop_box_xyxy = (
            float(board_estimate.crop_box.x_min),
            float(board_estimate.crop_box.y_min),
            float(board_estimate.crop_box.x_max),
            float(board_estimate.crop_box.y_max),
        )
    img_resized = resize_with_pad_rgb(img, crop_box_xyxy, image_size=target_size)
    return img_resized.astype(np.float32) / 255.0


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply training augmentation."""
    # Random rotation (small, to preserve gauge orientation roughly)
    angle = np.random.uniform(-8, 8)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Random brightness/contrast
    brightness = np.random.uniform(-0.08, 0.08)
    contrast = np.random.uniform(0.85, 1.15)
    rotated = np.clip((rotated + brightness - 0.5) * contrast + 0.5, 0.0, 1.0)

    # Random horizontal flip (need to adjust angle label)
    if np.random.rand() < 0.5:
        rotated = np.fliplr(rotated)
        flipped = True
    else:
        flipped = False

    return rotated, flipped


def build_hybrid_angle_scalar_model(
    image_size: int = 224,
    num_angle_bins: int = NUM_ANGLE_BINS,
    backbone: str = "mobilenet_v2",
    pretrained: bool = True,
    backbone_trainable: bool = True,
    head_units: int = 256,
    head_dropout: float = 0.3,
) -> keras.Model:
    """Build a hybrid model with scalar regression + angle classification heads."""
    if backbone == "mobilenet_v2":
        from embedded_gauge_reading_tinyml.models import _build_mobilenetv2_backbone
        inputs, features, base_model = _build_mobilenetv2_backbone(
            image_size, image_size,
            pretrained=pretrained,
            backbone_trainable=backbone_trainable,
            alpha=1.0,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # Global features
    x = keras.layers.GlobalAveragePooling2D(name="gap")(features)
    x = keras.layers.Dropout(head_dropout, name="dropout_1")(x)
    x = keras.layers.Dense(head_units, use_bias=False, name="dense_1")(x)
    x = keras.layers.LayerNormalization(name="ln_1")(x)
    x = keras.layers.Activation("swish", name="swish_1")(x)
    x = keras.layers.Dropout(head_dropout * 0.5, name="dropout_2")(x)

    # Scalar head: direct temperature regression with a bounded sigmoid output.
    scalar_linear = keras.layers.Dense(1, activation="sigmoid", name="scalar_sigmoid")(x)
    scalar_value = keras.layers.Rescaling(
        scale=VALUE_MAX - VALUE_MIN, offset=VALUE_MIN, name="scalar_value"
    )(scalar_linear)

    # Angle classification head
    angle_logits = keras.layers.Dense(num_angle_bins, name="angle_logits")(x)
    angle_probs = keras.layers.Softmax(name="angle_probs")(angle_logits)

    # Convert angle probs to temperature via expected value
    bin_centers = tf.constant(
        [bin_index_to_temperature(i) for i in range(num_angle_bins)],
        dtype=tf.float32,
    )
    angle_value = keras.layers.Dot(axes=-1, name="angle_value")([angle_probs, bin_centers[tf.newaxis, :]])

    model = keras.Model(
        inputs=inputs,
        outputs={
            "scalar_value": scalar_value,
            "angle_value": angle_value,
            "angle_logits": angle_logits,
        },
        name="hybrid_angle_scalar",
    )
    setattr(model, "_backbone", base_model)
    return model


def create_training_data(
    samples: list[Sample],
    spec: GaugeSpec,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create training arrays: images, temperatures, angle bin labels."""
    images = []
    temps = []
    angle_bins = []

    for s in samples:
        try:
            img = preprocess_image(s, IMAGE_SIZE)
            temp = needle_value(s, spec, strict=False)
            angle_rad = needle_angle_clockwise_rad(s)
            angle_deg = math.degrees(angle_rad) % 360.0
            bin_idx = angle_to_bin_index(angle_deg, spec)

            if augment:
                img, flipped = augment_image(img)
                if flipped:
                    # Flip horizontally: angle becomes 360 - angle
                    angle_deg = (360.0 - angle_deg) % 360.0
                    bin_idx = angle_to_bin_index(angle_deg, spec)

            images.append(img)
            temps.append(temp)
            angle_bins.append(bin_idx)
        except Exception as exc:
            logger.warning(f"Skip {s.image_path}: {exc}")

    return (
        np.array(images, dtype=np.float32),
        np.array(temps, dtype=np.float32),
        np.array(angle_bins, dtype=np.int32),
    )


def load_hard_cases() -> pd.DataFrame | None:
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


def train_hybrid_model(
    output_dir: Path,
    epochs: int = 80,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    seed: int = 42,
    scalar_loss_weight: float = 0.7,
    angle_loss_weight: float = 0.3,
) -> dict[str, Any]:
    """Train the hybrid angle+scalar model."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = load_gauge_specs()
    spec = specs["littlegood_home_temp_gauge_c"]

    # Load CVAT samples with geometry
    samples = load_dataset()
    logger.info(f"Loaded {len(samples)} CVAT samples")

    # Split train/val
    train_samples, val_samples = train_test_split(
        samples, test_size=0.15, random_state=seed
    )
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Build datasets
    train_imgs, train_temps, train_bins = create_training_data(train_samples, spec, augment=True)
    val_imgs, val_temps, val_bins = create_training_data(val_samples, spec, augment=False)

    logger.info(f"Train arrays: {train_imgs.shape}, temps range [{train_temps.min():.1f}, {train_temps.max():.1f}]")

    # One-hot angle labels
    train_angle_onehot = tf.keras.utils.to_categorical(train_bins, num_classes=NUM_ANGLE_BINS)
    val_angle_onehot = tf.keras.utils.to_categorical(val_bins, num_classes=NUM_ANGLE_BINS)

    train_ds = tf.data.Dataset.from_tensor_slices((
        train_imgs,
        {
            "scalar_value": train_temps,
            "angle_value": train_temps,  # Both heads supervised with temperature
            "angle_logits": train_angle_onehot,
        }
    )).shuffle(min(len(train_imgs), 500)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((
        val_imgs,
        {
            "scalar_value": val_temps,
            "angle_value": val_temps,
            "angle_logits": val_angle_onehot,
        }
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build model
    model = build_hybrid_angle_scalar_model(
        image_size=IMAGE_SIZE,
        num_angle_bins=NUM_ANGLE_BINS,
        backbone="mobilenet_v2",
        pretrained=True,
        backbone_trainable=True,
        head_units=256,
        head_dropout=0.3,
    )

    def scalar_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def angle_ce(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "scalar_value": scalar_mse,
            "angle_value": scalar_mse,
            "angle_logits": angle_ce,
        },
        loss_weights={
            "scalar_value": scalar_loss_weight,
            "angle_value": angle_loss_weight * 0.0,  # Don't use angle_value directly for loss
            "angle_logits": angle_loss_weight,
        },
        metrics={
            "scalar_value": [keras.metrics.MeanAbsoluteError(name="mae")],
            "angle_value": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_weights.weights.h5"),
            save_weights_only=True,
            monitor="val_scalar_value_mae",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_scalar_value_mae",
            mode="min",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_scalar_value_mae",
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    logger.info(f"\n=== Training hybrid model ({epochs} epochs) ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on hard cases
    hard_df = load_hard_cases()
    if hard_df is not None and len(hard_df) > 0:
        logger.info(f"\n=== Evaluating on {len(hard_df)} hard cases ===")
        hard_imgs = []
        for _, row in hard_df.iterrows():
            try:
                img_path = str(REPO_ROOT / row["image_path"])
                img = preprocess_eval_image(img_path, IMAGE_SIZE)
                hard_imgs.append(img)
            except Exception as exc:
                logger.warning(f"Skip hard case {row['image_path']}: {exc}")
                hard_imgs.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32))

        hard_imgs = np.array(hard_imgs, dtype=np.float32)
        preds = model.predict(hard_imgs, verbose=1, batch_size=8)
        scalar_preds = preds["scalar_value"].flatten()
        angle_preds = preds["angle_value"].flatten()

        # Ensemble: weighted average
        ensemble_preds = 0.7 * scalar_preds + 0.3 * angle_preds

        hard_df = hard_df.copy().reset_index(drop=True)
        hard_df["scalar_pred"] = scalar_preds
        hard_df["angle_pred"] = angle_preds
        hard_df["ensemble_pred"] = ensemble_preds
        hard_df["scalar_error"] = np.abs(hard_df["scalar_pred"] - hard_df["value"])
        hard_df["angle_error"] = np.abs(hard_df["angle_pred"] - hard_df["value"])
        hard_df["ensemble_error"] = np.abs(hard_df["ensemble_pred"] - hard_df["value"])
        hard_df.to_csv(output_dir / "hard_case_predictions.csv", index=False)

        for name, col in [("scalar", "scalar_error"), ("angle", "angle_error"), ("ensemble", "ensemble_error")]:
            errors = hard_df[col].values
            metrics = {
                f"{name}_mae": float(np.mean(errors)),
                f"{name}_rmse": float(np.sqrt(np.mean(errors**2))),
                f"{name}_max_error": float(np.max(errors)),
                f"{name}_pct_under_5c": float(np.mean(errors < 5.0) * 100),
            }
            logger.info(f"\n{name.upper()} results:")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}")

        with open(output_dir / "hard_case_metrics.json", "w") as f:
            json.dump({
                "scalar_mae": float(np.mean(hard_df["scalar_error"])),
                "angle_mae": float(np.mean(hard_df["angle_error"])),
                "ensemble_mae": float(np.mean(hard_df["ensemble_error"])),
            }, f, indent=2)

    model.save(output_dir / "model.keras")
    logger.info(f"Saved model to {output_dir / 'model.keras'}")

    with open(output_dir / "history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )

    return {"model": model, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid angle+scalar model")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "training" / "hybrid_angle_scalar_v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scalar-loss-weight", type=float, default=0.7)
    parser.add_argument("--angle-loss-weight", type=float, default=0.3)
    args = parser.parse_args()

    train_hybrid_model(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        scalar_loss_weight=args.scalar_loss_weight,
        angle_loss_weight=args.angle_loss_weight,
    )


if __name__ == "__main__":
    main()
