"""
Train an enhanced CNN model that beats prod v0.3.

Architecture improvements (literature-backed):
1. Multi-scale feature fusion — combines early/mid/late MobileNetV2 features
2. CBAM attention — channel + spatial attention (Woo et al., ECCV 2018)
3. Wider head (256 units) with LayerNorm for stability
4. Linear output — no sigmoid saturation at extremes
5. Auxiliary sweep-fraction head for geometric supervision
6. Cosine annealing with warm restarts
7. EMA weight averaging
8. Temperature-aware loss weighting
9. CutMix augmentation
10. Ensemble of multiple heads

Data strategy:
- Train on ALL available data (full_scalar_manifest_v1.csv = ~538 samples)
- Hold out hard_cases.csv for evaluation ONLY (never seen during training)
- Use the fixed training crop (same as firmware prod v0.3)

Usage:
    poetry run python scripts/train_enhanced_model.py --variant multi_scale
    poetry run python scripts/train_enhanced_model.py --variant ensemble --num-heads 3
    poetry run python scripts/train_enhanced_model.py --variant coord_attn
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import keras
import numpy as np
import tensorflow as tf

# Add ml/src to path
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.models import (
    build_mobilenetv2_enhanced_regression_model,
    build_mobilenetv2_enhanced_ensemble_model,
    build_mobilenetv2_regression_model,
)

# ─── Constants ───────────────────────────────────────────────────────────────

TRAINING_CROP_X_MIN = 0.1027
TRAINING_CROP_Y_MIN = 0.2573
TRAINING_CROP_X_MAX = 0.7987
TRAINING_CROP_Y_MAX = 0.8071
IMAGE_SIZE = 224
VALUE_MIN = -30.0
VALUE_MAX = 50.0
VALUE_SPAN = VALUE_MAX - VALUE_MIN

ML_ROOT: Path = PROJECT_ROOT
DATA_DIR: Path = ML_ROOT / "data"
CAPTURED_DIR: Path = DATA_DIR / "captured_images"
RAW_DIR: Path = DATA_DIR / "raw"
ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "training"
LOGS_DIR: Path = ML_ROOT / "artifacts" / "training_logs"

# ─── Data Loading ────────────────────────────────────────────────────────────


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Load a CSV manifest with image_path,value columns."""
    rows: list[dict[str, str]] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def resolve_image_path(rel_path: str) -> Path:
    """Resolve a relative image path to an absolute path."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    # Strip leading 'ml/' prefix if present (manifest paths already include it)
    rel = rel_path
    if rel.startswith("ml/"):
        rel = rel[3:]
    elif rel.startswith("ml\\"):
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
    # Fallback: try the original path
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
    """Load an image, apply the fixed training crop, resize with pad to target.

    Uses pure PIL/numpy — no TF ops — so it works inside tf.numpy_function.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x0 = int(crop_x_min * w)
    y0 = int(crop_y_min * h)
    x1 = int(crop_x_max * w)
    y1 = int(crop_y_max * h)
    crop = img.crop((x0, y0, x1, y1))

    # Resize with pad using pure PIL: compute scale, resize, paste onto canvas.
    crop_w, crop_h = crop.size
    scale = min(target_size / crop_w, target_size / crop_h)
    new_w = int(round(crop_w * scale))
    new_h = int(round(crop_h * scale))
    resized = crop.resize((new_w, new_h), Image.BILINEAR)

    # Create target canvas and paste centered
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas.paste(resized, (x_offset, y_offset))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    return arr


def build_dataset(
    manifest_rows: list[dict[str, str]],
    *,
    augment: bool = True,
    batch_size: int = 16,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a tf.data pipeline from manifest rows."""
    paths: list[str] = []
    values: list[float] = []

    for row in manifest_rows:
        img_path = row["image_path"]
        value = float(row["value"])
        # Verify file exists before adding to dataset
        resolved = resolve_image_path(img_path)
        if not resolved.exists():
            print(f"  [WARN] Skipping missing file: {resolved}")
            continue
        paths.append(img_path)
        values.append(value)

    if not paths:
        raise ValueError("No valid image paths found in manifest!")

    print(f"  [DATA] {len(paths)} valid samples loaded")

    paths_np = np.array(paths, dtype=str)
    values_np = np.array(values, dtype=np.float32)

    def _load_fn(p: str, v: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Load image, return (image, value, weight)."""
        img = tf.numpy_function(
            lambda path: _safe_load_image(path.decode("utf-8")),
            [p],
            tf.float32,
        )
        img.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))

        # Temperature-aware weight
        weight = temperature_aware_weight(v)

        return img, v, weight

    dataset = tf.data.Dataset.from_tensor_slices((paths_np, values_np))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths_np), seed=seed)

    dataset = dataset.map(_load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(
            lambda img, v, w: (augment_image(img), v, w),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return dataset


def _safe_load_image(path_str: str) -> np.ndarray:
    """Load and crop an image, returning a zero array on failure."""
    try:
        return load_and_crop_image(resolve_image_path(path_str))
    except Exception as e:
        print(f"  [WARN] Failed to load {path_str}: {e}")
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)


def build_multi_output_dataset(
    manifest_rows: list[dict[str, str]],
    *,
    augment: bool = True,
    batch_size: int = 16,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a tf.data pipeline for multi-output models (gauge_value + sweep_fraction)."""
    paths: list[str] = []
    values: list[float] = []

    for row in manifest_rows:
        img_path = row["image_path"]
        value = float(row["value"])
        resolved = resolve_image_path(img_path)
        if not resolved.exists():
            print(f"  [WARN] Skipping missing file: {resolved}")
            continue
        paths.append(img_path)
        values.append(value)

    if not paths:
        raise ValueError("No valid image paths found in manifest!")

    print(f"  [DATA] {len(paths)} valid samples loaded")

    paths_np = np.array(paths, dtype=str)
    values_np = np.array(values, dtype=np.float32)

    def _load_fn(
        p: str, v: tf.Tensor
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor], tf.Tensor]:
        """Load image, return (image, {gauge_value, sweep_fraction}, weight)."""
        img = tf.numpy_function(
            lambda path: _safe_load_image(path.decode("utf-8")),
            [p],
            tf.float32,
        )
        img.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))

        weight = temperature_aware_weight(v)

        # Compute sweep fraction target from gauge value
        sweep_frac = (v - VALUE_MIN) / VALUE_SPAN
        sweep_frac = tf.clip_by_value(sweep_frac, 0.0, 1.0)

        targets = {
            "gauge_value": v,
            "sweep_fraction": sweep_frac,
        }
        return img, targets, weight

    dataset = tf.data.Dataset.from_tensor_slices((paths_np, values_np))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths_np), seed=seed)

    dataset = dataset.map(_load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(
            lambda img, t, w: (augment_image(img), t, w),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return dataset


# ─── Augmentation ────────────────────────────────────────────────────────────


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """Apply photometric and geometric augmentation matching board reality."""
    img_shape = tf.shape(image)
    h, w = img_shape[0], img_shape[1]

    # 1. Random crop jitter (simulate dial position variation)
    scale = tf.random.uniform([], minval=0.88, maxval=1.0, dtype=tf.float32)
    crop_h = tf.maximum(2, tf.cast(tf.cast(h, tf.float32) * scale, tf.int32))
    crop_w = tf.maximum(2, tf.cast(tf.cast(w, tf.float32) * scale, tf.int32))
    image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
    image = tf.image.resize(image, [h, w])

    # 2. Random translation (5% max offset)
    max_offset = tf.cast(tf.cast(h, tf.float32) * 0.05, tf.int32)
    if max_offset > 0:
        offset_y = tf.random.uniform([], -max_offset, max_offset + 1, dtype=tf.int32)
        offset_x = tf.random.uniform([], -max_offset, max_offset + 1, dtype=tf.int32)
        pad_h = h + 2 * max_offset
        pad_w = w + 2 * max_offset
        image_padded = tf.image.resize_with_pad(image, pad_h, pad_w)
        image = tf.image.crop_to_bounding_box(
            image_padded, max_offset + offset_y, max_offset + offset_x, h, w
        )

    # 3. Brightness/exposure (wider range for board robustness)
    brightness_delta = tf.random.uniform([], -0.25, 0.25, dtype=tf.float32)
    image = tf.image.adjust_brightness(image, brightness_delta)

    # 4. Contrast
    contrast_factor = tf.random.uniform([], 0.65, 1.45, dtype=tf.float32)
    image = tf.image.adjust_contrast(image, contrast_factor)

    # 5. Saturation
    saturation_factor = tf.random.uniform([], 0.75, 1.35, dtype=tf.float32)
    image = tf.image.adjust_saturation(image, saturation_factor)

    # 6. Hue (small amount)
    hue_delta = tf.random.uniform([], -0.05, 0.05, dtype=tf.float32)
    image = tf.image.adjust_hue(image, hue_delta)

    # 7. Gamma correction (simulate non-linear camera response)
    gamma = tf.random.uniform([], 0.6, 1.8, dtype=tf.float32)
    image = tf.image.adjust_gamma(image, gamma)

    # 8. Random Gaussian noise (sensor noise)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    # 9. Random glare/specular reflection (common on gauge glass)
    if tf.random.uniform([]) < 0.3:
        glare_x = tf.random.uniform([], 0.1, 0.9)
        glare_y = tf.random.uniform([], 0.1, 0.9)
        glare_radius = tf.random.uniform([], 0.05, 0.2)
        glare_strength = tf.random.uniform([], 0.1, 0.4)
        yy, xx = tf.meshgrid(
            tf.linspace(0.0, 1.0, h),
            tf.linspace(0.0, 1.0, w),
            indexing="ij",
        )
        dist = tf.sqrt((xx - glare_x) ** 2 + (yy - glare_y) ** 2)
        glare_mask = tf.exp(-(dist**2) / (2.0 * glare_radius**2))
        glare_mask = glare_mask[..., tf.newaxis]
        image = (
            image * (1.0 - glare_mask * glare_strength) + glare_mask * glare_strength
        )

    return tf.clip_by_value(image, 0.0, 1.0)


# ─── Temperature-Aware Loss ──────────────────────────────────────────────────


def temperature_aware_weight(true_temp: tf.Tensor) -> tf.Tensor:
    """Compute per-sample loss weights for full-range accuracy.

    Hot band (35-50°C): 2.5x — the primary failure mode
    Cold band (<0°C): 2.0x — secondary failure mode
    Low band (0-20°C): 1.3x — slight boost
    Mid band (20-35°C): 1.0x — baseline
    """
    hot = (
        tf.cast(tf.logical_and(true_temp >= 35.0, true_temp <= 50.0), tf.float32) * 2.5
    )
    cold = tf.cast(true_temp < 0.0, tf.float32) * 2.0
    low = tf.cast(tf.logical_and(true_temp >= 0.0, true_temp < 20.0), tf.float32) * 1.3
    mid = tf.cast(tf.logical_and(true_temp >= 20.0, true_temp < 35.0), tf.float32) * 1.0
    return hot + cold + low + mid


def make_weighted_mse_loss():
    """Create a temperature-weighted MSE loss function."""

    def weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        squared_error = tf.square(y_true_f - y_pred_f)
        weights = temperature_aware_weight(y_true_f)
        return tf.reduce_mean(squared_error * weights)

    weighted_mse.__name__ = "weighted_mse"
    return weighted_mse


# ─── Cosine Annealing LR Schedule ────────────────────────────────────────────


class CosineAnnealingSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing with warm restarts (Loshchilov & Hutter, ICLR 2017)."""

    def __init__(
        self,
        initial_lr: float = 1e-4,
        min_lr: float = 1e-6,
        warmup_steps: int = 200,
        total_steps: int = 2000,
        cycle_length: int = 1000,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycle_length = cycle_length

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step_f = tf.cast(step, tf.float32)
        # Warmup phase
        warmup_lr = self.initial_lr * (step_f / tf.cast(self.warmup_steps, tf.float32))
        warmup_lr = tf.minimum(warmup_lr, self.initial_lr)

        # Cosine annealing after warmup
        cycle_pos = tf.math.floormod(
            step_f - tf.cast(self.warmup_steps, tf.float32),
            tf.cast(self.cycle_length, tf.float32),
        )
        cosine_decay = 0.5 * (
            1.0
            + tf.cos(
                tf.constant(np.pi, tf.float32)
                * cycle_pos
                / tf.cast(self.cycle_length, tf.float32)
            )
        )
        annealed_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

        return tf.where(
            step_f < tf.cast(self.warmup_steps, tf.float32), warmup_lr, annealed_lr
        )

    def get_config(self) -> dict:
        return {
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "cycle_length": self.cycle_length,
        }


# ─── EMA Callback ────────────────────────────────────────────────────────────


class EMACallback(keras.callbacks.Callback):
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of all trainable weights and updates them
    at the end of each batch. At the end of training, the EMA weights
    are set as the model weights for better generalization.
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow_weights: list[tf.Tensor] | None = None
        self.backup_weights: list[tf.Tensor] | None = None

    def on_train_begin(self, logs=None):
        self.shadow_weights = [
            tf.Variable(w, dtype=w.dtype, trainable=False)
            for w in self.model.trainable_weights
        ]

    def on_batch_end(self, batch, logs=None):
        if self.shadow_weights is None:
            return
        for shadow, weight in zip(self.shadow_weights, self.model.trainable_weights):
            shadow.assign(self.decay * shadow + (1.0 - self.decay) * weight)

    def on_test_begin(self, logs=None):
        """Backup current weights and set EMA weights for evaluation."""
        if self.shadow_weights is None:
            return
        self.backup_weights = [tf.identity(w) for w in self.model.trainable_weights]
        for shadow, weight in zip(self.shadow_weights, self.model.trainable_weights):
            weight.assign(shadow)

    def on_test_end(self, logs=None):
        """Restore original weights after evaluation."""
        if self.backup_weights is None or self.shadow_weights is None:
            return
        for weight, backup in zip(self.model.trainable_weights, self.backup_weights):
            weight.assign(backup)

    def on_train_end(self, logs=None):
        """Set EMA weights as final model weights."""
        if self.shadow_weights is None:
            return
        for weight, shadow in zip(self.model.trainable_weights, self.shadow_weights):
            weight.assign(shadow)


# ─── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_on_manifest(
    model: keras.Model,
    manifest_path: Path,
    *,
    batch_size: int = 8,
) -> dict[str, float]:
    """Evaluate model on a manifest and return per-sample + aggregate metrics."""
    rows = load_manifest(manifest_path)
    predictions: list[float] = []
    targets: list[float] = []
    per_sample: list[dict] = []

    for row in rows:
        img_path = resolve_image_path(row["image_path"])
        target = float(row["value"])

        if not img_path.exists():
            print(f"  [WARN] Image not found: {img_path}")
            continue

        img = load_and_crop_image(img_path)
        batch = img[np.newaxis, ...]
        pred_out = model.predict(batch, verbose=0)
        # Handle both single-output (array) and multi-output (dict) models
        if isinstance(pred_out, dict):
            pred = float(pred_out["gauge_value"][0][0])
        elif isinstance(pred_out, list):
            pred = float(pred_out[0][0][0])
        else:
            pred = float(pred_out[0][0])

        predictions.append(pred)
        targets.append(target)
        per_sample.append(
            {
                "image": img_path.name,
                "true": target,
                "pred": pred,
                "abs_error": abs(pred - target),
            }
        )

    if not predictions:
        return {"mae": -1.0, "max_error": -1.0, "count": 0, "per_sample": []}

    preds_np = np.array(predictions)
    targets_np = np.array(targets)
    abs_errors = np.abs(preds_np - targets_np)

    # Per-band MAE
    cold_mask = targets_np < 0.0
    low_mask = (targets_np >= 0.0) & (targets_np < 20.0)
    mid_mask = (targets_np >= 20.0) & (targets_np < 35.0)
    hot_mask = targets_np >= 35.0

    results = {
        "mae": float(np.mean(abs_errors)),
        "max_error": float(np.max(abs_errors)),
        "rmse": float(np.sqrt(np.mean(abs_errors**2))),
        "count": len(predictions),
        "over_5c": int(np.sum(abs_errors > 5.0)),
        "over_10c": int(np.sum(abs_errors > 10.0)),
        "cold_mae": (
            float(np.mean(abs_errors[cold_mask])) if np.any(cold_mask) else -1.0
        ),
        "low_mae": float(np.mean(abs_errors[low_mask])) if np.any(low_mask) else -1.0,
        "mid_mae": float(np.mean(abs_errors[mid_mask])) if np.any(mid_mask) else -1.0,
        "hot_mae": float(np.mean(abs_errors[hot_mask])) if np.any(hot_mask) else -1.0,
        "per_sample": per_sample,
    }
    return results


def print_eval_results(name: str, results: dict[str, float]) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  MAE:       {results['mae']:.4f}°C")
    print(f"  RMSE:      {results['rmse']:.4f}°C")
    print(f"  Max Error: {results['max_error']:.4f}°C")
    print(f"  Over 5°C:  {results['over_5c']}/{results['count']}")
    print(f"  Over 10°C: {results['over_10c']}/{results['count']}")
    print(f"  Cold MAE:  {results['cold_mae']:.4f}°C")
    print(f"  Low MAE:   {results['low_mae']:.4f}°C")
    print(f"  Mid MAE:   {results['mid_mae']:.4f}°C")
    print(f"  Hot MAE:   {results['hot_mae']:.4f}°C")
    print(f"{'='*60}\n")


# ─── Training ────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    variant: Literal["multi_scale", "coord_attn", "ensemble", "baseline"] = (
        "multi_scale"
    )
    epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    seed: int = 42
    head_units: int = 256
    head_dropout: float = 0.3
    num_heads: int = 3
    backbone_trainable: bool = True
    alpha: float = 1.0
    train_manifest: str = "full_scalar_manifest_v1.csv"
    val_manifest: str = ""
    hard_case_manifest: str = "hard_cases.csv"
    run_name: str = "enhanced_model"
    artifacts_dir: str = ""


def train_enhanced(config: TrainConfig) -> keras.Model:
    """Run the full training pipeline."""
    print(f"\n{'#'*60}")
    print(f"  Training Enhanced Model — Variant: {config.variant}")
    print(f"{'#'*60}\n")

    # Set seeds
    keras.utils.set_random_seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    # Configure GPU
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"[TRAIN] GPUs: {len(gpus)} available")

    # Load training data
    train_manifest_path = DATA_DIR / config.train_manifest
    if not train_manifest_path.exists():
        train_manifest_path = ML_ROOT / config.train_manifest
    print(f"[TRAIN] Loading training data from: {train_manifest_path}")
    train_rows = load_manifest(train_manifest_path)
    print(f"[TRAIN] Training samples: {len(train_rows)}")

    # Load hard cases for eval (NEVER for training)
    hard_case_path = DATA_DIR / config.hard_case_manifest
    if not hard_case_path.exists():
        hard_case_path = ML_ROOT / config.hard_case_manifest
    print(f"[TRAIN] Hard cases for eval: {hard_case_path}")
    hard_case_rows = load_manifest(hard_case_path)
    print(f"[TRAIN] Hard case samples: {len(hard_case_rows)}")

    # Filter hard cases from training data (ensure no leakage)
    hard_case_names = {Path(r["image_path"]).name for r in hard_case_rows}
    filtered_train_rows = [
        r for r in train_rows if Path(r["image_path"]).name not in hard_case_names
    ]
    removed = len(train_rows) - len(filtered_train_rows)
    if removed > 0:
        print(
            f"[TRAIN] Removed {removed} hard-case images from training set (no leakage)"
        )

    # Split training into train/val (90/10)
    np.random.shuffle(filtered_train_rows)
    val_split = int(len(filtered_train_rows) * 0.1)
    val_rows = filtered_train_rows[:val_split]
    train_rows_final = filtered_train_rows[val_split:]
    print(f"[TRAIN] Train: {len(train_rows_final)}, Val: {len(val_rows)}")

    # Build datasets
    steps_per_epoch = max(len(train_rows_final) // config.batch_size, 1)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs
    cycle_steps = steps_per_epoch * 30  # Restart every 30 epochs

    is_multi_output = config.variant == "multi_scale"
    dataset_builder = build_multi_output_dataset if is_multi_output else build_dataset

    train_ds = dataset_builder(
        train_rows_final,
        augment=True,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_ds = dataset_builder(
        val_rows,
        augment=False,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Build model
    print(f"\n[TRAIN] Building model (variant={config.variant})...")
    if config.variant == "multi_scale":
        model = build_mobilenetv2_enhanced_regression_model(
            IMAGE_SIZE,
            IMAGE_SIZE,
            pretrained=True,
            backbone_trainable=config.backbone_trainable,
            alpha=config.alpha,
            head_units=config.head_units,
            head_dropout=config.head_dropout,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            use_multi_scale=True,
            use_coord_attention=False,
        )
    elif config.variant == "coord_attn":
        model = build_mobilenetv2_enhanced_regression_model(
            IMAGE_SIZE,
            IMAGE_SIZE,
            pretrained=True,
            backbone_trainable=config.backbone_trainable,
            alpha=config.alpha,
            head_units=config.head_units,
            head_dropout=config.head_dropout,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            use_multi_scale=False,
            use_coord_attention=True,
        )
    elif config.variant == "ensemble":
        model = build_mobilenetv2_enhanced_ensemble_model(
            IMAGE_SIZE,
            IMAGE_SIZE,
            pretrained=True,
            backbone_trainable=config.backbone_trainable,
            alpha=config.alpha,
            head_units=config.head_units,
            head_dropout=config.head_dropout,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            num_heads=config.num_heads,
        )
    elif config.variant == "baseline":
        model = build_mobilenetv2_regression_model(
            IMAGE_SIZE,
            IMAGE_SIZE,
            pretrained=True,
            backbone_trainable=config.backbone_trainable,
            alpha=1.0,
            head_units=128,
            head_dropout=0.2,
            linear_output=True,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
        )
    else:
        raise ValueError(f"Unknown variant: {config.variant}")

    model.summary()
    total_params = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"[TRAIN] Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Learning rate schedule
    lr_schedule = CosineAnnealingSchedule(
        initial_lr=config.learning_rate,
        min_lr=config.min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        cycle_length=cycle_steps,
    )

    # Compile
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    if config.variant in ("multi_scale",):
        # Multi-scale model has two outputs
        model.compile(
            optimizer=optimizer,
            loss={
                "gauge_value": make_weighted_mse_loss(),
                "sweep_fraction": keras.losses.MeanSquaredError(),
            },
            loss_weights={
                "gauge_value": 1.0,
                "sweep_fraction": 0.3,
            },
            metrics={
                "gauge_value": [
                    keras.metrics.MeanAbsoluteError(name="mae"),
                    keras.metrics.RootMeanSquaredError(name="rmse"),
                ],
            },
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=make_weighted_mse_loss(),
            metrics=[
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        )

    # Callbacks
    artifacts_dir = (
        Path(config.artifacts_dir)
        if config.artifacts_dir
        else (ARTIFACTS_DIR / config.run_name)
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(artifacts_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(artifacts_dir / "training_log.csv"),
        ),
        EMACallback(decay=0.999),
    ]

    # Train
    print(f"\n[TRAIN] Starting training for {config.epochs} epochs...")
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    elapsed = time.time() - start_time
    print(f"\n[TRAIN] Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save final model
    final_path = artifacts_dir / "model.keras"
    model.save(str(final_path))
    print(f"[TRAIN] Final model saved to: {final_path}")

    # Evaluate on hard cases
    print(f"\n[TRAIN] Evaluating on hard cases...")
    hard_results = evaluate_on_manifest(model, hard_case_path)
    print_eval_results(f"Hard Cases — {config.variant}", hard_results)

    # Evaluate on validation set
    val_results = evaluate_on_manifest(
        model,
        manifest_path=None,  # Will use val_rows directly
    )
    # Manual val eval
    val_preds = []
    val_targets = []
    for row in val_rows:
        img_path = resolve_image_path(row["image_path"])
        target = float(row["value"])
        if img_path.exists():
            img = load_and_crop_image(img_path)
            pred_out = model.predict(img[np.newaxis, ...], verbose=0)
            if isinstance(pred_out, dict):
                pred = float(pred_out["gauge_value"][0][0])
            elif isinstance(pred_out, list):
                pred = float(pred_out[0][0][0])
            else:
                pred = float(pred_out[0][0])
            val_preds.append(pred)
            val_targets.append(target)

    if val_preds:
        val_abs = np.abs(np.array(val_preds) - np.array(val_targets))
        val_results = {
            "mae": float(np.mean(val_abs)),
            "max_error": float(np.max(val_abs)),
            "count": len(val_preds),
        }
        print_eval_results(f"Validation Set — {config.variant}", val_results)

    # Save results
    results = {
        "variant": config.variant,
        "run_name": config.run_name,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "train_samples": len(train_rows_final),
        "val_samples": len(val_rows),
        "hard_case_samples": len(hard_case_rows),
        "epochs_trained": len(history.history["loss"]),
        "training_time_min": elapsed / 60,
        "hard_case": hard_results,
        "validation": val_results,
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }

    results_path = artifacts_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[TRAIN] Results saved to: {results_path}")

    return model


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train enhanced gauge-reading model.")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["multi_scale", "coord_attn", "ensemble", "baseline"],
        default="multi_scale",
        help="Model architecture variant.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--head-units", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument(
        "--train-manifest", type=str, default="full_scalar_manifest_v1.csv"
    )
    parser.add_argument("--no-backbone-trainable", action="store_true")
    parser.add_argument(
        "--device", type=str, choices=["auto", "cpu", "gpu"], default="gpu"
    )

    args = parser.parse_args()

    # Auto-generate run name
    run_name = (
        args.run_name
        or f"enhanced_{args.variant}_e{args.epochs}_lr{args.learning_rate}_h{args.head_units}_s{args.seed}"
    )

    config = TrainConfig(
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        head_units=args.head_units,
        head_dropout=args.head_dropout,
        num_heads=args.num_heads,
        seed=args.seed,
        backbone_trainable=not args.no_backbone_trainable,
        train_manifest=args.train_manifest,
        run_name=run_name,
        artifacts_dir=args.artifacts_dir or str(ARTIFACTS_DIR / run_name),
    )

    train_enhanced(config)


if __name__ == "__main__":
    main()
