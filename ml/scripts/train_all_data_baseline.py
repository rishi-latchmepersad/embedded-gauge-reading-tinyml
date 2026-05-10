#!/usr/bin/env python3
"""Train on all merged data using the proven canonical baseline pipeline.

This script merges all available manifests, creates a 70/15/15 split,
and trains using the EXACT same pipeline as train_canonical_baseline.py
which achieved 7.92°C MAE on 141 images.

With 400+ images, this should significantly improve results.

Usage:
    cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
    poetry run python scripts/train_all_data_baseline.py \
        --output-dir artifacts/training/all_data_baseline
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
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from PIL import Image, ImageOps

# Add ml/src to path
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.models import (
    build_mobilenetv2_dual_resolution_interval_model,
    build_mobilenetv2_dual_resolution_regression_model,
    build_mobilenetv2_interval_model,
    build_mobilenetv2_polar_regression_model,
    build_mobilenetv2_polar_sweep_distribution_model,
    build_mobilenetv2_polar_dualview_regression_model,
    build_mobilenetv2_sweep_distribution_model,
    build_mobilenetv2_ordinal_model,
    build_mobilenetv2_regression_model,
)
from embedded_gauge_reading_tinyml.board_crop_compare import (
    load_rgb_image,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
VALUE_MIN = -30.0
VALUE_MAX = 50.0


def _configure_mobilenet_backbone_trainability(
    base_model: keras.Model,
    *,
    trainable: bool,
    unfreeze_last_n: int = 0,
    freeze_batchnorm: bool = True,
) -> None:
    """Apply a staged fine-tuning policy to the MobileNetV2 backbone."""
    base_model.trainable = trainable
    for layer in base_model.layers:
        layer.trainable = trainable

    if not trainable:
        return

    if unfreeze_last_n > 0:
        cutoff = max(0, len(base_model.layers) - int(unfreeze_last_n))
        for index, layer in enumerate(base_model.layers):
            layer.trainable = index >= cutoff

    if freeze_batchnorm:
        for layer in base_model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False


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


def _load_cropped_pil_image(
    image_path: str,
    crop_box_xyxy: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Load an image from disk and apply an optional crop box.

    We reuse this helper for both the normal scalar view and the new
    polar-unwrapped view so the two branches stay spatially aligned.
    """
    img = load_rgb_image(image_path)
    pil_image = Image.fromarray(img, mode="RGB")
    if crop_box_xyxy is None:
        return pil_image

    x0, y0, x1, y1 = crop_box_xyxy
    left = max(0, int(np.floor(x0)))
    top = max(0, int(np.floor(y0)))
    right = min(pil_image.width, int(np.ceil(x1)))
    bottom = min(pil_image.height, int(np.ceil(y1)))
    if right <= left:
        right = min(pil_image.width, left + 1)
    if bottom <= top:
        bottom = min(pil_image.height, top + 1)
    return pil_image.crop((left, top, right, bottom))


def load_scalar_manifest(file_path: Path, repo_root: Path) -> pd.DataFrame:
    """Load a prebuilt scalar manifest and normalize its path columns."""
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
    if "sample_weight" in df.columns:
        df["sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["image_path_resolved"].apply(lambda p: Path(p).exists())]
    if "sample_weight" in df.columns:
        df = df.dropna(subset=["sample_weight"])
    return df.reset_index(drop=True)


def load_manifest(file_path: Path, repo_root: Path) -> pd.DataFrame | None:
    """Load a manifest CSV and standardize columns."""
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


def load_training_dataframe(
    repo_root: Path,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Load the training dataframe from a custom manifest or the legacy merge path."""
    if manifest_path is not None:
        logger.info(f"Loading scalar manifest: {manifest_path}")
        return load_scalar_manifest(manifest_path, repo_root)

    logger.info("Loading all manifests using the legacy merge path...")
    return merge_all_manifests(repo_root)


def merge_all_manifests(repo_root: Path) -> pd.DataFrame:
    """Load and merge all available manifests with deduplication."""
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


def load_precomputed_crop_boxes(
    boxes_path: Path, repo_root: Path
) -> dict[str, tuple[float, float, float, float]]:
    """Load precomputed rectifier crop boxes keyed by resolved image path."""
    df = pd.read_csv(boxes_path)
    required_columns = {"image_path", "x0", "y0", "x1", "y1"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Crop-box CSV is missing columns {sorted(missing)}: {boxes_path}"
        )

    boxes: dict[str, tuple[float, float, float, float]] = {}
    for _, row in df.iterrows():
        normalized = normalize_path(str(row["image_path"]), repo_root)
        resolved = str(resolve_full_path(normalized, repo_root))
        boxes[resolved] = (
            float(row["x0"]),
            float(row["y0"]),
            float(row["x1"]),
            float(row["y1"]),
        )

    logger.info(f"Loaded {len(boxes)} precomputed rectifier crop boxes.")
    return boxes


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
    """Compute sample weights inversely proportional to bin frequency, capped at 5.0."""
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


def _value_to_ordinal_threshold_vector(
    value: float,
    *,
    value_min: float,
    value_max: float,
    threshold_step: float,
) -> np.ndarray:
    """Convert one scalar value into a cumulative ordinal-threshold vector."""
    if threshold_step <= 0.0:
        raise ValueError("threshold_step must be > 0.")
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    span = value_max - value_min
    num_thresholds = max(int(np.ceil(span / threshold_step)), 2)
    thresholds = value_min + (
        np.arange(num_thresholds, dtype=np.float32) + 0.5
    ) * np.float32(threshold_step)
    return (np.float32(value) > thresholds).astype(np.float32)


def _value_to_interval_index(
    value: float,
    *,
    value_min: float,
    value_max: float,
    bin_width: float,
) -> np.int32:
    """Map one scalar value into a coarse interval-bin index."""
    if bin_width <= 0.0:
        raise ValueError("bin_width must be > 0.")
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    span = value_max - value_min
    num_bins = max(int(np.ceil(span / bin_width)), 2)
    raw_index = int(np.floor((float(value) - value_min) / bin_width))
    clipped_index = int(np.clip(raw_index, 0, num_bins - 1))
    return np.int32(clipped_index)


def _value_to_sweep_distribution(
    value: float,
    *,
    value_min: float,
    value_max: float,
    num_bins: int,
    sigma_bins: float,
) -> np.ndarray:
    """Convert one scalar value into a smooth sweep-bin distribution target."""
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2.")
    if sigma_bins <= 0.0:
        raise ValueError("sigma_bins must be > 0.")
    if value_max <= value_min:
        raise ValueError("value_max must be > value_min.")

    span = value_max - value_min
    fraction = float(np.clip((float(value) - value_min) / span, 0.0, 1.0))
    center_bin = fraction * float(num_bins - 1)
    bin_indices = np.arange(num_bins, dtype=np.float32)
    distribution = np.exp(
        -0.5 * ((bin_indices - np.float32(center_bin)) / np.float32(sigma_bins)) ** 2
    )
    total = float(np.sum(distribution))
    if total > 0.0:
        distribution /= np.float32(total)
    return distribution.astype(np.float32)


def preprocess_image(
    image_path: str,
    target_size: tuple[int, int] = (224, 224),
    crop_box_xyxy: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Load and preprocess an image for training.

    When a rectifier crop is available, we crop first so the scalar model sees
    the same board-aligned framing it will receive at inference time.
    """
    pil_image = _load_cropped_pil_image(image_path, crop_box_xyxy)

    resized = ImageOps.pad(
        pil_image,
        target_size,
        method=Image.BILINEAR,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )
    img_resized = np.asarray(resized, dtype=np.uint8)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized


def preprocess_polar_image(
    image_path: str,
    target_size: tuple[int, int] = (224, 224),
    crop_box_xyxy: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Load and polar-unwarp an image for the dual-view gauge model.

    The gauge face is roughly circular, so a polar projection turns the dial
    into a more linear angle-versus-radius image. That makes the needle and
    tick structure easier for the CNN to separate.
    """
    pil_image = _load_cropped_pil_image(image_path, crop_box_xyxy)
    rgb_image = np.asarray(pil_image, dtype=np.uint8)
    if rgb_image.size == 0:
        raise ValueError(f"Image is empty after cropping: {image_path}")

    height, width = rgb_image.shape[:2]
    center = (float(width) * 0.5, float(height) * 0.5)
    max_radius = max(1.0, float(min(height, width)) * 0.5)
    polar_width = int(target_size[1])
    polar_height = int(target_size[0])

    # OpenCV maps angle across the horizontal axis and radius down the vertical
    # axis, which flattens the dial into a rectangular "spoke band" image.
    polar_image = cv2.warpPolar(
        rgb_image,
        (polar_width, polar_height),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS,
    )
    if polar_image.shape[0] != polar_height or polar_image.shape[1] != polar_width:
        polar_image = cv2.resize(
            polar_image,
            (polar_width, polar_height),
            interpolation=cv2.INTER_LINEAR,
        )

    polar_normalized = polar_image.astype(np.float32) / 255.0
    return polar_normalized


def create_dataset(
    df: pd.DataFrame,
    batch_size: int,
    shuffle: bool = False,
    repeat: bool = False,
    use_weights: bool = False,
    augment: bool = False,
    augment_mode: str = "standard",
    crop_boxes: dict[str, tuple[float, float, float, float]] | None = None,
    aux_head_kind: str | None = None,
    polar_only_model: bool = False,
    polar_dualview_model: bool = False,
    polar_sweep_distribution_model: bool = False,
    ordinal_threshold_step: float = 10.0,
    interval_bin_width: float = 5.0,
    sweep_distribution_bins: int = 81,
    sweep_distribution_sigma_bins: float = 1.75,
) -> tf.data.Dataset:
    """Create a TensorFlow dataset from a DataFrame.

    The loader keeps only metadata in memory and streams the image pixels from
    disk through a generator. That avoids the large one-shot tensor material-
    ization step that was stalling the dual-resolution runs.
    """
    logger.info(f"Loading {len(df)} images into memory...")
    loaded_records: list[
        tuple[str, float, float | None, tuple[float, float, float, float] | None]
    ] = []

    for index, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            if index == 1 or index % 100 == 0:
                logger.info(
                    "  ...loaded %d/%d images so far (next=%s)",
                    index - 1,
                    len(df),
                    row["image_path_resolved"],
                )
            crop_box = None
            if crop_boxes is not None:
                crop_box = crop_boxes.get(str(row["image_path_resolved"]))
            # We validate the image path up front so bad files are skipped early.
            if polar_only_model or polar_sweep_distribution_model:
                _ = preprocess_polar_image(
                    row["image_path_resolved"],
                    (224, 224),
                    crop_box_xyxy=crop_box,
                )
            else:
                _ = preprocess_image(
                    row["image_path_resolved"],
                    (224, 224),
                    crop_box_xyxy=crop_box,
                )
            sample_weight = (
                float(row["sample_weight"]) if use_weights and "sample_weight" in df.columns else None
            )
            loaded_records.append(
                (
                    str(row["image_path_resolved"]),
                    float(row["value"]),
                    sample_weight,
                    crop_box,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to load {row['image_path_resolved']}: {e}")
            continue

    logger.info(f"Successfully loaded {len(loaded_records)} images")

    if aux_head_kind is not None and aux_head_kind not in {
        "ordinal",
        "interval",
        "sweep_distribution",
    }:
        raise ValueError(f"Unsupported aux_head_kind: {aux_head_kind}")
    if polar_sweep_distribution_model and aux_head_kind not in {None, "sweep_distribution"}:
        raise ValueError(
            "polar_sweep_distribution_model currently supports sweep_distribution "
            "targets only."
        )
    if (polar_only_model or polar_sweep_distribution_model) and augment:
        logger.info(
            "Polar-geometry model requested with augmentation; disabling "
            "augmentation to keep the simplified pipeline stable."
        )
        augment = False

    ordinal_target_dim = max(
        int(np.ceil((VALUE_MAX - VALUE_MIN) / ordinal_threshold_step)), 2
    )
    interval_target_dim = max(
        int(np.ceil((VALUE_MAX - VALUE_MIN) / interval_bin_width)), 2
    )

    def _record_generator() -> Any:
        """Yield preprocessed images and the matching training targets."""
        for image_path, value, sample_weight, crop_box in loaded_records:
            if polar_only_model or polar_sweep_distribution_model:
                polar_image = preprocess_polar_image(
                    image_path,
                    (224, 224),
                    crop_box_xyxy=crop_box,
                )
            else:
                image = preprocess_image(
                    image_path,
                    (224, 224),
                    crop_box_xyxy=crop_box,
                )
            if polar_dualview_model:
                polar_image = preprocess_polar_image(
                    image_path,
                    (224, 224),
                    crop_box_xyxy=crop_box,
                )
            if aux_head_kind == "ordinal":
                ordinal_target = _value_to_ordinal_threshold_vector(
                    value,
                    value_min=VALUE_MIN,
                    value_max=VALUE_MAX,
                    threshold_step=ordinal_threshold_step,
                ).astype(np.float32)
                target = {
                    "gauge_value": np.float32(value),
                    "ordinal_logits": ordinal_target,
                }
                if use_weights and sample_weight is not None:
                    weights = {
                        "gauge_value": np.float32(sample_weight),
                        "ordinal_logits": np.float32(sample_weight),
                    }
                    if polar_dualview_model:
                        yield {
                            "full_image": image,
                            "polar_image": polar_image,
                        }, target, weights
                    elif polar_only_model or polar_sweep_distribution_model:
                        yield {"polar_image": polar_image}, target, weights
                    else:
                        yield image, target, weights
                else:
                    if polar_dualview_model:
                        yield {
                            "full_image": image,
                            "polar_image": polar_image,
                        }, target
                    elif polar_only_model or polar_sweep_distribution_model:
                        yield {"polar_image": polar_image}, target
                    else:
                        yield image, target
            elif aux_head_kind == "interval":
                interval_target = np.int32(
                    _value_to_interval_index(
                        value,
                        value_min=VALUE_MIN,
                        value_max=VALUE_MAX,
                        bin_width=interval_bin_width,
                    )
                )
                target = {
                    "gauge_value": np.float32(value),
                    "interval_logits": interval_target,
                }
                if use_weights and sample_weight is not None:
                    weights = {
                        "gauge_value": np.float32(sample_weight),
                        "interval_logits": np.float32(sample_weight),
                    }
                    if polar_dualview_model:
                        yield {
                            "full_image": image,
                            "polar_image": polar_image,
                        }, target, weights
                    elif polar_only_model or polar_sweep_distribution_model:
                        yield {"polar_image": polar_image}, target, weights
                    else:
                        yield image, target, weights
                else:
                    if polar_dualview_model:
                        yield {
                            "full_image": image,
                            "polar_image": polar_image,
                        }, target
                    elif polar_only_model or polar_sweep_distribution_model:
                        yield {"polar_image": polar_image}, target
                    else:
                        yield image, target
            elif aux_head_kind == "sweep_distribution":
                sweep_distribution_target = _value_to_sweep_distribution(
                    value,
                    value_min=VALUE_MIN,
                    value_max=VALUE_MAX,
                    num_bins=sweep_distribution_bins,
                    sigma_bins=sweep_distribution_sigma_bins,
                ).astype(np.float32)
                target = {
                    "gauge_value": np.float32(value),
                    "sweep_distribution_logits": sweep_distribution_target,
                }
                if use_weights and sample_weight is not None:
                    weights = {
                        "gauge_value": np.float32(sample_weight),
                        "sweep_distribution_logits": np.float32(sample_weight),
                    }
                    if polar_dualview_model:
                        yield {
                            "full_image": image,
                            "polar_image": polar_image,
                        }, target, weights
                    elif polar_only_model or polar_sweep_distribution_model:
                        yield {"polar_image": polar_image}, target, weights
                    else:
                        yield image, target, weights
                else:
                    if polar_dualview_model:
                        yield {
                            "full_image": image,
                            "polar_image": polar_image,
                        }, target
                    elif polar_only_model or polar_sweep_distribution_model:
                        yield {"polar_image": polar_image}, target
                    else:
                        yield image, target
            else:
                if use_weights and sample_weight is not None:
                    if polar_dualview_model:
                        yield (
                            {
                                "full_image": image,
                                "polar_image": polar_image,
                            },
                            np.float32(value),
                            np.float32(sample_weight),
                        )
                    elif polar_only_model or polar_sweep_distribution_model:
                        yield {"polar_image": polar_image}, np.float32(value), np.float32(sample_weight)
                    else:
                        yield image, np.float32(value), np.float32(sample_weight)
                else:
                    if polar_dualview_model:
                        yield {
                            "full_image": image,
                            "polar_image": polar_image,
                        }, np.float32(value)
                    elif polar_only_model or polar_sweep_distribution_model:
                        yield {"polar_image": polar_image}, np.float32(value)
                    else:
                        yield image, np.float32(value)

    if polar_dualview_model:
        image_spec: Any = {
            "full_image": tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            "polar_image": tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        }
    elif polar_only_model or polar_sweep_distribution_model:
        image_spec = {
            "polar_image": tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
        }
    else:
        image_spec = tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
    if aux_head_kind == "ordinal":
        target_spec = {
            "gauge_value": tf.TensorSpec(shape=(), dtype=tf.float32),
            "ordinal_logits": tf.TensorSpec(
                shape=(ordinal_target_dim,), dtype=tf.float32
            ),
        }
        if use_weights and "sample_weight" in df.columns:
            weight_spec = {
                "gauge_value": tf.TensorSpec(shape=(), dtype=tf.float32),
                "ordinal_logits": tf.TensorSpec(shape=(), dtype=tf.float32),
            }
            dataset = tf.data.Dataset.from_generator(
                _record_generator,
                output_signature=(image_spec, target_spec, weight_spec),
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                _record_generator,
                output_signature=(image_spec, target_spec),
            )
    elif aux_head_kind == "interval":
        target_spec = {
            "gauge_value": tf.TensorSpec(shape=(), dtype=tf.float32),
            "interval_logits": tf.TensorSpec(shape=(), dtype=tf.int32),
        }
        if use_weights and "sample_weight" in df.columns:
            weight_spec = {
                "gauge_value": tf.TensorSpec(shape=(), dtype=tf.float32),
                "interval_logits": tf.TensorSpec(shape=(), dtype=tf.float32),
            }
            dataset = tf.data.Dataset.from_generator(
                _record_generator,
                output_signature=(image_spec, target_spec, weight_spec),
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                _record_generator,
                output_signature=(image_spec, target_spec),
            )
    elif aux_head_kind == "sweep_distribution":
        target_spec = {
            "gauge_value": tf.TensorSpec(shape=(), dtype=tf.float32),
            "sweep_distribution_logits": tf.TensorSpec(
                shape=(sweep_distribution_bins,), dtype=tf.float32
            ),
        }
        if use_weights and "sample_weight" in df.columns:
            weight_spec = {
                "gauge_value": tf.TensorSpec(shape=(), dtype=tf.float32),
                "sweep_distribution_logits": tf.TensorSpec(
                    shape=(), dtype=tf.float32
                ),
            }
            dataset = tf.data.Dataset.from_generator(
                _record_generator,
                output_signature=(image_spec, target_spec, weight_spec),
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                _record_generator,
                output_signature=(image_spec, target_spec),
            )
    else:
        if use_weights and "sample_weight" in df.columns:
            dataset = tf.data.Dataset.from_generator(
                _record_generator,
                output_signature=(
                    image_spec,
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                ),
            )
            dataset = dataset.map(lambda x, y, w: (x, y, w))
        else:
            dataset = tf.data.Dataset.from_generator(
                _record_generator,
                output_signature=(
                    image_spec,
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                ),
            )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(loaded_records), reshuffle_each_iteration=True
        )

    if repeat:
        # Keep training streams alive across epochs when Keras needs multiple passes.
        dataset = dataset.repeat()

    if augment:
        if augment_mode not in {"standard", "hard_preview"}:
            raise ValueError(f"Unsupported augment_mode: {augment_mode}")
        if polar_dualview_model:
            def _apply_shared_color_aug(
                image: tf.Tensor,
                *,
                brightness_delta: tf.Tensor,
                contrast_factor: tf.Tensor,
                saturation_factor: tf.Tensor,
                gamma: tf.Tensor,
                noise_std: float,
                blur_scale: float | None,
            ) -> tf.Tensor:
                """Apply the same photometric perturbation recipe to one image."""
                image = tf.image.adjust_brightness(image, brightness_delta)
                image = tf.image.adjust_contrast(image, contrast_factor)
                image = tf.image.adjust_saturation(image, saturation_factor)
                image = tf.clip_by_value(image, 0.0, 1.0)
                image = tf.pow(tf.clip_by_value(image, 0.0, 1.0), gamma)

                if blur_scale is not None:
                    image_shape = tf.shape(image)
                    image_h = image_shape[0]
                    image_w = image_shape[1]
                    down_h = tf.maximum(
                        2, tf.cast(tf.cast(image_h, tf.float32) * blur_scale, tf.int32)
                    )
                    down_w = tf.maximum(
                        2, tf.cast(tf.cast(image_w, tf.float32) * blur_scale, tf.int32)
                    )
                    image = tf.image.resize(image, [down_h, down_w], method="bilinear")
                    image = tf.image.resize(image, [image_h, image_w], method="bilinear")

                image = image + tf.random.normal(tf.shape(image), stddev=noise_std)
                return tf.clip_by_value(image, 0.0, 1.0)

            def _augment_dualview_inputs(
                inputs: dict[str, tf.Tensor],
            ) -> dict[str, tf.Tensor]:
                """Augment the paired full and polar views with shared settings."""
                if augment_mode == "hard_preview":
                    brightness_delta = tf.random.uniform([], minval=-0.12, maxval=0.05)
                    contrast_factor = tf.random.uniform([], minval=0.60, maxval=1.12)
                    saturation_factor = tf.random.uniform([], minval=0.82, maxval=1.05)
                    gamma = tf.where(
                        tf.random.uniform([]) < 0.75,
                        tf.random.uniform([], minval=1.10, maxval=2.30),
                        tf.random.uniform([], minval=0.80, maxval=1.02),
                    )
                    noise_std = 0.020
                    blur_scale = 0.82 if tf.random.uniform([]) < 0.65 else None
                else:
                    brightness_delta = tf.random.uniform([], minval=-0.18, maxval=0.18)
                    contrast_factor = tf.random.uniform([], minval=0.75, maxval=1.25)
                    saturation_factor = tf.random.uniform([], minval=0.85, maxval=1.15)
                    gamma = tf.where(
                        tf.random.uniform([]) < 0.25,
                        tf.random.uniform([], minval=1.00, maxval=2.20),
                        tf.where(
                            tf.random.uniform([]) < 0.25,
                            tf.random.uniform([], minval=0.60, maxval=1.00),
                            1.0,
                        ),
                    )
                    noise_std = 0.015
                    blur_scale = None

                full_image = _apply_shared_color_aug(
                    inputs["full_image"],
                    brightness_delta=brightness_delta,
                    contrast_factor=contrast_factor,
                    saturation_factor=saturation_factor,
                    gamma=gamma,
                    noise_std=noise_std,
                    blur_scale=blur_scale,
                )
                polar_image = _apply_shared_color_aug(
                    inputs["polar_image"],
                    brightness_delta=brightness_delta,
                    contrast_factor=contrast_factor,
                    saturation_factor=saturation_factor,
                    gamma=gamma,
                    noise_std=noise_std,
                    blur_scale=blur_scale,
                )
                return {"full_image": full_image, "polar_image": polar_image}

            if use_weights:
                dataset = dataset.map(
                    lambda x, y, w: (_augment_dualview_inputs(x), y, w),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
            else:
                dataset = dataset.map(
                    lambda x, y: (_augment_dualview_inputs(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
        else:
            def _augment_strong(image: tf.Tensor) -> tf.Tensor:
                """Apply strong augmentation matching board camera reality."""
                image_shape = tf.shape(image)
                image_h = image_shape[0]
                image_w = image_shape[1]

                # Crop jitter
                scale = tf.random.uniform([], minval=0.90, maxval=1.0, dtype=tf.float32)
                crop_h = tf.maximum(
                    2, tf.cast(tf.cast(image_h, tf.float32) * scale, tf.int32)
                )
                crop_w = tf.maximum(
                    2, tf.cast(tf.cast(image_w, tf.float32) * scale, tf.int32)
                )
                image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
                image = tf.image.resize(image, [image_h, image_w])

                # Brightness / exposure
                image = tf.image.random_brightness(image, max_delta=0.20)
                image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
                image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
                image = tf.clip_by_value(image, 0.0, 1.0)

                # Gamma augmentation
                gamma_dark = tf.random.uniform(
                    [], minval=1.0, maxval=2.2, dtype=tf.float32
                )
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

                # Glare blobs
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

                # Gaussian noise
                image = image + tf.random.normal(tf.shape(image), stddev=0.015)
                image = tf.clip_by_value(image, 0.0, 1.0)
                return image

            def _augment_hard_preview(image: tf.Tensor) -> tf.Tensor:
                """Apply heavier low-contrast augmentation for preview-style frames."""
                image_shape = tf.shape(image)
                image_h = image_shape[0]
                image_w = image_shape[1]

                # Preview frames tend to be slightly tighter and flatter, so we
                # bias the crop jitter toward a smaller crop than the standard path.
                scale = tf.random.uniform([], minval=0.82, maxval=0.98, dtype=tf.float32)
                crop_h = tf.maximum(
                    2, tf.cast(tf.cast(image_h, tf.float32) * scale, tf.int32)
                )
                crop_w = tf.maximum(
                    2, tf.cast(tf.cast(image_w, tf.float32) * scale, tf.int32)
                )
                image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
                image = tf.image.resize(image, [image_h, image_w])

                # Preview-style captures usually have weaker chroma, so grayscale is
                # a legitimate training signal rather than a destructive transform.
                if tf.random.uniform([]) < 0.70:
                    image = tf.image.rgb_to_grayscale(image)
                    image = tf.image.grayscale_to_rgb(image)

                # Flatten contrast and exposure more aggressively than the standard
                # path so the model sees the low-contrast end of the distribution.
                image = tf.image.random_brightness(image, max_delta=0.12)
                image = tf.image.random_contrast(image, lower=0.55, upper=1.18)
                image = tf.clip_by_value(image, 0.0, 1.0)

                # Push a lot of samples darker than the default augmentation does.
                gamma_dark = tf.random.uniform(
                    [], minval=1.15, maxval=2.40, dtype=tf.float32
                )
                gamma_bright = tf.random.uniform(
                    [], minval=0.80, maxval=1.05, dtype=tf.float32
                )
                apply_dark = tf.random.uniform([]) < 0.70
                gamma = tf.where(apply_dark, gamma_dark, gamma_bright)
                image = tf.pow(tf.clip_by_value(image, 0.0, 1.0), gamma)
                image = tf.clip_by_value(image, 0.0, 1.0)

                # Add a little blur by downsampling and upsampling; preview captures
                # often have softer edges than the cleaner rectified images.
                if tf.random.uniform([]) < 0.65:
                    down_h = tf.maximum(
                        2,
                        tf.cast(tf.cast(image_h, tf.float32) * 0.78, tf.int32),
                    )
                    down_w = tf.maximum(
                        2,
                        tf.cast(tf.cast(image_w, tf.float32) * 0.78, tf.int32),
                    )
                    image = tf.image.resize(image, [down_h, down_w], method="bilinear")
                    image = tf.image.resize(image, [image_h, image_w], method="bilinear")

                # Keep the glare/noise path, but make it a bit more aggressive so the
                # network learns to survive the reflective preview frames.
                BLOB_RES = 32
                mask = tf.zeros([BLOB_RES, BLOB_RES, 1], dtype=tf.float32)
                for _ in range(4):
                    active = tf.cast(tf.random.uniform([]) < 0.65, tf.float32)
                    cx = tf.random.uniform([], 2, BLOB_RES - 2, dtype=tf.float32)
                    cy = tf.random.uniform([], 2, BLOB_RES - 2, dtype=tf.float32)
                    brightness = tf.random.uniform([], 0.55, 1.0) * active
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

                image = image + tf.random.normal(tf.shape(image), stddev=0.02)
                image = tf.clip_by_value(image, 0.0, 1.0)
                return image

            if use_weights:
                if augment_mode == "hard_preview":
                    dataset = dataset.map(
                        lambda x, y, w: (_augment_hard_preview(x), y, w),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
                else:
                    dataset = dataset.map(
                        lambda x, y, w: (_augment_strong(x), y, w),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
            else:
                if augment_mode == "hard_preview":
                    dataset = dataset.map(
                        lambda x, y: (_augment_hard_preview(x), y),
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


def train_all_data_baseline(
    df: pd.DataFrame,
    output_dir: Path,
    epochs: int = 40,
    warmup_epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    fine_tune_lr: float = 1e-5,
    alpha: float = 1.0,
    head_units: int = 128,
    dropout: float = 0.2,
    seed: int = 42,
    crop_boxes: dict[str, tuple[float, float, float, float]] | None = None,
    init_model_path: Path | None = None,
    mobilenet_unfreeze_last_n: int = 0,
    mobilenet_freeze_batchnorm: bool = True,
    linear_output: bool = False,
    aux_head_kind: str | None = None,
    dual_resolution_model: bool = False,
    polar_only_model: bool = False,
    polar_dualview_model: bool = False,
    polar_sweep_distribution_model: bool = False,
    dual_resolution_crop_ratio: float = 0.78,
    ordinal_threshold_step: float = 10.0,
    interval_bin_width: float = 5.0,
    sweep_distribution_bins: int = 81,
    sweep_distribution_sigma_bins: float = 1.75,
    aux_loss_weight: float = 0.35,
    augment_mode: str = "standard",
) -> dict[str, Any]:
    """Train on all data using proven canonical baseline pipeline."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)

    if sum(
        int(flag)
        for flag in [
            dual_resolution_model,
            polar_only_model,
            polar_dualview_model,
            polar_sweep_distribution_model,
        ]
    ) > 1:
        raise ValueError(
            "dual_resolution_model, polar_only_model, polar_dualview_model, and "
            "polar_sweep_distribution_model "
            "cannot be enabled together"
        )
    if (polar_dualview_model or polar_only_model) and aux_head_kind is not None:
        raise ValueError(
            "polar-specific models currently support scalar regression only"
        )
    if polar_sweep_distribution_model and aux_head_kind not in {None, "sweep_distribution"}:
        raise ValueError(
            "polar_sweep_distribution_model currently supports sweep_distribution "
            "targets only"
        )
    if sweep_distribution_bins < 2:
        raise ValueError("sweep_distribution_bins must be >= 2.")
    if sweep_distribution_sigma_bins <= 0.0:
        raise ValueError("sweep_distribution_sigma_bins must be > 0.")

    # Create split: 70/15/15
    # Use broader 10°C bins for stratification to avoid single-sample bins
    broad_bins = (df["value"] / 10).astype(int)

    # Check if any bin has <2 samples — if so, use random split instead
    bin_counts = broad_bins.value_counts()
    if (bin_counts < 2).any():
        logger.warning(
            "Some bins have <2 samples, using random split instead of stratified"
        )
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

    # Compute sample weights using broader 10°C bins
    train_df = create_value_bins(train_df, bin_size=10.0)
    bin_weights = compute_sample_weights(train_df)
    if "sample_weight" in train_df.columns:
        train_df["sample_weight"] = (
            train_df["sample_weight"].astype(np.float32).values * bin_weights
        )
    else:
        train_df["sample_weight"] = bin_weights
    logger.info(
        f"Sample weight range: {train_df['sample_weight'].min():.3f} to {train_df['sample_weight'].max():.3f}"
    )

    # Keep the polar-geometry paths augmentation-light so we can isolate the
    # geometry change before adding more knobs.
    train_augment = not (
        polar_dualview_model or polar_only_model or polar_sweep_distribution_model
    )

    # Create datasets
    train_ds = create_dataset(
        train_df,
        batch_size,
        shuffle=True,
        repeat=True,
        use_weights=True,
        augment=train_augment,
        augment_mode=augment_mode,
        crop_boxes=crop_boxes,
        aux_head_kind=aux_head_kind,
        polar_only_model=polar_only_model,
        polar_dualview_model=polar_dualview_model,
        polar_sweep_distribution_model=polar_sweep_distribution_model,
        ordinal_threshold_step=ordinal_threshold_step,
        interval_bin_width=interval_bin_width,
        sweep_distribution_bins=sweep_distribution_bins,
        sweep_distribution_sigma_bins=sweep_distribution_sigma_bins,
    )
    val_ds = create_dataset(
        val_df,
        batch_size,
        shuffle=False,
        repeat=False,
        use_weights=False,
        augment=False,
        augment_mode=augment_mode,
        crop_boxes=crop_boxes,
        aux_head_kind=aux_head_kind,
        polar_only_model=polar_only_model,
        polar_dualview_model=polar_dualview_model,
        polar_sweep_distribution_model=polar_sweep_distribution_model,
        ordinal_threshold_step=ordinal_threshold_step,
        interval_bin_width=interval_bin_width,
        sweep_distribution_bins=sweep_distribution_bins,
        sweep_distribution_sigma_bins=sweep_distribution_sigma_bins,
    )
    test_ds = create_dataset(
        test_df,
        batch_size,
        shuffle=False,
        repeat=False,
        use_weights=False,
        augment=False,
        augment_mode=augment_mode,
        crop_boxes=crop_boxes,
        aux_head_kind=aux_head_kind,
        polar_only_model=polar_only_model,
        polar_dualview_model=polar_dualview_model,
        polar_sweep_distribution_model=polar_sweep_distribution_model,
        ordinal_threshold_step=ordinal_threshold_step,
        interval_bin_width=interval_bin_width,
        sweep_distribution_bins=sweep_distribution_bins,
        sweep_distribution_sigma_bins=sweep_distribution_sigma_bins,
    )

    # Avoid duplicating the ImageNet bootstrap when a warm-start checkpoint is provided.
    use_pretrained_backbone = init_model_path is None

    # Build model
    if aux_head_kind == "ordinal":
        logger.info(
            f"Building MobileNetV2 ordinal model (alpha={alpha}, head_units={head_units}, "
            f"dropout={dropout}, threshold_step={ordinal_threshold_step})..."
        )
        model = build_mobilenetv2_ordinal_model(
            image_height=IMAGE_SIZE,
            image_width=IMAGE_SIZE,
            alpha=alpha,
            head_units=head_units,
            head_dropout=dropout,
            pretrained=use_pretrained_backbone,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            threshold_step=ordinal_threshold_step,
        )
    elif aux_head_kind == "interval":
        if dual_resolution_model:
            logger.info(
                "Building MobileNetV2 dual-resolution interval model "
                f"(alpha={alpha}, head_units={head_units}, crop_ratio={dual_resolution_crop_ratio:.2f}, "
                f"dropout={dropout}, bin_width={interval_bin_width})..."
            )
            model = build_mobilenetv2_dual_resolution_interval_model(
                image_height=IMAGE_SIZE,
                image_width=IMAGE_SIZE,
                alpha=alpha,
                head_units=head_units,
                head_dropout=dropout,
                pretrained=use_pretrained_backbone,
                value_min=VALUE_MIN,
                value_max=VALUE_MAX,
                bin_width=interval_bin_width,
                crop_ratio=dual_resolution_crop_ratio,
            )
        else:
            logger.info(
                f"Building MobileNetV2 interval model (alpha={alpha}, head_units={head_units}, "
                f"dropout={dropout}, bin_width={interval_bin_width})..."
            )
            model = build_mobilenetv2_interval_model(
                image_height=IMAGE_SIZE,
                image_width=IMAGE_SIZE,
                alpha=alpha,
                head_units=head_units,
                head_dropout=dropout,
                pretrained=use_pretrained_backbone,
                value_min=VALUE_MIN,
                value_max=VALUE_MAX,
                bin_width=interval_bin_width,
            )
    elif aux_head_kind == "sweep_distribution" and not polar_sweep_distribution_model:
        logger.info(
            "Building MobileNetV2 sweep-distribution model "
            f"(alpha={alpha}, head_units={head_units}, dropout={dropout}, "
            f"bins={sweep_distribution_bins}, sigma={sweep_distribution_sigma_bins})..."
        )
        model = build_mobilenetv2_sweep_distribution_model(
            image_height=IMAGE_SIZE,
            image_width=IMAGE_SIZE,
            alpha=alpha,
            head_units=head_units,
            head_dropout=dropout,
            pretrained=use_pretrained_backbone,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            num_bins=sweep_distribution_bins,
        )
    elif polar_sweep_distribution_model:
        logger.info(
            "Building MobileNetV2 polar sweep-distribution model "
            f"(alpha={alpha}, head_units={head_units}, dropout={dropout}, "
            f"bins={sweep_distribution_bins}, sigma={sweep_distribution_sigma_bins})..."
        )
        model = build_mobilenetv2_polar_sweep_distribution_model(
            image_height=IMAGE_SIZE,
            image_width=IMAGE_SIZE,
            alpha=alpha,
            head_units=head_units,
            head_dropout=dropout,
            pretrained=use_pretrained_backbone,
            value_min=VALUE_MIN,
            value_max=VALUE_MAX,
            num_bins=sweep_distribution_bins,
        )
    else:
        if polar_only_model:
            logger.info(
                "Building MobileNetV2 polar-only model "
                f"(alpha={alpha}, head_units={head_units}, dropout={dropout}, "
                f"linear_output={linear_output})..."
            )
            model = build_mobilenetv2_polar_regression_model(
                image_height=IMAGE_SIZE,
                image_width=IMAGE_SIZE,
                alpha=alpha,
                head_units=head_units,
                head_dropout=dropout,
                pretrained=use_pretrained_backbone,
                linear_output=linear_output,
                value_min=VALUE_MIN,
                value_max=VALUE_MAX,
            )
        elif polar_dualview_model:
            logger.info(
                "Building MobileNetV2 polar-dualview model "
                f"(alpha={alpha}, head_units={head_units}, dropout={dropout}, "
                f"linear_output={linear_output})..."
            )
            model = build_mobilenetv2_polar_dualview_regression_model(
                image_height=IMAGE_SIZE,
                image_width=IMAGE_SIZE,
                alpha=alpha,
                head_units=head_units,
                head_dropout=dropout,
                pretrained=use_pretrained_backbone,
                linear_output=linear_output,
                value_min=VALUE_MIN,
                value_max=VALUE_MAX,
            )
        elif dual_resolution_model:
            logger.info(
                "Building MobileNetV2 dual-resolution model "
                f"(alpha={alpha}, head_units={head_units}, crop_ratio={dual_resolution_crop_ratio:.2f}, "
                f"dropout={dropout}, linear_output={linear_output})..."
            )
            model = build_mobilenetv2_dual_resolution_regression_model(
                image_height=IMAGE_SIZE,
                image_width=IMAGE_SIZE,
                alpha=alpha,
                head_units=head_units,
                head_dropout=dropout,
                pretrained=use_pretrained_backbone,
                linear_output=linear_output,
                value_min=VALUE_MIN,
                value_max=VALUE_MAX,
                crop_ratio=dual_resolution_crop_ratio,
            )
        else:
            logger.info(
                f"Building MobileNetV2 (alpha={alpha}, head_units={head_units}, dropout={dropout})..."
            )
            model = build_mobilenetv2_regression_model(
                image_height=IMAGE_SIZE,
                image_width=IMAGE_SIZE,
                alpha=alpha,
                head_units=head_units,
                head_dropout=dropout,
                pretrained=use_pretrained_backbone,
                linear_output=linear_output,
                value_min=VALUE_MIN,
                value_max=VALUE_MAX,
            )

    if init_model_path is not None:
        if not init_model_path.exists():
            raise FileNotFoundError(f"Init model not found: {init_model_path}")
        logger.info(f"Warm-starting from init model: {init_model_path}")
        if init_model_path.name.endswith(".weights.h5"):
            # Weights-only checkpoints avoid the heavier full-model deserialization path.
            # We allow shape mismatches so the shared MobileNetV2 trunk is reused
            # while the new dual-resolution heads are initialized fresh.
            model.load_weights(str(init_model_path), skip_mismatch=True)
            logger.info(
                "Loaded warm-start weights from weights-only checkpoint with skip_mismatch=True."
            )
        else:
            init_model = keras.models.load_model(
                init_model_path,
                compile=False,
                safe_mode=False,
                custom_objects={
                    "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
                },
            )
            loaded_layers = 0
            skipped_layers = 0
            source_layers = {layer.name: layer for layer in init_model.layers}
            for layer in model.layers:
                source_layer = source_layers.get(layer.name)
                if source_layer is None:
                    continue
                source_weights = source_layer.get_weights()
                target_weights = layer.get_weights()
                if not source_weights or not target_weights:
                    continue
                if len(source_weights) != len(target_weights):
                    skipped_layers += 1
                    continue
                if any(
                    source.shape != target.shape
                    for source, target in zip(source_weights, target_weights)
                ):
                    skipped_layers += 1
                    continue
                layer.set_weights(source_weights)
                loaded_layers += 1
            logger.info(
                "Loaded warm-start weights into %d matching layers (%d skipped).",
                loaded_layers,
                skipped_layers,
            )

    def _compile_current_model(current_lr: float) -> None:
        """Compile the current model with the right head-specific losses."""
        optimizer = keras.optimizers.AdamW(
            learning_rate=current_lr,
            weight_decay=1e-4,
            clipnorm=1.0,
        )
        if aux_head_kind == "ordinal":
            model.compile(
                optimizer=optimizer,
                loss={
                    "gauge_value": keras.losses.Huber(delta=1.0),
                    "ordinal_logits": keras.losses.BinaryCrossentropy(from_logits=True),
                },
                loss_weights={
                    "gauge_value": 1.0,
                    "ordinal_logits": aux_loss_weight,
                },
                metrics={
                    "gauge_value": [
                        keras.metrics.MeanAbsoluteError(name="mae"),
                        keras.metrics.MeanSquaredError(name="mse"),
                    ],
                    "ordinal_logits": [
                        keras.metrics.BinaryAccuracy(name="acc", threshold=0.0),
                    ],
                },
            )
        elif aux_head_kind == "sweep_distribution":
            model.compile(
                optimizer=optimizer,
                loss={
                    "gauge_value": keras.losses.Huber(delta=1.0),
                    "sweep_distribution_logits": keras.losses.CategoricalCrossentropy(
                        from_logits=True
                    ),
                },
                loss_weights={
                    "gauge_value": 1.0,
                    "sweep_distribution_logits": aux_loss_weight,
                },
                metrics={
                    "gauge_value": [
                        keras.metrics.MeanAbsoluteError(name="mae"),
                        keras.metrics.MeanSquaredError(name="mse"),
                    ],
                    "sweep_distribution_logits": [
                        keras.metrics.CategoricalAccuracy(name="acc"),
                    ],
                },
            )
        elif aux_head_kind == "interval":
            model.compile(
                optimizer=optimizer,
                loss={
                    "gauge_value": keras.losses.Huber(delta=1.0),
                    "interval_logits": keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ),
                },
                loss_weights={
                    "gauge_value": 1.0,
                    "interval_logits": aux_loss_weight,
                },
                metrics={
                    "gauge_value": [
                        keras.metrics.MeanAbsoluteError(name="mae"),
                        keras.metrics.MeanSquaredError(name="mse"),
                    ],
                    "interval_logits": [
                        keras.metrics.SparseCategoricalAccuracy(name="acc"),
                    ],
                },
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.Huber(delta=1.0),
                metrics=["mae", "mse"],
            )

    _compile_current_model(learning_rate)

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

    _configure_mobilenet_backbone_trainability(
        base_model,
        trainable=False,
        unfreeze_last_n=0,
        freeze_batchnorm=mobilenet_freeze_batchnorm,
    )

    # Callbacks
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_weights.weights.h5"
    monitor_metric = "val_gauge_value_mae" if aux_head_kind is not None else "val_mae"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor_metric,
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode="min",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Phase 1: Warmup
    logger.info(f"\n=== Phase 1: Warmup ({warmup_epochs} epochs, frozen backbone) ===")
    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=int(np.ceil(len(train_df) / batch_size)),
        validation_steps=int(np.ceil(len(val_df) / batch_size)),
        epochs=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: Fine-tune
    fine_tune_epochs = epochs - warmup_epochs
    logger.info(
        f"\n=== Phase 2: Fine-tune ({fine_tune_epochs} epochs, unfrozen backbone) ==="
    )

    _configure_mobilenet_backbone_trainability(
        base_model,
        trainable=True,
        unfreeze_last_n=mobilenet_unfreeze_last_n,
        freeze_batchnorm=mobilenet_freeze_batchnorm,
    )
    _compile_current_model(fine_tune_lr)

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=int(np.ceil(len(train_df) / batch_size)),
        validation_steps=int(np.ceil(len(val_df) / batch_size)),
        epochs=epochs,
        initial_epoch=warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    logger.info("\n=== Evaluating on test set ===")
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    logger.info(f"Test metrics: {test_metrics}")

    # Save model
    model.save(output_dir / "model.keras")
    logger.info(f"Saved model to {output_dir / 'model.keras'}")

    # Save history
    history_combined = {
        "warmup": {
            k: [float(v) for v in vals] for k, vals in history_warmup.history.items()
        },
        "finetune": {
            k: [float(v) for v in vals] for k, vals in history_finetune.history.items()
        },
    }
    with open(output_dir / "history.json", "w") as f:
        json.dump(history_combined, f, indent=2)

    # Save predictions
    predictions_raw = model.predict(test_ds, verbose=1)
    if isinstance(predictions_raw, dict):
        predictions = np.asarray(predictions_raw["gauge_value"], dtype=np.float32).reshape(-1)
    else:
        predictions = np.asarray(predictions_raw, dtype=np.float32).reshape(-1)
    test_df = test_df.copy()
    test_df["prediction"] = predictions
    test_df["abs_error"] = np.abs(test_df["prediction"] - test_df["value"])
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Compute detailed metrics
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
    }
    logger.info(f"\nDetailed Test Metrics:")
    for k, v in metrics.items():
        if "pct" in k:
            logger.info(f"  {k}: {v*100:.1f}%")
        else:
            logger.info(f"  {k}: {v:.2f}°C")

    with open(output_dir / "metrics.json", "w") as f:
        eval_metrics = {
            f"eval_{key}": float(value)
            for key, value in test_metrics.items()
            if isinstance(value, (int, float, np.floating))
        }
        json.dump({**metrics, **eval_metrics}, f, indent=2)

    return {"model": model, "metrics": metrics, "output_dir": output_dir}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on all data using proven baseline pipeline"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional prebuilt scalar manifest CSV to train from.",
    )
    parser.add_argument(
        "--precomputed-crop-boxes",
        type=str,
        default=None,
        help="Optional CSV of rectifier crop boxes keyed by image_path.",
    )
    parser.add_argument(
        "--no-gpu-memory-growth",
        action="store_true",
        default=False,
        help="Skip TensorFlow GPU memory-growth probing for WSL stability.",
    )
    parser.add_argument(
        "--init-model",
        type=str,
        default=None,
        help="Optional Keras checkpoint used to warm-start the regressor.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Total epochs")
    parser.add_argument("--warmup-epochs", type=int, default=8, help="Warmup epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate used for the warmup phase.",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-5,
        help="Learning rate used for the fine-tuning phase.",
    )
    parser.add_argument(
        "--mobilenet-unfreeze-last-n",
        type=int,
        default=0,
        help="Number of MobileNetV2 backbone layers to unfreeze during fine-tuning.",
    )
    parser.add_argument(
        "--mobilenet-freeze-batchnorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep MobileNetV2 BatchNorm layers in inference mode during fine-tuning.",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="MobileNetV2 width multiplier"
    )
    parser.add_argument(
        "--head-units",
        type=int,
        default=128,
        help="Dense units in the regression head.",
    )
    parser.add_argument(
        "--dual-resolution-crop-ratio",
        type=float,
        default=0.78,
        help="Center-crop ratio used by the dual-resolution branch.",
    )
    parser.add_argument(
        "--aux-head-kind",
        type=str,
        default="none",
        choices=["none", "ordinal", "interval", "sweep_distribution"],
        help="Optional auxiliary head to add to the scalar model.",
    )
    parser.add_argument(
        "--ordinal-threshold-step",
        type=float,
        default=10.0,
        help="Threshold spacing for the ordinal auxiliary head.",
    )
    parser.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.35,
        help="Loss weight for the auxiliary ordinal head.",
    )
    parser.add_argument(
        "--interval-bin-width",
        type=float,
        default=5.0,
        help="Bin width for the interval auxiliary head.",
    )
    parser.add_argument(
        "--sweep-distribution-bins",
        type=int,
        default=81,
        help="Number of bins used by the sweep-distribution auxiliary head.",
    )
    parser.add_argument(
        "--sweep-distribution-sigma-bins",
        type=float,
        default=1.75,
        help="Gaussian width, in bins, used to build the sweep-distribution target.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Head dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--linear-output",
        action="store_true",
        help="Use a linear regression head instead of sigmoid-rescaled output.",
    )
    parser.add_argument(
        "--dual-resolution-model",
        action="store_true",
        help="Use the new dual-resolution MobileNetV2 architecture.",
    )
    parser.add_argument(
        "--polar-dualview-model",
        action="store_true",
        help="Use the new full-frame + polar-unwrapped MobileNetV2 architecture.",
    )
    parser.add_argument(
        "--polar-only-model",
        action="store_true",
        help="Use a single-input polar-unwrapped MobileNetV2 architecture.",
    )
    parser.add_argument(
        "--polar-sweep-distribution-model",
        action="store_true",
        help="Use a single-input polar sweep-distribution MobileNetV2 architecture.",
    )
    parser.add_argument(
        "--augment-mode",
        type=str,
        choices=["standard", "hard_preview"],
        default="standard",
        help="Choose the train-time augmentation profile.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_gpu_memory_growth:
        # WSL can stall on the GPU probe, so let TensorFlow initialize lazily.
        gpus = []
    else:
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

    crop_boxes = None
    if args.precomputed_crop_boxes:
        boxes_path = Path(args.precomputed_crop_boxes)
        if not boxes_path.is_absolute():
            boxes_path = PROJECT_ROOT / boxes_path
        crop_boxes = load_precomputed_crop_boxes(boxes_path, REPO_ROOT)
        df.to_csv(output_dir / "merged_dataset_with_boxes.csv", index=False)

    init_model_path = Path(args.init_model) if args.init_model else None
    if init_model_path is not None and not init_model_path.is_absolute():
        init_model_path = PROJECT_ROOT / init_model_path

    train_all_data_baseline(
        df,
        output_dir=output_dir,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fine_tune_lr=args.fine_tune_lr,
        alpha=args.alpha,
        head_units=args.head_units,
        dropout=args.dropout,
        seed=args.seed,
        crop_boxes=crop_boxes,
        init_model_path=init_model_path,
        mobilenet_unfreeze_last_n=args.mobilenet_unfreeze_last_n,
        mobilenet_freeze_batchnorm=args.mobilenet_freeze_batchnorm,
        linear_output=args.linear_output,
        aux_head_kind=None if args.aux_head_kind == "none" else args.aux_head_kind,
        dual_resolution_model=args.dual_resolution_model,
        polar_only_model=args.polar_only_model,
        polar_dualview_model=args.polar_dualview_model,
        polar_sweep_distribution_model=args.polar_sweep_distribution_model,
        dual_resolution_crop_ratio=args.dual_resolution_crop_ratio,
        ordinal_threshold_step=args.ordinal_threshold_step,
        interval_bin_width=args.interval_bin_width,
        sweep_distribution_bins=args.sweep_distribution_bins,
        sweep_distribution_sigma_bins=args.sweep_distribution_sigma_bins,
        aux_loss_weight=args.aux_loss_weight,
        augment_mode=args.augment_mode,
    )

    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
