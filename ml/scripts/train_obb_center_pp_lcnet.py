"""PP-LCNet joint OBB + center localizer at 320x320.

PP-LCNet is a lightweight CNN optimized for CPU inference. This script trains
a PP-LCNet-style backbone with dual OBB and center prediction heads.

Target: <2.5 MB int8 TFLite, <10 px center MAE at 320x320.

Usage:
  cd ml && poetry run python scripts/train_obb_center_pp_lcnet.py
"""

from __future__ import annotations

import os as _os
_GPU_MEMORY_LIMIT_MB = 3900

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=_GPU_MEMORY_LIMIT_MB)],
    )
del _os, _GPU_MEMORY_LIMIT_MB

import keras
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.training import (
    TrainingExample,
    _compute_edge_weights,
    _compute_fullframe_obb_params,
)

# --- Paths -------------------------------------------------------------------
ML_ROOT: Path = PROJECT_ROOT
ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "training"
MANIFEST_PATH: Path = ML_ROOT / "data" / "merged_geometry_board_manifest.csv"
AI_CENTERS_CSV: Path = ML_ROOT / "data" / "ai_annotated_board_captures.csv"
ANNOTATE_30_CSV: Path = ML_ROOT / "data" / "annotate_30" / "annotations.csv"
ANNOTATE_BATCH2_CSV: Path = (
    ML_ROOT / "data" / "annotate_batch2" / "annotations_batch2.csv"
)
CAPTURED_IMAGES_DIR: Path = ML_ROOT / "data" / "captured_images_320"
YUV_LABELS_CSV: Path = ML_ROOT / "data" / "capture_2026-06-07_labels_v2.csv"

# --- Constants ---------------------------------------------------------------
GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO: float = 0.3076

IMAGE_HEIGHT: int = 320
IMAGE_WIDTH: int = 320
SEED: int = 42
BATCH_SIZE: int = 4
EPOCHS: int = 150
LEARNING_RATE: float = 1e-3
VAL_FRACTION: float = 0.20
HEAD_UNITS_1: int = 256
HEAD_UNITS_2: int = 128
HEAD_DROPOUT_1: float = 0.25
HEAD_DROPOUT_2: float = 0.15
CENTER_LOSS_WEIGHT: float = 3.0

POSITIONAL_AUG_CY_MIN: float = 0.15
POSITIONAL_AUG_CY_MAX: float = 0.95
POSITIONAL_AUG_CX_MIN: float = 0.15
POSITIONAL_AUG_CX_MAX: float = 0.85


def build_pp_lcnet_obb_center_model(
    image_height: int = 320,
    image_width: int = 320,
    *,
    scale: float = 1.0,
    head_units_1: int = 256,
    head_units_2: int = 128,
    head_dropout_1: float = 0.25,
    head_dropout_2: float = 0.15,
) -> keras.Model:
    """Build PP-LCNet-style backbone with dual OBB + center heads.

    PP-LCNet uses large-kernel depthwise separable convolutions for efficiency.
    """
    inputs = keras.Input(shape=(image_height, image_width, 3), name="image")

    # CoordConv positional encoding
    y_grid = tf.linspace(-1.0, 1.0, image_height)
    x_grid = tf.linspace(-1.0, 1.0, image_width)
    yy, xx = tf.meshgrid(y_grid, x_grid, indexing="ij")
    coords = tf.stack([xx, yy], axis=-1)
    coords = coords[tf.newaxis, ...]
    coord_input = keras.layers.Lambda(
        lambda img: tf.tile(coords, [tf.shape(img)[0], 1, 1, 1]),
        name="coord_channels",
    )(inputs)

    x = keras.layers.Concatenate(name="rgb_coords")([inputs, coord_input])
    x = keras.layers.Conv2D(3, kernel_size=1, use_bias=False, name="coord_proj")(x)

    # PP-LCNet backbone (simplified)
    # Stage 1: 3x3 conv
    x = keras.layers.Conv2D(int(32 * scale), 3, strides=2, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("hard_swish")(x)

    # Stage 2-5: Large kernel DW convolutions
    for filters in [64, 128, 256, 512]:
        f = int(filters * scale)
        # Large kernel depthwise
        x = keras.layers.DepthwiseConv2D(7, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("hard_swish")(x)
        # Pointwise
        x = keras.layers.Conv2D(f, 1, use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("hard_swish")(x)
        # Stride in first block
        if filters == 64:
            x = keras.layers.AveragePooling2D(2)(x)

    # Global pooling
    x = keras.layers.GlobalAveragePooling2D()(x)

    # FC layers
    x = keras.layers.Dense(int(1024 * scale))(x)
    x = keras.layers.Activation("hard_swish")(x)
    x = keras.layers.Dropout(0.2)(x)

    # Shared head
    x = keras.layers.Dense(head_units_1, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout_1)(x)
    x = keras.layers.Dense(head_units_2, activation="swish")(x)
    x = keras.layers.Dropout(head_dropout_2)(x)

    # OBB head
    center_xy_obb = keras.layers.Dense(2, activation="sigmoid", name="obb_center_xy")(x)
    size_wh = keras.layers.Dense(2, activation="sigmoid", name="obb_size_wh")(x)
    angle_raw = keras.layers.Dense(2, name="obb_angle_raw")(x)
    angle_sincos = keras.layers.UnitNormalization(axis=-1, name="obb_angle_sincos")(angle_raw)
    obb_params = keras.layers.Concatenate(name="obb_params")([center_xy_obb, size_wh, angle_sincos])

    # Center head
    center_xy = keras.layers.Dense(2, activation="sigmoid", name="center_xy")(x)

    model = keras.Model(
        inputs=inputs,
        outputs={"obb_params": obb_params, "center_xy": center_xy},
        name=f"pp_lcnet_obb_center_s{scale:.1f}_h{head_units_1}-{head_units_2}",
    )
    return model


# --- Image preprocessing -----------------------------------------------------

def _preprocess_colour(image: tf.Tensor, image_height: int, image_width: int) -> tf.Tensor:
    """Resize-with-pad in RGB colour space."""
    image = tf.cast(image, tf.float32)
    crop_h = tf.shape(image)[0]
    crop_w = tf.shape(image)[1]
    scale = tf.minimum(
        tf.cast(image_height, tf.float32) / tf.cast(crop_h, tf.float32),
        tf.cast(image_width, tf.float32) / tf.cast(crop_w, tf.float32),
    )
    scaled_h = tf.cast(tf.cast(crop_h, tf.float32) * scale, tf.int32)
    scaled_w = tf.cast(tf.cast(crop_w, tf.float32) * scale, tf.int32)
    scaled_h = tf.maximum(scaled_h, 1)
    scaled_w = tf.maximum(scaled_w, 1)
    resized = tf.image.resize(image, [scaled_h, scaled_w], method="nearest")
    pad_y = (image_height - scaled_h) // 2
    pad_x = (image_width - scaled_w) // 2
    pad_bottom = image_height - scaled_h - pad_y
    pad_right = image_width - scaled_w - pad_x
    padded = tf.pad(resized, [[pad_y, pad_bottom], [pad_x, pad_right], [0, 0]])
    padded = tf.ensure_shape(padded, [image_height, image_width, 3])
    return padded / 255.0


# --- Data loading helpers ----------------------------------------------------

def _is_original_capture(filename: str) -> bool:
    """Filter out diagnostic derivative images."""
    name = Path(filename).name
    if any(tag in name for tag in (".gray.", "_preview", "_yuy2", "_glare_")):
        return False
    return True


def _load_fullframe_obb_data_colour(
    image_path: tf.Tensor,
    value: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load full image, resize-with-pad, attach OBB and center targets."""
    is_yuv = tf.strings.regex_full_match(image_path, ".*\\.yuv422$")

    def _load_yuv():
        raw = tf.io.read_file(image_path)
        yuyv = tf.io.decode_raw(raw, tf.uint8)
        yuyv = tf.reshape(yuyv, [320, 640])
        y = tf.cast(yuyv[:, 0::2], tf.float32)
        u = tf.cast(yuyv[:, 1::4], tf.float32) - 128.0
        v = tf.cast(yuyv[:, 3::4], tf.float32) - 128.0
        u = tf.repeat(u, 2, axis=1)
        v = tf.repeat(v, 2, axis=1)
        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u
        rgb = tf.stack([r, g, b], axis=-1)
        rgb = tf.clip_by_value(rgb, 0, 255)
        return tf.cast(rgb, tf.uint8)

    def _load_standard():
        image_bytes = tf.io.read_file(image_path)
        return tf.io.decode_image(image_bytes, channels=3, expand_animations=False)

    image = tf.cond(is_yuv, _load_yuv, _load_standard)
    image = tf.ensure_shape(image, [None, None, 3])
    image = _preprocess_colour(image, image_height, image_width)

    obb_target = tf.cast(obb_params, tf.float32)
    center_target = obb_target[:2]
    sample_weight = tf.cast(weight, tf.float32)

    return (
        image,
        {"obb_params": obb_target, "center_xy": center_target},
        {"obb_params": sample_weight, "center_xy": sample_weight * CENTER_LOSS_WEIGHT},
    )


def _obb_params_from_center_224(cx_224: float, cy_224: float, source_size: int = 224) -> np.ndarray:
    """Compute 320x320 OBB params from a center given in 224-pixel space."""
    radius = source_size * GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO
    return _compute_fullframe_obb_params(
        source_size, source_size, cx_224, cy_224, radius, radius, 0.0,
        IMAGE_HEIGHT, IMAGE_WIDTH,
    )


# --- Data source loaders -----------------------------------------------------

def _load_manifest_examples(examples: list[TrainingExample], seen: set[str]) -> int:
    """Load board-capture rows from merged geometry manifest."""
    added = 0
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "captured_images" not in row["image_path"]:
                continue
            qf = row.get("quality_flag", "")
            if qf not in ("clean", "manual"):
                continue
            fpath = str(Path(ML_ROOT).parent / row["image_path"])
            fpath = fpath.replace("/captured_images/", "/captured_images_320/")
            if fpath.endswith(".jpg"):
                fpath = fpath[:-4] + ".png"
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)

            source_w = int(float(row["source_width"]))
            source_h = int(float(row["source_height"]))
            cx = float(row["center_x_source"])
            cy = float(row["center_y_source"])
            radius = float(row.get("dial_radius_source", 0))
            if radius <= 0:
                continue

            obb_params = _compute_fullframe_obb_params(
                source_w, source_h, cx, cy, radius, radius, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(TrainingExample(
                image_path=fpath, value=float(row.get("temperature_c", 0)),
                crop_box_xyxy=(0.0, 0.0, float(source_w), float(source_h)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb_params,
            ))
            added += 1
    return added


def _load_pxl_photo_examples(examples: list[TrainingExample], seen: set[str]) -> int:
    """Load PXL phone photos from manifest."""
    added = 0
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "PXL_" not in row["image_path"]:
                continue
            qf = row.get("quality_flag", "")
            if qf not in ("clean", "manual"):
                continue
            fpath = str(Path(ML_ROOT).parent / row["image_path"])
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)

            source_w = int(float(row["source_width"]))
            source_h = int(float(row["source_height"]))
            cx = float(row["center_x_source"])
            cy = float(row["center_y_source"])
            radius = float(row.get("dial_radius_source", 0))
            if radius <= 0:
                continue

            obb_params = _compute_fullframe_obb_params(
                source_w, source_h, cx, cy, radius, radius, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(TrainingExample(
                image_path=fpath, value=float(row.get("temperature_c", 0)),
                crop_box_xyxy=(0.0, 0.0, float(source_w), float(source_h)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb_params,
            ))
            added += 1
    return added


def _load_annotate_csv(csv_path: Path, examples: list[TrainingExample], seen: set[str], *, source_size: int = 224) -> int:
    """Load center-label examples from annotate-style CSV."""
    added = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            if not _is_original_capture(fname):
                continue
            fpath = str(CAPTURED_IMAGES_DIR / fname)
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)

            cx = float(row["cx"])
            cy = float(row["cy"])
            obb_params = _obb_params_from_center_224(cx, cy, source_size)
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, float(source_size), float(source_size)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb_params,
            ))
            added += 1
    return added


def _load_ai_annotated_examples(examples: list[TrainingExample], seen: set[str]) -> int:
    """Load AI-annotated center examples."""
    added = 0
    if not AI_CENTERS_CSV.exists():
        return 0
    with open(AI_CENTERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row["image_path"]
            if not _is_original_capture(rel_path):
                continue
            rel_path = rel_path.replace("captured_images/", "captured_images_320/")
            fpath = str(ML_ROOT / "data" / rel_path)
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)

            cx = float(row["center_x"])
            cy = float(row["center_y"])
            obb_params = _obb_params_from_center_224(cx, cy, 224)
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, 224.0, 224.0),
                needle_unit_xy=(0.0, 0.0), obb_params=obb_params,
            ))
            added += 1
    return added


def _load_yuv_board_captures(examples: list[TrainingExample], seen: set[str]) -> int:
    """Load 320x320 YUV422 board captures with manual labels."""
    added = 0
    if not YUV_LABELS_CSV.exists():
        return 0
    with open(YUV_LABELS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yuv_file = row["image_path"]
            fpath = str(CAPTURED_IMAGES_DIR / yuv_file)
            if fpath in seen:
                continue
            if not Path(fpath).exists():
                continue
            seen.add(fpath)

            cx = float(row["center_x"])
            cy = float(row["center_y"])
            outer_r = float(row["outer_radius"])
            obb_params = _compute_fullframe_obb_params(
                320, 320, cx, cy, outer_r, outer_r, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, 320.0, 320.0),
                needle_unit_xy=(0.0, 0.0), obb_params=obb_params,
            ))
            added += 1
    return added


def _build_all_examples() -> list[TrainingExample]:
    """Load all labelled examples from every available source."""
    examples: list[TrainingExample] = []
    seen: set[str] = set()

    n1 = _load_pxl_photo_examples(examples, seen)
    print(f"  [1] PXL phone photos:         {n1:>4d}")

    n2 = _load_manifest_examples(examples, seen)
    print(f"  [2] Board captures (manifest): {n2:>4d}")

    n3 = _load_yuv_board_captures(examples, seen)
    print(f"  [3] YUV board captures:        {n3:>4d}")

    n4 = _load_annotate_csv(ANNOTATE_30_CSV, examples, seen)
    print(f"  [4] annotate_30:               {n4:>4d}")

    n5 = _load_annotate_csv(ANNOTATE_BATCH2_CSV, examples, seen)
    print(f"  [5] annotate_batch2:           {n5:>4d}")

    n6 = _load_ai_annotated_examples(examples, seen)
    print(f"  [6] AI annotated centers:      {n6:>4d}")

    print(f"  {'─' * 40}")
    print(f"  Total unique examples:         {len(examples):>4d}")
    return examples


# --- Augmentation ------------------------------------------------------------

def _augment_positional(image: tf.Tensor, obb_params: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Shift the gauge to a random position in the frame."""
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)

    orig_cx = obb_params[0]
    orig_cy = obb_params[1]

    target_cx = tf.random.uniform([], POSITIONAL_AUG_CX_MIN, POSITIONAL_AUG_CX_MAX)
    target_cy = tf.random.uniform([], POSITIONAL_AUG_CY_MIN, POSITIONAL_AUG_CY_MAX)

    shift_x = (target_cx - orig_cx) * img_w
    shift_y = (target_cy - orig_cy) * img_h

    transforms = tf.stack([1.0, 0.0, -shift_x, 0.0, 1.0, -shift_y, 0.0, 0.0])

    image_aug = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transforms[tf.newaxis, :],
        output_shape=[tf.shape(image)[0], tf.shape(image)[1]],
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]

    obb_aug = tf.stack([target_cx, target_cy, obb_params[2], obb_params[3], obb_params[4], obb_params[5]])
    return image_aug, obb_aug


def _augment_photometric(image: tf.Tensor) -> tf.Tensor:
    """Apply photometric-only augmentation."""
    image = tf.image.random_brightness(image, max_delta=0.30)
    image = tf.image.random_contrast(image, lower=0.50, upper=1.50)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


_ROTATION_MAX_RAD: float = 0.05


def _augment_rotation(image: tf.Tensor, obb_params: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply small random rotation around the image center."""
    angle = tf.random.uniform([], -_ROTATION_MAX_RAD, _ROTATION_MAX_RAD)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    a0 = cos_a
    a1 = -sin_a
    a2 = 0.5 - 0.5 * cos_a + 0.5 * sin_a
    b0 = sin_a
    b1 = cos_a
    b2 = 0.5 - 0.5 * cos_a - 0.5 * sin_a

    transforms = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])

    image_rot = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transforms[tf.newaxis, :],
        output_shape=[tf.shape(image)[0], tf.shape(image)[1]],
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]

    cx = obb_params[0]
    cy = obb_params[1]
    dx = cx - 0.5
    dy = cy - 0.5
    new_cx = cos_a * dx - sin_a * dy + 0.5
    new_cy = sin_a * dx + cos_a * dy + 0.5

    cos2t = obb_params[4]
    sin2t = obb_params[5]
    cos2a = tf.cos(2.0 * angle)
    sin2a = tf.sin(2.0 * angle)
    new_cos2t = cos2t * cos2a - sin2t * sin2a
    new_sin2t = sin2t * cos2a + cos2t * sin2a

    obb_rot = tf.stack([
        tf.clip_by_value(new_cx, 0.0, 1.0),
        tf.clip_by_value(new_cy, 0.0, 1.0),
        obb_params[2], obb_params[3],
        new_cos2t, new_sin2t,
    ])
    return image_rot, obb_rot


def _augment_train(image: tf.Tensor, y: dict[str, tf.Tensor], w: dict[str, tf.Tensor]) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Full training augmentation: positional → rotation → photometric."""
    image, new_obb = _augment_positional(image, y["obb_params"])
    image, new_obb = _augment_rotation(image, new_obb)
    new_center = new_obb[:2]
    y_aug = {"obb_params": new_obb, "center_xy": new_center}
    image = _augment_photometric(image)
    return image, y_aug, w


# --- Loss --------------------------------------------------------------------

class HuberLoss(keras.losses.Loss):
    """Huber loss with configurable delta."""

    def __init__(self, delta: float = 0.05, reduction: str = "sum_over_batch_size", name: str = "huber_loss"):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        diff = tf.abs(y_true - y_pred)
        quadratic = 0.5 * tf.square(diff)
        linear = self.delta * (diff - 0.5 * self.delta)
        per_element = tf.where(diff <= self.delta, quadratic, linear)
        return tf.reduce_mean(per_element, axis=-1)

    def get_config(self) -> dict:
        return {"delta": self.delta, "name": self.name}


# --- Main training routine ---------------------------------------------------

def main() -> None:
    """Train PP-LCNet OBB + center model."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"obb_center_pp_lcnet_{timestamp}"
    run_dir = ARTIFACTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PP-LCNet + CoordConv Joint OBB + Center Localizer")
    print(f"  scale=1.0, {IMAGE_HEIGHT}x{IMAGE_WIDTH}, head={HEAD_UNITS_1}-{HEAD_UNITS_2}")
    print(f"  Run: {run_dir}")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading all labelled examples...")
    examples = _build_all_examples()
    if len(examples) < 20:
        print("ERROR: Not enough examples. Need at least 20.")
        return

    # Split
    print("\n[2/6] Splitting data...")
    train_exs, temp_exs = train_test_split(examples, test_size=VAL_FRACTION * 2, random_state=SEED)
    val_exs, test_exs = train_test_split(temp_exs, test_size=0.5, random_state=SEED)
    print(f"  Train: {len(train_exs)}, Val: {len(val_exs)}, Test: {len(test_exs)}")

    # Build datasets
    print("\n[3/6] Building tf.data datasets...")
    edge_weights = _compute_edge_weights(train_exs, strength=0.5)

    def make_dataset(exs, shuffle, weights=None, augment=False):
        paths = [ex.image_path for ex in exs]
        values = [ex.value for ex in exs]
        obb = [ex.obb_params for ex in exs]
        crops = [ex.crop_box_xyxy for ex in exs]
        w_arr = weights.astype(np.float32) if weights is not None else np.ones(len(exs), dtype=np.float32)

        ds = tf.data.Dataset.from_tensor_slices((
            tf.constant(paths), tf.constant(values, dtype=tf.float32),
            tf.constant(np.array(obb, dtype=np.float32)),
            tf.constant(np.array(crops, dtype=np.float32)),
            tf.constant(w_arr),
        ))
        if shuffle:
            ds = ds.shuffle(len(exs), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda p, v, o, c, w: _load_fullframe_obb_data_colour(p, v, o, c, IMAGE_HEIGHT, IMAGE_WIDTH, w), num_parallel_calls=2)
        if augment:
            ds = ds.map(_augment_train, num_parallel_calls=2)
        ds = ds.batch(BATCH_SIZE, drop_remainder=shuffle)
        ds = ds.prefetch(2)
        return ds

    train_ds = make_dataset(train_exs, shuffle=True, weights=edge_weights, augment=True)
    val_ds = make_dataset(val_exs, shuffle=False)
    test_ds = make_dataset(test_exs, shuffle=False)

    # Build model
    print("\n[4/6] Building PP-LCNet model...")
    model = build_pp_lcnet_obb_center_model(
        image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH,
        scale=1.0, head_units_1=HEAD_UNITS_1, head_units_2=HEAD_UNITS_2,
        head_dropout_1=HEAD_DROPOUT_1, head_dropout_2=HEAD_DROPOUT_2,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={"obb_params": HuberLoss(delta=0.05), "center_xy": HuberLoss(delta=0.02)},
        loss_weights={"obb_params": 1.0, "center_xy": CENTER_LOSS_WEIGHT},
        metrics={"obb_params": [keras.metrics.MeanAbsoluteError(name="mae")], "center_xy": [keras.metrics.MeanAbsoluteError(name="mae")]},
    )

    model.summary(print_fn=lambda s: print(f"  {s}"))

    # Train
    print(f"\n[5/6] Training ({EPOCHS} epochs)...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
        keras.callbacks.ModelCheckpoint(str(run_dir / "best_model.keras"), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

    # Evaluate
    print("\n[6/6] Evaluating on test set...")
    test_results = model.evaluate(test_ds, verbose=0)
    metric_names = model.metrics_names
    for name, val in zip(metric_names, test_results):
        print(f"  {name}: {val:.6f}")

    test_preds = model.predict(test_ds, verbose=0)
    test_obb_preds = test_preds["obb_params"]
    test_center_preds = test_preds["center_xy"]
    test_obb_true = np.array([ex.obb_params for ex in test_exs])
    test_center_true = test_obb_true[:, :2]

    obb_param_names = ["cx", "cy", "w", "h", "cos2t", "sin2t"]
    per_param_mae = np.mean(np.abs(test_obb_preds - test_obb_true), axis=0)
    print("\n  OBB per-parameter MAE (normalised):")
    for name, mae in zip(obb_param_names, per_param_mae):
        print(f"    {name}: {mae:.6f}")

    cx_err = np.mean(np.abs(test_center_preds[:, 0] - test_center_true[:, 0])) * IMAGE_WIDTH
    cy_err = np.mean(np.abs(test_center_preds[:, 1] - test_center_true[:, 1])) * IMAGE_HEIGHT
    euclidean = np.mean(np.sqrt(
        np.square((test_center_preds[:, 0] - test_center_true[:, 0]) * IMAGE_WIDTH)
        + np.square((test_center_preds[:, 1] - test_center_true[:, 1]) * IMAGE_HEIGHT)
    ))
    print(f"\n  Center error @320: cx={cx_err:.1f}px, cy={cy_err:.1f}px, euclidean={euclidean:.1f}px")

    model.save(str(run_dir / "final_model.keras"))
    print(f"\n  Model saved to: {run_dir / 'final_model.keras'}")

    # Export TFLite
    print("\n=== Exporting TFLite int8 ===")

    def representative_dataset():
        for ex in test_exs[:50]:
            if ex.image_path.endswith(".yuv422"):
                continue
            img = tf.io.read_file(ex.image_path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = _preprocess_colour(img, IMAGE_HEIGHT, IMAGE_WIDTH)
            yield [tf.expand_dims(img, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path = run_dir / "obb_center_pp_lcnet_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  TFLite int8: {tflite_path}")
    print(f"  Size: {size_kb:.1f} KB ({size_kb / 1024:.2f} MB)")

    # Save summary
    num_epochs = len(history.history.get("loss", []))
    val_losses = history.history.get("val_loss", [0])
    summary = {
        "run_name": run_name, "model": "PP-LCNet + CoordConv",
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH], "batch_size": BATCH_SIZE,
        "num_train": len(train_exs), "num_val": len(val_exs), "num_test": len(test_exs),
        "epochs_trained": num_epochs, "final_val_loss": float(min(val_losses)),
        "test_loss": float(test_results[0]), "test_obb_mae": float(per_param_mae.mean()),
        "test_center_mae_px": {"cx": float(cx_err), "cy": float(cy_err), "euclidean": float(euclidean)},
        "per_param_mae_obb": {name: float(mae) for name, mae in zip(obb_param_names, per_param_mae)},
        "tflite_size_kb": float(size_kb), "tflite_under_2_5_mb": size_kb < 2560,
    }
    summary_path = run_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {summary_path}")

    print("\n=== Done ===")
    if size_kb >= 2560:
        print(f"WARNING: TFLite size {size_kb:.0f} KB exceeds 2.5 MB target!")
    else:
        print(f"TFLite size {size_kb:.0f} KB is under the 2.5 MB target. ✓")


if __name__ == "__main__":
    main()
