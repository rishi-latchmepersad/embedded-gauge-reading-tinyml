"""Training data pipeline for SimCC v4 model.

Uses the curated labelled_captured_images.json manifest (420 images,
394 with center+tip keypoint annotations). Each example includes:
  - Firmware crop box (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
  - Center and tip in source coordinates
  - true_angle_degrees (measured, not inverse-mapped)

The pipeline:
  1. Loads the image and applies the firmware crop
  2. Resizes with pad to 224x224
  3. Maps center and tip to canvas coordinates
  4. Generates Gaussian soft targets for SimCC heads
  5. Generates sigmoid target for center_xy head
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[4]
ML_ROOT = REPO_ROOT / "ml"

INPUT_SIZE = 224
NUM_BINS = 112
SIGMA_BINS = 1.75
SUB_BIN_WIDTH = INPUT_SIZE / NUM_BINS

MANIFEST_PATH = ML_ROOT / "data" / "labelled_captured_images.json"


@dataclass
class SimCCv4Example:
    """One training example for the SimCC v4 model."""
    image_path: str
    # Center and tip in source coordinates.
    center_x: float
    center_y: float
    tip_x: float
    tip_y: float
    # Source image dimensions.
    src_w: int
    src_h: int
    # Firmware crop box in source coordinates.
    crop_x1: float
    crop_y1: float
    crop_x2: float
    crop_y2: float
    # Source kind for quality tracking.
    source_kind: str = "pxl_geometry"


def coord_to_simcc_target(coord_pixels: float) -> np.ndarray:
    """Convert a coordinate in pixels to a 112-bin Gaussian SimCC target."""
    bins = np.arange(NUM_BINS, dtype=np.float32)
    center_bin = coord_pixels / SUB_BIN_WIDTH
    target = np.exp(-((bins - center_bin) ** 2) / (2.0 * SIGMA_BINS ** 2))
    total = target.sum()
    if total > 0:
        target /= total
    return target.astype(np.float32)


def load_examples(manifest_path: Path = MANIFEST_PATH) -> list[SimCCv4Example]:
    """Load training examples from the curated manifest.

    Only includes images with both center and tip keypoint annotations.
    Uses crop_x_min/crop_x_max for reviewed_geometry, and loose_crop
    fields for pxl_geometry.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    examples = []
    for img in manifest["images"]:
        for ann in img["annotations"]:
            row = ann["source_row"]
            # Need center, tip, source dimensions.
            if not all(k in row and row[k] is not None for k in [
                "center_x_source", "center_y_source",
                "tip_x_source", "tip_y_source",
                "source_width", "source_height",
            ]):
                continue
            # Determine crop box — use firmware crop if available,
            # else use loose crop (PXL), else use full image.
            if all(k in row and row[k] is not None for k in [
                "crop_x_min", "crop_y_min", "crop_x_max", "crop_y_max",
            ]):
                crop_x1, crop_y1 = float(row["crop_x_min"]), float(row["crop_y_min"])
                crop_x2, crop_y2 = float(row["crop_x_max"]), float(row["crop_y_max"])
            elif all(k in row and row[k] is not None for k in [
                "loose_crop_x1", "loose_crop_y1",
                "loose_crop_x2", "loose_crop_y2",
            ]):
                crop_x1, crop_y1 = float(row["loose_crop_x1"]), float(row["loose_crop_y1"])
                crop_x2, crop_y2 = float(row["loose_crop_x2"]), float(row["loose_crop_y2"])
            else:
                crop_x1, crop_y1 = 0.0, 0.0
                crop_x2 = float(row["source_width"])
                crop_y2 = float(row["source_height"])
            examples.append(SimCCv4Example(
                image_path=str(REPO_ROOT / img["image_path"]),
                center_x=float(row["center_x_source"]),
                center_y=float(row["center_y_source"]),
                tip_x=float(row["tip_x_source"]),
                tip_y=float(row["tip_y_source"]),
                src_w=int(row["source_width"]),
                src_h=int(row["source_height"]),
                crop_x1=crop_x1,
                crop_y1=crop_y1,
                crop_x2=crop_x2,
                crop_y2=crop_y2,
                source_kind=ann.get("source_kind", "unknown"),
            ))
    return examples


def crop_and_resize(image: np.ndarray, x1: float, y1: float,
                     x2: float, y2: float, canvas: int = INPUT_SIZE):
    """Crop to (x1,y1,x2,y2) in source coords, resize_with_pad to canvas.

    Returns (canvas_image, scale, pad_x, pad_y).
    """
    h, w = image.shape[:2]
    x1c, y1c = max(0.0, x1), max(0.0, y1)
    x2c = min(float(w), max(x1c + 1.0, x2))
    y2c = min(float(h), max(y1c + 1.0, y2))
    ix1, iy1 = int(math.floor(x1c)), int(math.floor(y1c))
    ix2, iy2 = int(math.ceil(x2c)), int(math.ceil(y2c))
    crop_w, crop_h = max(1, ix2 - ix1), max(1, iy2 - iy1)
    cropped = image[iy1:iy1 + crop_h, ix1:ix1 + crop_w]
    scale = min(canvas / crop_w, canvas / crop_h)
    sh, sw = max(1, int(round(crop_h * scale))), max(1, int(round(crop_w * scale)))
    resized = cv2.resize(cropped, (sw, sh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (canvas - sw) // 2, (canvas - sh) // 2
    canvas_img = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    canvas_img[pad_y:pad_y + sh, pad_x:pad_x + sw] = resized
    return canvas_img, scale, pad_x, pad_y


def preprocess_example(ex: SimCCv4Example, augmentation: bool = True):
    """Load, crop, resize, and generate SimCC targets."""
    image = cv2.imread(ex.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read {ex.image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    canvas, scale, pad_x, pad_y = crop_and_resize(
        image, ex.crop_x1, ex.crop_y1, ex.crop_x2, ex.crop_y2,
        canvas=INPUT_SIZE,
    )

    # Map center and tip from source coords to canvas coords.
    def source_to_canvas(sx, sy):
        return (sx - ex.crop_x1) * scale + pad_x, \
               (sy - ex.crop_y1) * scale + pad_y

    center_cx, center_cy = source_to_canvas(ex.center_x, ex.center_y)
    tip_cx, tip_cy = source_to_canvas(ex.tip_x, ex.tip_y)

    # Clamp to canvas bounds.
    center_cx = max(0.0, min(float(INPUT_SIZE - 1), center_cx))
    center_cy = max(0.0, min(float(INPUT_SIZE - 1), center_cy))
    tip_cx = max(0.0, min(float(INPUT_SIZE - 1), tip_cx))
    tip_cy = max(0.0, min(float(INPUT_SIZE - 1), tip_cy))

    # Augmentation: brightness/contrast only.
    if augmentation:
        alpha = np.random.uniform(0.85, 1.15)
        beta = np.random.uniform(-15, 15)
        canvas = np.clip(
            alpha * canvas.astype(np.float32) + beta, 0, 255
        ).astype(np.uint8)

    targets = {
        # Center detector: (cx, cy) in [0, 1].
        "center_xy": np.array(
            [center_cx / INPUT_SIZE, center_cy / INPUT_SIZE],
            dtype=np.float32,
        ),
        # SimCC: 4 heads with Gaussian soft targets.
        "center_x": coord_to_simcc_target(center_cx),
        "center_y": coord_to_simcc_target(center_cy),
        "tip_x": coord_to_simcc_target(tip_cx),
        "tip_y": coord_to_simcc_target(tip_cy),
    }
    return canvas, targets


def build_tf_dataset(examples, batch_size=16, shuffle=True, augment=True):
    """Build a tf.data.Dataset from a list of SimCCv4Example."""

    def _gen():
        indices = list(range(len(examples)))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            try:
                img, targets = preprocess_example(examples[i], augmentation=augment)
                yield img, targets
            except Exception:
                continue

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=[INPUT_SIZE, INPUT_SIZE, 3], dtype=tf.uint8),
            {
                "center_xy": tf.TensorSpec(shape=[2], dtype=tf.float32),
                "center_x": tf.TensorSpec(shape=[NUM_BINS], dtype=tf.float32),
                "center_y": tf.TensorSpec(shape=[NUM_BINS], dtype=tf.float32),
                "tip_x": tf.TensorSpec(shape=[NUM_BINS], dtype=tf.float32),
                "tip_y": tf.TensorSpec(shape=[NUM_BINS], dtype=tf.float32),
            },
        ),
    )
    ds = ds.map(
        lambda img, t: (tf.cast(img, tf.float32) / 255.0, t),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat()  # Make it infinite so Keras can stop by epoch count.
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def dataset_steps_per_epoch(examples, batch_size: int) -> int:
    """Number of batches per epoch (after drop_remainder)."""
    return max(1, len(examples) // batch_size)


if __name__ == "__main__":
    examples = load_examples()
    print(f"Total examples: {len(examples)}")
    source_kinds = {}
    for e in examples:
        source_kinds[e.source_kind] = source_kinds.get(e.source_kind, 0) + 1
    print(f"Source distribution: {source_kinds}")

    if examples:
        img, targets = preprocess_example(examples[0], augmentation=False)
        print(f"\nFirst example ({examples[0].source_kind}):")
        print(f"  Image: {img.shape}")
        print(f"  center_xy: {targets['center_xy']}")
        for k in ["center_x", "center_y", "tip_x", "tip_y"]:
            probs = targets[k]
            bin_idx = int(probs.argmax())
            coord = (bin_idx + 0.5) * SUB_BIN_WIDTH
            print(f"  {k}: peak_bin={bin_idx}, decoded_coord={coord:.1f}px")
