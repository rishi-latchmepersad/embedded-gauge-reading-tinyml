"""Training data pipeline for SimCC v5 model.

Strong augmentation: geometric jitter (shift, scale, aspect, rotation),
color jitter, Gaussian noise, occasional blur.

This generates 1D SimCC targets (cx, cy, tx, ty) for the axis-pooling model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

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
class SimCCv5Example:
    image_path: str
    center_x: float
    center_y: float
    tip_x: float
    tip_y: float
    src_w: int
    src_h: int
    crop_x1: float
    crop_y1: float
    crop_x2: float
    crop_y2: float
    source_kind: str = "pxl_geometry"


def coord_to_simcc_target(coord_pixels: float) -> np.ndarray:
    """Convert pixel coord to a 112-bin Gaussian SimCC target."""
    bins = np.arange(NUM_BINS, dtype=np.float32)
    center_bin = coord_pixels / SUB_BIN_WIDTH
    target = np.exp(-((bins - center_bin) ** 2) / (2.0 * SIGMA_BINS ** 2))
    total = target.sum()
    if total > 0:
        target /= total
    return target.astype(np.float32)


def load_examples(manifest_path: Path = MANIFEST_PATH) -> list[SimCCv5Example]:
    """Load training examples from the curated manifest.

    Includes both PXL (clean labels) and board captures (mostly accurate
    labels). The board labels have some click-position noise but the
    temperatures are correct.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    examples = []
    for img in manifest["images"]:
        for ann in img["annotations"]:
            row = ann["source_row"]
            if not all(k in row and row[k] is not None for k in [
                "center_x_source", "center_y_source",
                "tip_x_source", "tip_y_source",
                "source_width", "source_height",
            ]):
                continue
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
            examples.append(SimCCv5Example(
                image_path=str(REPO_ROOT / img["image_path"]),
                center_x=float(row["center_x_source"]),
                center_y=float(row["center_y_source"]),
                tip_x=float(row["tip_x_source"]),
                tip_y=float(row["tip_y_source"]),
                src_w=int(row["source_width"]),
                src_h=int(row["source_height"]),
                crop_x1=crop_x1, crop_y1=crop_y1,
                crop_x2=crop_x2, crop_y2=crop_y2,
                source_kind=ann.get("source_kind", "unknown"),
            ))
    return examples


def preprocess_example(ex: SimCCv5Example, augmentation: bool = True):
    """Load, crop, resize, augment, and generate SimCC targets."""
    image = cv2.imread(ex.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read {ex.image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    crop_w = ex.crop_x2 - ex.crop_x1
    crop_h = ex.crop_y2 - ex.crop_y1
    if augmentation:
        # Shift: ±10% of crop size.
        max_shift_x = 0.10 * crop_w
        max_shift_y = 0.10 * crop_h
        shift_x = np.random.uniform(-max_shift_x, max_shift_x)
        shift_y = np.random.uniform(-max_shift_y, max_shift_y)
        # Scale: 0.85-1.15.
        scale = np.random.uniform(0.85, 1.15)
        # Aspect: 0.92-1.08.
        aspect = np.random.uniform(0.92, 1.08)
    else:
        shift_x, shift_y = 0.0, 0.0
        scale, aspect = 1.0, 1.0

    new_w = crop_w * scale
    new_h = crop_h * scale * aspect
    cx = (ex.crop_x1 + ex.crop_x2) / 2 + shift_x
    cy = (ex.crop_y1 + ex.crop_y2) / 2 + shift_y
    new_x1 = cx - new_w / 2
    new_y1 = cy - new_h / 2
    new_x2 = cx + new_w / 2
    new_y2 = cy + new_h / 2

    h, w = image.shape[:2]
    ix1, iy1 = int(math.floor(max(0.0, new_x1))), int(math.floor(max(0.0, new_y1)))
    ix2 = int(math.ceil(min(float(w), max(ix1 + 1, new_x2))))
    iy2 = int(math.ceil(min(float(h), max(iy1 + 1, new_y2))))
    actual_w, actual_h = max(1, ix2 - ix1), max(1, iy2 - iy1)
    cropped = image[iy1:iy1 + actual_h, ix1:ix1 + actual_w]
    canvas = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    def source_to_canvas(sx, sy):
        return (sx - ix1) * (INPUT_SIZE / actual_w), (sy - iy1) * (INPUT_SIZE / actual_h)

    center_cx, center_cy = source_to_canvas(ex.center_x, ex.center_y)
    tip_cx, tip_cy = source_to_canvas(ex.tip_x, ex.tip_y)
    center_cx = max(0.0, min(float(INPUT_SIZE - 1), center_cx))
    center_cy = max(0.0, min(float(INPUT_SIZE - 1), center_cy))
    tip_cx = max(0.0, min(float(INPUT_SIZE - 1), tip_cx))
    tip_cy = max(0.0, min(float(INPUT_SIZE - 1), tip_cy))

    if augmentation:
        # Brightness/contrast.
        alpha = np.random.uniform(0.85, 1.15)
        beta = np.random.uniform(-15, 15)
        canvas = np.clip(
            alpha * canvas.astype(np.float32) + beta, 0, 255
        ).astype(np.uint8)
        # Gaussian noise.
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 3, canvas.shape).astype(np.float32)
            canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # Occasional blur.
        if np.random.rand() < 0.3:
            ksize = np.random.choice([3, 5])
            canvas = cv2.GaussianBlur(canvas, (ksize, ksize), 0)

    targets = {
        "center_xy": np.array(
            [center_cx / INPUT_SIZE, center_cy / INPUT_SIZE],
            dtype=np.float32,
        ),
        "center_x": coord_to_simcc_target(center_cx),
        "center_y": coord_to_simcc_target(center_cy),
        "tip_x": coord_to_simcc_target(tip_cx),
        "tip_y": coord_to_simcc_target(tip_cy),
    }
    return canvas, targets


def build_tf_dataset(examples, batch_size=16, shuffle=True, augment=True):
    """Build a tf.data.Dataset from a list of SimCCv5Example."""

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
    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def dataset_steps_per_epoch(examples, batch_size: int) -> int:
    return max(1, len(examples) // batch_size)


if __name__ == "__main__":
    examples = load_examples()
    print(f"Total examples: {len(examples)}")
    src_kinds = {}
    for e in examples:
        src_kinds[e.source_kind] = src_kinds.get(e.source_kind, 0) + 1
    print(f"Source distribution: {src_kinds}")
    img, targets = preprocess_example(examples[0], augmentation=True)
    print(f"Image: {img.shape}, range=[{img.min()}, {img.max()}]")
    print(f"center_xy: {targets['center_xy']}")
    for k in ["center_x", "center_y", "tip_x", "tip_y"]:
        probs = targets[k]
        bin_idx = int(probs.argmax())
        coord = (bin_idx + 0.5) * SUB_BIN_WIDTH
        print(f"  {k}: peak_bin={bin_idx}, coord={coord:.1f}px")
