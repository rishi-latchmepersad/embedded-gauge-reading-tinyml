"""Training data pipeline for the SimCC v2 model.

Takes the OBB-cropped gauge region (224x224) and generates SimCC targets
for center and tip keypoints.

Data flow:
  1. Load the source image and crop to the gauge bounding box.
  2. Resize with pad to 224x224.
  3. Map center and tip coordinates from source to canvas.
  4. Convert to SimCC soft targets (112-bin distributions).
  5. Apply augmentation (brightness, contrast, no rotation).

Why no rotation?
  SimCC is for the OBB-cropped region, which is already axis-aligned.
  Rotation would break the SimCC coordinate convention.
"""

from __future__ import annotations

import csv
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
SUB_BIN_WIDTH = INPUT_SIZE / NUM_BINS  # 2.0 pixels per bin.

MANIFEST_PATH = ML_ROOT / "data" / "geometry_heatmap_v12_all_data_manifest.csv"
BC_PATH = ML_ROOT / "data" / "board_captures_labeled.csv"


@dataclass
class SimCCExample:
    """One training example for the SimCC model."""
    image_path: str
    # Gauge center and tip in source coordinates.
    center_x: float
    center_y: float
    tip_x: float
    tip_y: float
    # Source image dimensions.
    src_w: int
    src_h: int
    # Loose crop box in source coordinates.
    crop_x1: float
    crop_y1: float
    crop_x2: float
    crop_y2: float
    quality: str = "clean"


def coord_to_simcc_target(coord_pixels: float) -> np.ndarray:
    """Convert a coordinate in pixels to a 112-bin SimCC soft target."""
    bin_float = coord_pixels / SUB_BIN_WIDTH
    left = int(math.floor(bin_float))
    right = left + 1
    weight_right = bin_float - left
    weight_left = 1.0 - weight_right

    left = max(0, min(NUM_BINS - 1, left))
    right = max(0, min(NUM_BINS - 1, right))

    target = np.zeros(NUM_BINS, dtype=np.float32)
    target[left] += weight_left
    target[right] += weight_right
    total = target.sum()
    if total > 0:
        target /= total
    return target


def load_pxl_examples() -> list[SimCCExample]:
    """Load PXL phone photo examples."""
    examples = []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qf = r.get("label_quality", "")
            if qf not in ("clean", "manual"):
                continue
            examples.append(SimCCExample(
                image_path=str(REPO_ROOT / r["image_path"]),
                center_x=float(r["center_x_source"]),
                center_y=float(r["center_y_source"]),
                tip_x=float(r["tip_x_source"]),
                tip_y=float(r["tip_y_source"]),
                src_w=int(r["source_width"]),
                src_h=int(r["source_height"]),
                crop_x1=float(r["loose_crop_x1"]),
                crop_y1=float(r["loose_crop_y1"]),
                crop_x2=float(r["loose_crop_x2"]),
                crop_y2=float(r["loose_crop_y2"]),
                quality=qf,
            ))
    return examples


def load_board_examples() -> list[SimCCExample]:
    """Load board capture examples."""
    examples = []
    if not BC_PATH.exists():
        return examples
    with open(BC_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            examples.append(SimCCExample(
                image_path=str(REPO_ROOT / r["image_path"]),
                center_x=float(r["center_x"]),
                center_y=float(r["center_y"]),
                tip_x=float(r["tip_x"]),
                tip_y=float(r["tip_y"]),
                src_w=int(r["source_width"]),
                src_h=int(r["source_height"]),
                crop_x1=0.0, crop_y1=0.0,
                crop_x2=float(r["source_width"]),
                crop_y2=float(r["source_height"]),
                quality="manual",
            ))
    return examples


def crop_and_resize(image: np.ndarray, x1: float, y1: float,
                     x2: float, y2: float, canvas: int = INPUT_SIZE):
    """Crop to (x1,y1,x2,y2) in source coords, resize_with_pad to canvas.

    Returns (canvas_image, scale, pad_x, pad_y, src_w, src_h).
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


def preprocess_example(
    ex: SimCCExample, augmentation: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load, crop, resize, and generate SimCC targets.

    Returns:
        (image_224x224x3, targets dict with center_x/y, tip_x/y, confidence)
    """
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
        cx = (sx - ex.crop_x1) * scale + pad_x
        cy = (sy - ex.crop_y1) * scale + pad_y
        return cx, cy

    center_cx, center_cy = source_to_canvas(ex.center_x, ex.center_y)
    tip_cx, tip_cy = source_to_canvas(ex.tip_x, ex.tip_y)

    # Clamp to canvas bounds.
    center_cx = max(0.0, min(float(INPUT_SIZE - 1), center_cx))
    center_cy = max(0.0, min(float(INPUT_SIZE - 1), center_cy))
    tip_cx = max(0.0, min(float(INPUT_SIZE - 1), tip_cx))
    tip_cy = max(0.0, min(float(INPUT_SIZE - 1), tip_cy))

    # Augmentation: brightness/contrast only (no rotation).
    if augmentation:
        alpha = np.random.uniform(0.85, 1.15)
        beta = np.random.uniform(-15, 15)
        canvas = np.clip(
            alpha * canvas.astype(np.float32) + beta, 0, 255
        ).astype(np.uint8)

    targets = {
        "center_x": coord_to_simcc_target(center_cx),
        "center_y": coord_to_simcc_target(center_cy),
        "tip_x": coord_to_simcc_target(tip_cx),
        "tip_y": coord_to_simcc_target(tip_cy),
        "confidence": np.array([1.0], dtype=np.float32),
    }

    return canvas, targets


def build_tf_dataset(
    examples: list[SimCCExample],
    batch_size: int = 16,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from a list of SimCCExample."""

    def _gen() -> Iterator[tuple[np.ndarray, dict]]:
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
                "center_x": tf.TensorSpec(shape=[NUM_BINS], dtype=tf.float32),
                "center_y": tf.TensorSpec(shape=[NUM_BINS], dtype=tf.float32),
                "tip_x": tf.TensorSpec(shape=[NUM_BINS], dtype=tf.float32),
                "tip_y": tf.TensorSpec(shape=[NUM_BINS], dtype=tf.float32),
                "confidence": tf.TensorSpec(shape=[1], dtype=tf.float32),
            },
        ),
    )
    ds = ds.map(
        lambda img, t: (tf.cast(img, tf.float32) / 255.0, t),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    pxl = load_pxl_examples()
    bc = load_board_examples()
    print(f"PXL examples: {len(pxl)}")
    print(f"Board examples: {len(bc)}")
    print(f"Total: {len(pxl) + len(bc)}")

    if pxl:
        img, targets = preprocess_example(pxl[0], augmentation=False)
        print(f"\nFirst PXL example:")
        print(f"  Image: {img.shape}")
        for k, v in targets.items():
            if k != "confidence":
                # Decode the SimCC target to verify.
                probs = v
                bin_idx = np.argmax(probs)
                coord = (bin_idx + 0.5) * SUB_BIN_WIDTH
                print(f"  {k}: peak_bin={bin_idx}, decoded_coord={coord:.1f}px")
            else:
                print(f"  {k}: {v}")

    if bc:
        img, targets = preprocess_example(bc[0], augmentation=False)
        print(f"\nFirst board example:")
        print(f"  Image: {img.shape}")
        for k, v in targets.items():
            if k != "confidence":
                probs = v
                bin_idx = np.argmax(probs)
                coord = (bin_idx + 0.5) * SUB_BIN_WIDTH
                print(f"  {k}: peak_bin={bin_idx}, decoded_coord={coord:.1f}px")
            else:
                print(f"  {k}: {v}")
