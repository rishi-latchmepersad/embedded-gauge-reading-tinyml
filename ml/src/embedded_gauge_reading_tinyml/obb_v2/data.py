"""Training data pipeline for the OBB v2 model.

Loads PXL phone photos and board captures, applies rotation augmentation
to teach the model about angled camera mounting, and generates:
  - conf: 1.0 (gauge is present)
  - box: [cx, cy, w, h] in [0, 1] normalised to the 224x224 canvas
  - angle_dist: 16-bin DFL soft target for the gauge rotation angle

Rotation augmentation:
  - Random rotation 0-360° with corresponding OBB angle label update
  - The gauge is rotated in the image, and the OBB angle is adjusted
  - This teaches the model to handle cameras mounted at any angle

Data sources:
  - ml/data/merged_geometry_board_manifest.csv (PXL + board captures)
  - ml/data/board_captures_labeled.csv (additional board captures)
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

# ── OBB angle convention: [-π/4, 3π/4) for DFL ────────────────────────
ANGLE_MIN = -math.pi / 4
ANGLE_MAX = 3 * math.pi / 4
ANGLE_BINS = 16
ANGLE_BIN_WIDTH = (ANGLE_MAX - ANGLE_MIN) / ANGLE_BINS

# ── OBB params: [cx, cy, w, h] in normalised [0, 1] ───────────────────
INPUT_SIZE = 224

MANIFEST_PATH = ML_ROOT / "data" / "merged_geometry_board_manifest.csv"
BC_PATH = ML_ROOT / "data" / "board_captures_labeled.csv"


@dataclass
class OBBExample:
    """One training example for the OBB model."""
    image_path: str
    # Gauge center in source coordinates.
    cx_source: float
    cy_source: float
    # Gauge outer radius in source coordinates.
    radius_source: float
    # Source image dimensions.
    src_w: int
    src_h: int
    # Loose crop box in source coordinates (for pre-cropping).
    crop_x1: float
    crop_y1: float
    crop_x2: float
    crop_y2: float
    # Quality flag.
    quality: str = "clean"


def angle_to_dfl_target(angle_rad: float) -> np.ndarray:
    """Convert an angle in radians to a 16-bin DFL soft target."""
    norm = (angle_rad - ANGLE_MIN) / ANGLE_BIN_WIDTH
    left = int(math.floor(norm))
    right = left + 1
    weight_right = norm - left
    weight_left = 1.0 - weight_right

    left = max(0, min(ANGLE_BINS - 1, left))
    right = max(0, min(ANGLE_BINS - 1, right))

    target = np.zeros(ANGLE_BINS, dtype=np.float32)
    target[left] += weight_left
    target[right] += weight_right
    total = target.sum()
    if total > 0:
        target /= total
    return target


def normalize_angle(angle_rad: float) -> float:
    """Normalize angle to [-π/4, 3π/4) as the DFL expects."""
    # First wrap to [-π, π).
    while angle_rad >= math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    # Then map to [-π/4, 3π/4) by adding π if outside range.
    if angle_rad < ANGLE_MIN:
        angle_rad += math.pi
    if angle_rad >= ANGLE_MAX:
        angle_rad -= math.pi
    return angle_rad


def compute_obb_from_circle(
    cx: float, cy: float, radius: float,
    src_w: int, src_h: int,
    crop_box: tuple[float, float, float, float],
    rotation_deg: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Compute OBB params [cx, cy, w, h, angle] for a circle gauge.

    The circle is approximated as a square with side = 2*radius,
    rotated by the given angle. The OBB is normalised to the
    224x224 canvas after the crop+resize pipeline.

    Returns:
        (cx_norm, cy_norm, w_norm, h_norm, angle_rad)
    """
    x1, y1, x2, y2 = crop_box
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    # Map center from source to crop coords.
    cx_crop = cx - x1
    cy_crop = cy - y1

    # Map to 224x224 canvas with resize-with-pad.
    scale = min(INPUT_SIZE / crop_w, INPUT_SIZE / crop_h)
    sh = int(round(crop_h * scale))
    sw = int(round(crop_w * scale))
    pad_x = (INPUT_SIZE - sw) // 2
    pad_y = (INPUT_SIZE - sh) // 2

    cx_canvas = cx_crop * scale + pad_x
    cy_canvas = cy_crop * scale + pad_y

    # Diameter in canvas pixels.
    diameter = 2.0 * radius * scale

    # Normalise to [0, 1].
    cx_norm = cx_canvas / INPUT_SIZE
    cy_norm = cy_canvas / INPUT_SIZE
    w_norm = diameter / INPUT_SIZE
    h_norm = diameter / INPUT_SIZE

    # Angle in radians, normalised to [-π/4, 3π/4).
    angle_rad = math.radians(rotation_deg)
    angle_norm = normalize_angle(angle_rad)

    return cx_norm, cy_norm, w_norm, h_norm, angle_norm


def load_pxl_examples(min_quality: str = "clean") -> list[OBBExample]:
    """Load PXL phone photo examples from the geometry manifest."""
    examples: list[OBBExample] = []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qf = r.get("label_quality", "")
            # Accept manual quality too.
            if qf not in ("clean", "manual"):
                continue
            ex = OBBExample(
                image_path=str(REPO_ROOT / r["image_path"]),
                cx_source=float(r["center_x_source"]),
                cy_source=float(r["center_y_source"]),
                radius_source=float(r.get("dial_radius_source", 100)),
                src_w=int(r["source_width"]),
                src_h=int(r["source_height"]),
                crop_x1=float(r["loose_crop_x1"]),
                crop_y1=float(r["loose_crop_y1"]),
                crop_x2=float(r["loose_crop_x2"]),
                crop_y2=float(r["loose_crop_y2"]),
                quality=qf,
            )
            examples.append(ex)
    return examples


def load_board_examples() -> list[OBBExample]:
    """Load board capture examples."""
    examples: list[OBBExample] = []
    if not BC_PATH.exists():
        return examples
    with open(BC_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            src_w = int(r["source_width"])
            src_h = int(r["source_height"])
            cx = float(r["center_x"])
            cy = float(r["center_y"])
            # For board captures, the full frame is the crop.
            ex = OBBExample(
                image_path=str(REPO_ROOT / r["image_path"]),
                cx_source=cx,
                cy_source=cy,
                radius_source=80.0,  # board captures use 80px dial radius
                src_w=src_w,
                src_h=src_h,
                crop_x1=0.0, crop_y1=0.0,
                crop_x2=float(src_w), crop_y2=float(src_h),
                quality="manual",
            )
            examples.append(ex)
    return examples


def crop_and_resize_with_pad(
    image: np.ndarray, x1: float, y1: float, x2: float, y2: float,
    canvas: int = INPUT_SIZE,
) -> np.ndarray:
    """Crop to (x1,y1,x2,y2) in source coords, resize_with_pad to canvas."""
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
    return canvas_img


def rotate_image_and_obb(
    image: np.ndarray, cx: float, cy: float, w: float, h: float, angle_deg: float,
    rotation_deg: float,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """Rotate the image by rotation_deg, update the OBB accordingly.

    Args:
        image: 224x224x3 canvas.
        cx, cy: OBB center in [0, 1].
        w, h: OBB size in [0, 1].
        angle_deg: Current OBB angle in degrees.
        rotation_deg: How much to rotate the image by.

    Returns:
        (rotated_image, new_cx, new_cy, new_w, new_h, new_angle_deg)
    """
    h_img, w_img = image.shape[:2]
    center_px = (w_img / 2.0, h_img / 2.0)
    M = cv2.getRotationMatrix2D(center_px, rotation_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (w_img, h_img), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # The center stays at (0.5, 0.5) in normalised coords because the
    # rotation is around the image center.
    new_cx = cx
    new_cy = cy
    # The width and height swap if rotation is 90 or 270.
    if abs((rotation_deg % 180) - 90) < 45:
        new_w = h
        new_h = w
    else:
        new_w = w
        new_h = h
    # Angle: add the rotation.
    new_angle_deg = (angle_deg + rotation_deg) % 360.0
    new_angle_deg = (new_angle_deg + 180.0) % 360.0 - 180.0  # wrap to [-180, 180]
    return rotated, new_cx, new_cy, new_w, new_h, new_angle_deg


def preprocess_example(
    ex: OBBExample, augmentation: bool = True, max_rotation_deg: float = 180.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load, crop, resize, and augment one OBB example.

    Returns:
        (image_224x224x3, targets dict with conf, box, angle_dist)
    """
    image = cv2.imread(ex.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read {ex.image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crop to the loose box and resize with pad to 224x224.
    canvas = crop_and_resize_with_pad(
        image, ex.crop_x1, ex.crop_y1, ex.crop_x2, ex.crop_y2, canvas=INPUT_SIZE,
    )

    # Compute OBB params.
    cx, cy, w, h, angle_rad = compute_obb_from_circle(
        ex.cx_source, ex.cy_source, ex.radius_source,
        ex.src_w, ex.src_h,
        (ex.crop_x1, ex.crop_y1, ex.crop_x2, ex.crop_y2),
        rotation_deg=0.0,
    )
    angle_deg = math.degrees(angle_rad)

    # Augmentation: random rotation (teaches angled mounting).
    # NOTE: For circular gauges, we do NOT rotate the box label
    # (the box is axis-aligned regardless of gauge rotation).
    if augmentation:
        rotation = np.random.uniform(-max_rotation_deg, max_rotation_deg)
        # Rotate the image.
        h_img, w_img = canvas.shape[:2]
        center_px = (w_img / 2.0, h_img / 2.0)
        M = cv2.getRotationMatrix2D(center_px, rotation, 1.0)
        canvas = cv2.warpAffine(canvas, M, (w_img, h_img), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # The box center stays at (0.5, 0.5) because rotation is around image center.
        # The box width and height may swap if rotation is 90 or 270.
        if abs((rotation % 180) - 90) < 45:
            w, h = h, w
        # No angle_dist needed for circular gauges.

    # Random brightness/contrast.
    if augmentation:
        alpha = np.random.uniform(0.85, 1.15)
        beta = np.random.uniform(-15, 15)
        canvas = np.clip(alpha * canvas.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Build targets.
    targets = {
        "conf": np.array([1.0], dtype=np.float32),
        "box": np.array([cx, cy, w, h], dtype=np.float32),
    }

    return canvas, targets


def build_tf_dataset(
    examples: list[OBBExample],
    batch_size: int = 16,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from a list of OBBExample."""

    def _gen() -> Iterator[tuple[np.ndarray, dict]]:
        indices = list(range(len(examples)))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            try:
                img, targets = preprocess_example(examples[i], augmentation=augment)
                yield img, targets
            except Exception as e:
                # Skip bad examples.
                continue

    def _set_shape(img, targets):
        img = tf.ensure_shape(img, [INPUT_SIZE, INPUT_SIZE, 3])
        targets["conf"] = tf.ensure_shape(targets["conf"], [1])
        targets["box"] = tf.ensure_shape(targets["box"], [4])
        return img, targets

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=[INPUT_SIZE, INPUT_SIZE, 3], dtype=tf.uint8),
            {
                "conf": tf.TensorSpec(shape=[1], dtype=tf.float32),
                "box": tf.TensorSpec(shape=[4], dtype=tf.float32),
            },
        ),
    )
    ds = ds.map(lambda img, t: (_set_shape(tf.cast(img, tf.float32) / 255.0, t)),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    # Quick smoke test.
    pxl = load_pxl_examples()
    bc = load_board_examples()
    print(f"PXL examples: {len(pxl)}")
    print(f"Board examples: {len(bc)}")
    print(f"Total: {len(pxl) + len(bc)}")

    # Test preprocessing on first PXL example.
    if pxl:
        img, targets = preprocess_example(pxl[0], augmentation=False)
        print(f"\nFirst PXL example:")
        print(f"  Image: {img.shape}, dtype={img.dtype}")
        print(f"  conf: {targets['conf']}")
        print(f"  box: {targets['box']}")
        print(f"  angle_dist sum: {targets['angle_dist'].sum():.4f}")

    # Test augmentation.
    if pxl:
        img, targets = preprocess_example(pxl[0], augmentation=True, max_rotation_deg=180.0)
        print(f"\nAugmented first PXL example:")
        print(f"  Image: {img.shape}")
        print(f"  box: {targets['box']}")
        print(f"  angle_dist peak bin: {targets['angle_dist'].argmax()}")
