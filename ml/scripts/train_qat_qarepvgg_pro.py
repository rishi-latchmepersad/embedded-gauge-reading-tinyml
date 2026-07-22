"""QARepVGG-Pro heatmap-based OBB + centre localizer with QAT.

Architecture:
  Stem(48) → QAStage(96)×2 → QAStage(144)×2 → QAStage(192)×2 → QAStage(240)×1
    → 2× upsample → 3×3 refine(96) → [heatmap(1), box_size(2), angle(2)] at 40×40

Training flow:
  1. Multi-branch FP32 warmup (3×3 + 1×1 + identity BN per block)
  2. reparameterize_model() → single 3×3 conv per block
  3. tfmot.quantize_model() on fused model
  4. QAT fine-tune
  5. TFLite int8 export

Usage:
  cd ml && setsid poetry run python scripts/train_qat_qarepvgg_pro.py > /tmp/qarepvgg_pro.log 2>&1 & disown
"""

from __future__ import annotations

import os as _os
_GPU_MEMORY_LIMIT_MB = int(_os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "15000"))

import json
import sys
import csv
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

import tf_keras as keras
import tensorflow_model_optimization as tfmot
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
from embedded_gauge_reading_tinyml.tf_models import (
    build_qarepvgg_multi,
    reparameterize_model,
)
from embedded_gauge_reading_tinyml.heatmap_utils import make_gaussian_heatmap


# ── Paths ────────────────────────────────────────────────────────────────────

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
GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO: float = 0.3076


# ── Config ───────────────────────────────────────────────────────────────────

IMAGE_HEIGHT: int = 320
IMAGE_WIDTH: int = 320
HEATMAP_SIZE: int = 40        # 320 / 8
SIGMA_PIXELS: float = 1.5     # Gaussian sigma in heatmap pixels
SEED: int = 42
BATCH_SIZE: int = 6    # α=1.75 model is 2.46M params — batch 6 fits 4GB GPU
EPOCHS_WARMUP: int = 200
EPOCHS_QAT: int = 30
LEARNING_RATE: float = 1e-3
QAT_LEARNING_RATE: float = 1e-5        # conservative — lets quant scales settle smoothly
VAL_FRACTION: float = 0.20
CENTER_LOSS_WEIGHT: float = 3.0
MODEL_ALPHA: float = 1.25

POSITIONAL_AUG_CY_MIN: float = 0.10
POSITIONAL_AUG_CY_MAX: float = 0.90
POSITIONAL_AUG_CX_MIN: float = 0.10
POSITIONAL_AUG_CX_MAX: float = 0.90
ROTATION_AUG_DEG: float = 15.0


# ── Target generation ────────────────────────────────────────────────────────

def _make_heatmap_target(cx: float, cy: float, h: int, w: int,
                          sigma: float = SIGMA_PIXELS) -> np.ndarray:
    return make_gaussian_heatmap(h, w, cx, cy, sigma_pixels=sigma)


def _make_grid_targets(cx: float, cy: float, w: float, h: float,
                        cos2t: float, sin2t: float,
                        grid_h: int, grid_w: int
                        ) -> tuple[np.ndarray, np.ndarray]:
    box_map = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    angle_map = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    cell_x = int(round(cx * (grid_w - 1)))
    cell_y = int(round(cy * (grid_h - 1)))
    cell_x = max(0, min(cell_x, grid_w - 1))
    cell_y = max(0, min(cell_y, grid_h - 1))
    box_map[cell_y, cell_x, :] = [w, h]
    angle_map[cell_y, cell_x, :] = [sin2t, cos2t]
    return box_map, angle_map


# ── Preprocessing ────────────────────────────────────────────────────────────

def _preprocess_colour(image: tf.Tensor, height: int, width: int) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    crop_h = tf.shape(image)[0]
    crop_w = tf.shape(image)[1]
    scale = tf.minimum(
        tf.cast(height, tf.float32) / tf.cast(crop_h, tf.float32),
        tf.cast(width, tf.float32) / tf.cast(crop_w, tf.float32),
    )
    scaled_h = tf.maximum(tf.cast(tf.cast(crop_h, tf.float32) * scale, tf.int32), 1)
    scaled_w = tf.maximum(tf.cast(tf.cast(crop_w, tf.float32) * scale, tf.int32), 1)
    resized = tf.image.resize(image, [scaled_h, scaled_w], method="nearest")
    pad_y = (height - scaled_h) // 2
    pad_x = (width - scaled_w) // 2
    pad_bottom = height - scaled_h - pad_y
    pad_right = width - scaled_w - pad_x
    padded = tf.pad(resized, [[pad_y, pad_bottom], [pad_x, pad_right], [0, 0]])
    padded = tf.ensure_shape(padded, [height, width, 3])
    return padded / 255.0


# ── Data augmentation (training only) ────────────────────────────────────────

def _augment_colour(image: tf.Tensor) -> tf.Tensor:
    """Simulate harsh industrial lighting: hue ±5%, sat 0.8–1.2, bri ±15%, contrast 0.8–1.2."""
    image = tf.image.random_hue(image, 0.05)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return tf.clip_by_value(image, 0.0, 1.0)


def _random_erasing(image: tf.Tensor, max_area: float = 0.15) -> tf.Tensor:
    """Coarse dropout patch simulating gauge smudges, dust, or partial shadows."""
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    h_box = tf.cast(h, tf.float32)
    w_box = tf.cast(w, tf.float32)
    area = tf.random.uniform([], 0.02, max_area) * h_box * w_box
    aspect = tf.random.uniform([], 0.3, 3.3)
    eh = tf.minimum(tf.cast(tf.sqrt(area * aspect), tf.int32), h - 1)
    ew = tf.minimum(tf.cast(tf.sqrt(area / aspect), tf.int32), w - 1)
    y0 = tf.random.uniform([], 0, tf.maximum(h - eh, 1), dtype=tf.int32)
    x0 = tf.random.uniform([], 0, tf.maximum(w - ew, 1), dtype=tf.int32)
    fill_val = tf.random.uniform([], 0.3, 0.7)
    patch = tf.ones([eh, ew, 1], dtype=tf.float32)
    mask = tf.pad(patch, [[y0, h - y0 - eh], [x0, w - x0 - ew], [0, 0]])
    return image * (1.0 - mask) + fill_val * mask


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_sample(
    image_path: tf.Tensor, obb_params: tf.Tensor, weight: tf.Tensor,
    augment: bool = False,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
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
        rgb = tf.stack([
            y + 1.402 * v,
            y - 0.344136 * u - 0.714136 * v,
            y + 1.772 * u,
        ], axis=-1)
        return tf.cast(tf.clip_by_value(rgb, 0, 255), tf.uint8)

    def _load_standard():
        return tf.io.decode_image(
            tf.io.read_file(image_path), channels=3, expand_animations=False,
        )

    image = tf.cond(is_yuv, _load_yuv, _load_standard)
    image = tf.ensure_shape(image, [None, None, 3])
    image = _preprocess_colour(image, IMAGE_HEIGHT, IMAGE_WIDTH)

    # Training-time augmentation — colour jitter + random patch erasing
    if augment:
        image = _augment_colour(image)
        image = _random_erasing(image)

    obb = tf.cast(obb_params, tf.float32)
    cx, cy, w, h, cos2t, sin2t = obb[0], obb[1], obb[2], obb[3], obb[4], obb[5]

    def _gen_hm(cx_n, cy_n):
        hm = _make_heatmap_target(
            float(cx_n), float(cy_n), HEATMAP_SIZE, HEATMAP_SIZE,
        )
        return hm.astype(np.float32)

    def _gen_grid(cx_n, cy_n, wn, hn, c2t, s2t):
        bm, am = _make_grid_targets(
            float(cx_n), float(cy_n), float(wn), float(hn),
            float(c2t), float(s2t), HEATMAP_SIZE, HEATMAP_SIZE,
        )
        return bm, am

    heatmap = tf.numpy_function(_gen_hm, [cx, cy], tf.float32)
    heatmap = tf.ensure_shape(heatmap, [HEATMAP_SIZE, HEATMAP_SIZE])
    heatmap = heatmap[..., tf.newaxis]

    box, angle = tf.numpy_function(
        _gen_grid, [cx, cy, w, h, cos2t, sin2t],
        [tf.float32, tf.float32],
    )
    box = tf.ensure_shape(box, [HEATMAP_SIZE, HEATMAP_SIZE, 2])
    angle = tf.ensure_shape(angle, [HEATMAP_SIZE, HEATMAP_SIZE, 2])

    sw = tf.cast(weight, tf.float32)

    targets = {
        "heatmap": heatmap,
        "box_size": box,
        "angle": angle,
    }
    return image, targets, None


# ── OBB param helper ─────────────────────────────────────────────────────────

def _obb_params_from_center_224(
    cx_224: float, cy_224: float, source_size: int = 224,
) -> np.ndarray:
    radius = source_size * GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO
    return _compute_fullframe_obb_params(
        source_size, source_size, cx_224, cy_224,
        radius, radius, 0.0, IMAGE_HEIGHT, IMAGE_WIDTH,
    )


def _is_original_capture(filename: str) -> bool:
    name = Path(filename).name
    if any(tag in name for tag in (".gray.", "_preview", "_yuy2", "_glare_")):
        return False
    return True


def _load_manifest_examples(examples: list[TrainingExample], seen: set[str]) -> int:
    import csv
    added = 0
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "captured_images" not in row["image_path"]:
                continue
            qf = row.get("quality_flag", "")
            if qf not in ("clean", "manual"):
                continue
            fpath = str(ML_ROOT.parent / row["image_path"])
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
            obb = _compute_fullframe_obb_params(
                source_w, source_h, cx, cy, radius, radius, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(TrainingExample(
                image_path=fpath, value=float(row.get("temperature_c", 0)),
                crop_box_xyxy=(0.0, 0.0, float(source_w), float(source_h)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb,
            ))
            added += 1
    return added


def _load_pxl_photo_examples(examples: list[TrainingExample], seen: set[str]) -> int:
    import csv
    added = 0
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "PXL_" not in row["image_path"]:
                continue
            qf = row.get("quality_flag", "")
            if qf not in ("clean", "manual"):
                continue
            fpath = str(ML_ROOT.parent / row["image_path"])
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
            obb = _compute_fullframe_obb_params(
                source_w, source_h, cx, cy, radius, radius, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(TrainingExample(
                image_path=fpath, value=float(row.get("temperature_c", 0)),
                crop_box_xyxy=(0.0, 0.0, float(source_w), float(source_h)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb,
            ))
            added += 1
    return added


def _load_annotate_csv(
    csv_path: Path, examples: list[TrainingExample], seen: set[str],
    *, source_size: int = 224,
) -> int:
    import csv
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
            obb = _obb_params_from_center_224(cx, cy, source_size)
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, float(source_size), float(source_size)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb,
            ))
            added += 1
    return added


def _load_ai_annotated_examples(examples: list[TrainingExample], seen: set[str]) -> int:
    import csv
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
            obb = _obb_params_from_center_224(cx, cy, 224)
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, 224.0, 224.0),
                needle_unit_xy=(0.0, 0.0), obb_params=obb,
            ))
            added += 1
    return added


def _load_yuv_board_captures(examples: list[TrainingExample], seen: set[str]) -> int:
    import csv
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
            obb = _compute_fullframe_obb_params(
                320, 320, cx, cy, outer_r, outer_r, 0.0,
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, 320.0, 320.0),
                needle_unit_xy=(0.0, 0.0), obb_params=obb,
            ))
            added += 1
    return added


def _build_all_examples() -> list[TrainingExample]:
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


# ── Loss ─────────────────────────────────────────────────────────────────────

class FocalHeatmapLoss(keras.losses.Loss):
    def __init__(self, alpha: float = 2.0, gamma: float = 4.0,
                 reduction: str = "sum_over_batch_size", name: str = "focal_heat"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        pred = tf.sigmoid(y_pred)
        true = tf.cast(y_true, tf.float32)
        pos_mask = true > 0.5
        pos_loss = -tf.where(
            pos_mask,
            (1 - pred) ** self.alpha * tf.math.log(tf.clip_by_value(pred, 1e-7, 1.0)),
            tf.zeros_like(pred),
        )
        neg_weight = (1 - true) ** self.gamma
        neg_loss = -tf.where(
            true < 0.5,
            neg_weight * pred ** self.alpha * tf.math.log(tf.clip_by_value(1 - pred, 1e-7, 1.0)),
            tf.zeros_like(pred),
        )
        return tf.reduce_mean(pos_loss + neg_loss)

    def get_config(self) -> dict:
        return {"alpha": self.alpha, "gamma": self.gamma, "name": self.name}


class MaskedHuberLoss(keras.losses.Loss):
    def __init__(self, delta: float = 1.0 / 9.0,
                 reduction: str = "sum_over_batch_size", name: str = "masked_huber"):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        diff = tf.abs(y_true - y_pred)
        quad = 0.5 * tf.square(diff)
        lin = self.delta * (diff - 0.5 * self.delta)
        huber = tf.where(diff <= self.delta, quad, lin)
        mask = tf.cast(tf.reduce_max(tf.abs(y_true), axis=-1, keepdims=True) > 0, tf.float32)
        masked = huber * mask
        N = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(masked) / N

    def get_config(self) -> dict:
        return {"delta": self.delta, "name": self.name}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"qat_qarepvgg_pro_a{MODEL_ALPHA}_{timestamp}"
    run_dir = ARTIFACTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"QARepVGG-Pro α={MODEL_ALPHA} — Heatmap OBB + Centre")
    print(f"  {IMAGE_HEIGHT}x{IMAGE_WIDTH} → {HEATMAP_SIZE}x{HEATMAP_SIZE} heatmap")
    print(f"  FP32 warmup: {EPOCHS_WARMUP} epochs, QAT: {EPOCHS_QAT} epochs")
    print(f"  Flow: multi-branch → reparameterize → QAT → TFLite int8")
    print(f"  Run: {run_dir}")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/7] Loading examples...")
    examples = _build_all_examples()
    print(f"  Loaded {len(examples)} examples")

    # ── 2. Split ─────────────────────────────────────────────────────────
    print("\n[2/7] Splitting...")
    train_exs, temp_exs = train_test_split(
        examples, test_size=VAL_FRACTION * 2, random_state=SEED,
    )
    val_exs, test_exs = train_test_split(temp_exs, test_size=0.5, random_state=SEED)
    print(f"  Train: {len(train_exs)}  Val: {len(val_exs)}  Test: {len(test_exs)}")

    # ── 3. Build datasets ────────────────────────────────────────────────
    print("\n[3/7] Building tf.data datasets...")
    edge_weights = _compute_edge_weights(train_exs, strength=0.5)

    def make_dataset(exs, shuffle, weights=None, augment=False):
        paths = [ex.image_path for ex in exs]
        obb = [ex.obb_params for ex in exs]
        w_arr = weights.astype(np.float32) if weights is not None else np.ones(len(exs), dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((
            tf.constant(paths), tf.constant(np.array(obb, dtype=np.float32)),
            tf.constant(w_arr),
        ))
        if shuffle:
            ds = ds.shuffle(len(exs), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda p, o, w, aug=tf.constant(augment): _load_sample(p, o, w, augment=aug),
            num_parallel_calls=4,
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=shuffle)
        ds = ds.prefetch(2)
        return ds

    train_ds = make_dataset(train_exs, shuffle=True, weights=edge_weights, augment=True)
    val_ds = make_dataset(val_exs, shuffle=False, augment=False)
    test_ds = make_dataset(test_exs, shuffle=False, augment=False)

    # ── 4a. Build multi-branch model ─────────────────────────────────────
    print("\n[4a/7] Building multi-branch model for FP32 warmup...")
    model_multi = build_qarepvgg_multi(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3), alpha=MODEL_ALPHA, use_se=False,  # no SE: tfmot can't quantise Multiply
    )
    model_multi.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=LEARNING_RATE,
                decay_steps=EPOCHS_WARMUP * max(1, len(train_exs) // BATCH_SIZE),
                alpha=1e-4,
            ),
            weight_decay=1e-5,  # L2 penalty on kernel weights — curbs overfitting at α=1.75
            clipnorm=1.0,
        ),
        loss={
            "heatmap": FocalHeatmapLoss(alpha=2.0, gamma=4.0),
            "box_size": MaskedHuberLoss(delta=1.0/9.0),
            "angle": MaskedHuberLoss(delta=1.0/9.0),
        },
        loss_weights={
            "heatmap": 1.0,
            "box_size": 1.0,
            "angle": CENTER_LOSS_WEIGHT,
        },
        metrics={
            "heatmap": ["mae"],
            "box_size": ["mae"],
            "angle": ["mae"],
        },
    )
    params_multi = model_multi.count_params()
    print(f"  Multi-branch params: {params_multi:,} (~{params_multi*4/1024/1024:.1f}M FP32)")

    # ── 4b. FP32 warmup ──────────────────────────────────────────────────
    print(f"\n[4b/7] FP32 warmup ({EPOCHS_WARMUP} epochs, multi-branch)...")
    warmup_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=15,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
        keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_warmup_multi.keras"),
            monitor="val_loss", mode="min", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1,
        ),
    ]
    model_multi.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_WARMUP, callbacks=warmup_callbacks, verbose=2,
    )
    model_multi.load_weights(str(run_dir / "best_warmup_multi.keras"))

    # ── 4c. Reparameterize ───────────────────────────────────────────────
    print(f"\n[4c/7] Reparameterizing: multi-branch → fused single-conv...")
    model_fused = reparameterize_model(model_multi, use_se=False)
    params_fused = model_fused.count_params()
    print(f"  Fused params: {params_fused:,} (Δ={params_multi-params_fused:,} BNs removed)")
    print(f"  Est. INT8 size: {params_fused/1024/1024:.2f}M → ~{params_fused*1.2/1024/1024:.1f} MB TFLite")
    model_fused.save(run_dir / "fused_model.keras")

    # ── 5a. Build QAT model ─────────────────────────────────────────────
    print(f"\n[5a/7] Applying tfmot.quantize_model() to fused model...")
    # Ensure graph network flag for functional Model compatibility
    try:
        model_fused._is_graph_network = True
    except AttributeError:
        pass

    qat_model = tfmot.quantization.keras.quantize_model(model_fused)

    qat_model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=QAT_LEARNING_RATE,
                decay_steps=EPOCHS_QAT * max(1, len(train_exs) // BATCH_SIZE),
                alpha=1e-3,
            ),
            clipnorm=1.0,
        ),
        loss={
            "heatmap": FocalHeatmapLoss(alpha=2.0, gamma=4.0),
            "box_size": MaskedHuberLoss(delta=1.0/9.0),
            "angle": MaskedHuberLoss(delta=1.0/9.0),
        },
        loss_weights={
            "heatmap": 1.0,
            "box_size": 1.0,
            "angle": CENTER_LOSS_WEIGHT,
        },
        metrics={
            "heatmap": ["mae"],
            "box_size": ["mae"],
            "angle": ["mae"],
        },
    )

    # ── 5b. QAT fine-tune ───────────────────────────────────────────────
    print(f"\n[5b/7] QAT fine-tune ({EPOCHS_QAT} epochs, full schedule — no early stop)...")
    qat_callbacks = [
        keras.callbacks.CSVLogger(str(run_dir / "qat_log.csv")),
        keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_qat.keras"),
            monitor="val_loss", mode="min", save_best_only=True, verbose=1,
        ),
    ]
    qat_model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_QAT, callbacks=qat_callbacks, verbose=2,
    )
    # Restore best-QAT checkpoint after the 30-epoch run
    qat_ckpt = run_dir / "best_qat.keras"
    if qat_ckpt.exists():
        qat_model.load_weights(str(qat_ckpt))

    # ── 6. Export TFLite int8 ────────────────────────────────────────────
    print("\n[6/7] Exporting TFLite int8...")

    def representative_dataset():
        for ex in test_exs[:50]:
            if ex.image_path.endswith(".yuv422"):
                continue
            img = tf.io.read_file(ex.image_path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = _preprocess_colour(img, IMAGE_HEIGHT, IMAGE_WIDTH)
            yield [tf.expand_dims(img, 0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path = run_dir / f"qarepvgg_pro_a{MODEL_ALPHA}_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  TFLite int8: {tflite_path}")
    print(f"  Size: {size_kb:.1f} KB ({size_kb / 1024:.2f} MB)")

    # ── 7. Evaluate (TFLite int8, parabolic sub-pixel) ──────────────────
    print("\n[7/7] Evaluating TFLite int8 with parabolic sub-pixel decoder...")

    def _decode_parabolic(heatmap: np.ndarray) -> tuple[float, float]:
        """2D parabolic Taylor refinement around argmax — ~0.1–0.3 px sub-pixel."""
        h, w = heatmap.shape
        idx = np.argmax(heatmap.flat)
        y0, x0 = idx // w, idx % w
        px = float(x0)
        py = float(y0)
        if 1 <= x0 <= w - 2:
            d = 2.0 * heatmap[y0, x0] - heatmap[y0, x0 - 1] - heatmap[y0, x0 + 1]
            if abs(d) > 1e-8:
                px = x0 + (heatmap[y0, x0 - 1] - heatmap[y0, x0 + 1]) / (2.0 * d)
        if 1 <= y0 <= h - 2:
            d = 2.0 * heatmap[y0, x0] - heatmap[y0 - 1, x0] - heatmap[y0 + 1, x0]
            if abs(d) > 1e-8:
                py = y0 + (heatmap[y0 - 1, x0] - heatmap[y0 + 1, x0]) / (2.0 * d)
        return max(0.0, min(px, w - 1)), max(0.0, min(py, h - 1))

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    in_scale = in_d["quantization_parameters"]["scales"][0]
    in_zp = in_d["quantization_parameters"]["zero_points"][0]

    hm_out = None
    for d in interp.get_output_details():
        if d["shape"][-1] == 1 and len(d["shape"]) == 4:
            hm_out = d
            break
    hm_scale = hm_out["quantization_parameters"]["scales"][0]
    hm_zp = hm_out["quantization_parameters"]["zero_points"][0]

    errors_soft = []
    errors_para = []
    for ex in test_exs:
        if ex.image_path.endswith(".yuv422"):
            continue
        try:
            img = tf.io.decode_image(
                tf.io.read_file(ex.image_path), channels=3, expand_animations=False,
            ).numpy().astype(np.float32)
        except Exception:
            continue
        h, w_img = img.shape[:2]
        sc = min(320 / h, 320 / w_img)
        nh, nw = int(h * sc), int(w_img * sc)
        resized = tf.image.resize(img, [nh, nw], method="nearest").numpy().astype(np.float32)
        pad = np.zeros((320, 320, 3), dtype=np.float32)
        py_pad, px_pad = (320 - nh) // 2, (320 - nw) // 2
        pad[py_pad:py_pad + nh, px_pad:px_pad + nw] = resized
        q_in = (pad / 255.0 / in_scale + in_zp).astype(np.int8)
        interp.set_tensor(in_d["index"], q_in[np.newaxis])
        interp.invoke()
        hm_raw = interp.get_tensor(hm_out["index"])[0, :, :, 0]
        hm_logits = (hm_raw.astype(np.float32) - hm_zp) * hm_scale
        hm = 1.0 / (1.0 + np.exp(-hm_logits))

        # Soft-argmax
        s = hm.sum()
        if s > 0:
            hm_n = hm / s
            ys, xs = np.meshgrid(
                np.arange(HEATMAP_SIZE, dtype=np.float32),
                np.arange(HEATMAP_SIZE, dtype=np.float32), indexing="ij",
            )
            psx = float(np.sum(hm_n * xs))
            psy = float(np.sum(hm_n * ys))
        else:
            psx = psy = 0.0

        # Parabolic
        ppx, ppy = _decode_parabolic(hm)

        cx_t = ex.obb_params[0]
        cy_t = ex.obb_params[1]
        for (px, py), err_list in [((psx, psy), errors_soft), ((ppx, ppy), errors_para)]:
            ex_px = abs(px / 39.0 - cx_t) * 320
            ey_px = abs(py / 39.0 - cy_t) * 320
            err_list.append((ex_px, ey_px))

    for name, errs in [("Soft-argmax", errors_soft), ("Parabolic", errors_para)]:
        e = np.array(errs)
        eucl = np.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2)
        print(f"  {name}:  "
              f"cx={e[:, 0].mean():.1f} px  cy={e[:, 1].mean():.1f} px  "
              f"Eucl mean={eucl.mean():.1f} px  median={np.median(eucl):.1f} px")

    e_para = np.array(errors_para)
    eucl_para = np.sqrt(e_para[:, 0] ** 2 + e_para[:, 1] ** 2)
    cx_err = float(e_para[:, 0].mean())
    cy_err = float(e_para[:, 1].mean())
    euclidean = float(eucl_para.mean())
    eucl_med = float(np.median(eucl_para))

    # ── Save summary ────────────────────────────────────────────────────
    summary = {
        "run_name": run_name,
        "model": f"qarepvgg_pro_a{MODEL_ALPHA}",
        "alpha": MODEL_ALPHA,
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "heatmap_size": HEATMAP_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_warmup": EPOCHS_WARMUP,
        "epochs_qat": EPOCHS_QAT,
        "qat_lr": QAT_LEARNING_RATE,
        "weight_decay": 1e-5,
        "augmentation": True,
        "use_se": False,
        "num_train": len(train_exs),
        "num_val": len(val_exs),
        "num_test": len(test_exs),
        "params_multi": int(params_multi),
        "params_fused": int(params_fused),
        "test_center_mae_px_parabolic": {
            "cx": cx_err, "cy": cy_err, "euclidean": euclidean, "median": eucl_med,
        },
        "tflite_size_kb": float(size_kb),
        "tflite_under_2_5_mb": size_kb < 2560,
    }
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {run_dir / 'training_summary.json'}")
    print("\n=== Done ===")
    if size_kb >= 2560:
        print(f"WARNING: TFLite {size_kb:.0f} KB exceeds 2.5 MB!")
    else:
        print(f"TFLite {size_kb:.0f} KB under 2.5 MB. ✓")


if __name__ == "__main__":
    main()
