"""QARepVGG-Mini heatmap-based OBB + centre localizer with QAT.

Architecture:
  Stem(32) → QAStage(64)×2 → QAStage(96)×2 → QAStage(128)×2 → QAStage(160)×1
    → 2× upsample → 1×1 proj(64) → [heatmap(1), box_size(2), angle(2)] at 32×32

Output:
  heatmap (32×32)  — 2D Gaussian centre peak (logits → sigmoid)
  box_size (32×32) — w, h  (per-cell regression, valid only at peak)
  angle (32×32)    — sin(2θ), cos(2θ) via tanh

Target: 256×256 input, 8× downsampling = 32×32 heatmap grid.

Usage:
  cd ml && setsid poetry run python scripts/train_qat_qarepvgg_mini.py > /tmp/qarepvgg.log 2>&1 & disown
"""

from __future__ import annotations

import os as _os
_GPU_MEMORY_LIMIT_MB = int(_os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "3900"))

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
from embedded_gauge_reading_tinyml.tf_models import build_qarepvgg_mini
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
BATCH_SIZE: int = 16
EPOCHS_WARMUP: int = 80
EPOCHS_QAT: int = 20
LEARNING_RATE: float = 1e-3
QAT_LEARNING_RATE: float = 5e-5
VAL_FRACTION: float = 0.20
CENTER_LOSS_WEIGHT: float = 3.0

POSITIONAL_AUG_CY_MIN: float = 0.10
POSITIONAL_AUG_CY_MAX: float = 0.90
POSITIONAL_AUG_CX_MIN: float = 0.10
POSITIONAL_AUG_CX_MAX: float = 0.90
ROTATION_AUG_DEG: float = 15.0


# ── Target generation ────────────────────────────────────────────────────────

def _make_heatmap_target(cx: float, cy: float, h: int, w: int,
                          sigma: float = SIGMA_PIXELS) -> np.ndarray:
    """Generate a 2D Gaussian heatmap at (cx, cy) normalised coords."""
    return make_gaussian_heatmap(h, w, cx, cy, sigma_pixels=sigma)


def _make_grid_targets(cx: float, cy: float, w: float, h: float,
                        cos2t: float, sin2t: float,
                        grid_h: int, grid_w: int
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Create box_size and angle maps with values only at the peak cell."""
    box_map = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    angle_map = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    # Find peak cell
    cell_x = int(round(cx * (grid_w - 1)))
    cell_y = int(round(cy * (grid_h - 1)))
    cell_x = max(0, min(cell_x, grid_w - 1))
    cell_y = max(0, min(cell_y, grid_h - 1))
    box_map[cell_y, cell_x, :] = [w, h]
    angle_map[cell_y, cell_x, :] = [sin2t, cos2t]
    return box_map, angle_map


# ── Preprocessing ────────────────────────────────────────────────────────────

def _preprocess_colour(image: tf.Tensor, height: int, width: int) -> tf.Tensor:
    """Resize-with-pad to target size."""
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


# ── Data loading (produces map targets) ──────────────────────────────────────

def _load_sample(
    image_path: tf.Tensor, obb_params: tf.Tensor, weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load image and produce heatmap + grid-target dicts."""
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

    obb = tf.cast(obb_params, tf.float32)  # [cx, cy, w, h, cos2t, sin2t]
    cx, cy, w, h, cos2t, sin2t = obb[0], obb[1], obb[2], obb[3], obb[4], obb[5]

    # Generate heatmap via py_function (np is simpler for Gaussian)
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
    heatmap = heatmap[..., tf.newaxis]  # [32, 32, 1]

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


def _augment_positional(
    image: tf.Tensor, heatmap: tf.Tensor, obb: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Random translate to make position-invariant."""
    cy_shift = tf.random.uniform([], POSITIONAL_AUG_CY_MIN, POSITIONAL_AUG_CY_MAX)
    cx_shift = tf.random.uniform([], POSITIONAL_AUG_CX_MIN, POSITIONAL_AUG_CX_MAX)
    # We skip image-space translation — positional augmentation via
    # random centre shifts is handled by the heatmap generation.
    # For simplicity, leave as identity.
    return image, heatmap, obb


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
    """Focal loss for 2D centre heatmap."""
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
    """Smooth-L1 (Huber) applied only where heatmap peak exists."""
    def __init__(self, delta: float = 1.0 / 9.0,
                 reduction: str = "sum_over_batch_size", name: str = "masked_huber"):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        diff = tf.abs(y_true - y_pred)
        quad = 0.5 * tf.square(diff)
        lin = self.delta * (diff - 0.5 * self.delta)
        huber = tf.where(diff <= self.delta, quad, lin)
        # Mask: only penalise positions where target > 0
        mask = tf.cast(tf.reduce_max(tf.abs(y_true), axis=-1, keepdims=True) > 0, tf.float32)
        masked = huber * mask
        N = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(masked) / N

    def get_config(self) -> dict:
        return {"delta": self.delta, "name": self.name}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"qat_qarepvgg_mini_{timestamp}"
    run_dir = ARTIFACTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QARepVGG-Mini — Heatmap OBB + Centre")
    print(f"  {IMAGE_HEIGHT}x{IMAGE_WIDTH} → {HEATMAP_SIZE}x{HEATMAP_SIZE} heatmap")
    print(f"  FP32 warmup: {EPOCHS_WARMUP} epochs, QAT: {EPOCHS_QAT} epochs")
    print(f"  Run: {run_dir}")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading examples...")
    examples = _build_all_examples()
    print(f"  Loaded {len(examples)} examples")

    # ── 2. Split ─────────────────────────────────────────────────────────
    print("\n[2/6] Splitting...")
    train_exs, temp_exs = train_test_split(
        examples, test_size=VAL_FRACTION * 2, random_state=SEED,
    )
    val_exs, test_exs = train_test_split(temp_exs, test_size=0.5, random_state=SEED)
    print(f"  Train: {len(train_exs)}  Val: {len(val_exs)}  Test: {len(test_exs)}")

    # ── 3. Build datasets ────────────────────────────────────────────────
    print("\n[3/6] Building tf.data datasets...")
    edge_weights = _compute_edge_weights(train_exs, strength=0.5)

    def make_dataset(exs, shuffle, weights=None):
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
            lambda p, o, w: _load_sample(p, o, w),
            num_parallel_calls=4,
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=shuffle)
        ds = ds.prefetch(2)
        return ds

    train_ds = make_dataset(train_exs, shuffle=True, weights=edge_weights)
    val_ds = make_dataset(val_exs, shuffle=False)
    test_ds = make_dataset(test_exs, shuffle=False)

    # ── 4. Build model ──────────────────────────────────────────────────
    print("\n[4/6] Building QARepVGG-Mini...")
    model = build_qarepvgg_mini((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=LEARNING_RATE,
                decay_steps=EPOCHS_WARMUP * max(1, len(train_exs) // BATCH_SIZE),
                alpha=1e-4,
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
        weighted_metrics=[],
    )
    total_params = model.count_params()
    est_mb = total_params / 1e6
    print(f"  Total params: {total_params:,} (~{est_mb:.1f}M → ~{est_mb:.1f} MB int8)")

    # ── 5a. FP32 warmup ──────────────────────────────────────────────────
    print(f"\n[5a/5] FP32 warmup ({EPOCHS_WARMUP} epochs)...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=15,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
        keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_warmup.keras"),
            monitor="val_loss", mode="min", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1,
        ),
    ]
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_WARMUP, callbacks=callbacks, verbose=2,
    )
    model.load_weights(str(run_dir / "best_warmup.keras"))

    # ── 5b. QAT ─────────────────────────────────────────────────────────
    print(f"\n[5b/5] QAT fine-tune ({EPOCHS_QAT} epochs)...")
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=QAT_LEARNING_RATE, clipnorm=1.0),
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
        weighted_metrics=[],
    )
    qat_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10,
            restore_best_weights=True, verbose=1,
        ),
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

    # ── 6. Evaluate ─────────────────────────────────────────────────────
    print("\n[6/6] Evaluating on test set...")
    test_results = qat_model.evaluate(test_ds, verbose=0)
    metric_names = qat_model.metrics_names
    for name, val in zip(metric_names, test_results):
        print(f"  {name}: {val:.6f}")

    # Decode centre from heatmap
    test_preds = qat_model.predict(test_ds, verbose=0)
    test_heatmaps = test_preds["heatmap"]  # (N, 32, 32, 1)
    test_boxes = test_preds["box_size"]
    test_angles = test_preds["angle"]

    # Soft-argmax centre from heatmap (handle all-negative logits)
    import warnings
    pred_centers_list = []
    for i in range(len(test_heatmaps)):
        hm = 1.0 / (1.0 + np.exp(-test_heatmaps[i, :, :, 0]))
        s = np.sum(hm)
        if s > 0:
            hm_n = hm / s
            ys, xs = np.meshgrid(np.arange(HEATMAP_SIZE, dtype=np.float32),
                                  np.arange(HEATMAP_SIZE, dtype=np.float32), indexing="ij")
            py = float(np.sum(hm_n * ys))
            px = float(np.sum(hm_n * xs))
        else:
            py, px = 0.0, 0.0
            warnings.warn(f"Test sample {i}: zero heatmap")
        pred_centers_list.append((px, py))
    pred_centers = np.array(pred_centers_list)  # (N, 2) — (x, y) as (col, row)

    test_obb_true = np.array([ex.obb_params for ex in test_exs])
    true_centers = test_obb_true[:, :2]  # (N, 2) — (cx, cy)

    cx_err = np.mean(np.abs(pred_centers[:, 0] - true_centers[:, 0])) * IMAGE_WIDTH
    cy_err = np.mean(np.abs(pred_centers[:, 1] - true_centers[:, 1])) * IMAGE_HEIGHT
    euclidean = np.mean(np.sqrt(
        np.square((pred_centers[:, 0] - true_centers[:, 0]) * IMAGE_WIDTH)
        + np.square((pred_centers[:, 1] - true_centers[:, 1]) * IMAGE_HEIGHT)
    ))
    print(f"\n  Centre error @{IMAGE_WIDTH}: cx={cx_err:.1f}px, cy={cy_err:.1f}px, "
          f"euclidean={euclidean:.1f}px")

    # ── Export TFLite int8 ──────────────────────────────────────────────
    print("\n=== Exporting TFLite int8 ===")

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
    tflite_path = run_dir / "qarepvgg_mini_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  TFLite int8: {tflite_path}")
    print(f"  Size: {size_kb:.1f} KB ({size_kb / 1024:.2f} MB)")

    # ── Save summary ────────────────────────────────────────────────────
    summary = {
        "run_name": run_name,
        "model": "qarepvgg_mini",
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "heatmap_size": HEATMAP_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_warmup": EPOCHS_WARMUP,
        "epochs_qat": EPOCHS_QAT,
        "num_train": len(train_exs),
        "num_val": len(val_exs),
        "num_test": len(test_exs),
        "total_params": int(total_params),
        "test_center_mae_px": {
            "cx": float(cx_err), "cy": float(cy_err), "euclidean": float(euclidean),
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
