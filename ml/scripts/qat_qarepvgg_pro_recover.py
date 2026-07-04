"""QARepVGG-Pro QAT recovery — picks up from warmup checkpoint.
Loads best_warmup_multi.keras → reparameterize → QAT → TFLite int8.

Usage:
  cd ml && poetry run python scripts/qat_qarepvgg_pro_recover.py
"""

from __future__ import annotations

import os as _os
_GPU_MEMORY_LIMIT_MB = int(_os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "3900"))

import json
import sys
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

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.tf_models import reparameterize_model

# ── Find latest warmup checkpoint ────────────────────────────────────────────

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts" / "training"
RUN_DIR: Path = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml/ml/artifacts/training/qat_qarepvgg_pro_a1.5_20260614_081150")

WARMUP_MODEL: Path = RUN_DIR / "best_warmup_multi.keras"
if not WARMUP_MODEL.exists():
    # Fallback: find most recent
    dirs = sorted(ARTIFACTS_DIR.glob("qat_qarepvgg_pro_a*"), reverse=True)
    for d in dirs:
        candidate = d / "best_warmup_multi.keras"
        if candidate.exists():
            RUN_DIR = d
            WARMUP_MODEL = candidate
            break
    else:
        print("ERROR: No warmup checkpoint found!")
        sys.exit(1)

print(f"Loading warmup model: {WARMUP_MODEL}")
model_multi = keras.models.load_model(WARMUP_MODEL, compile=False)
print(f"  Multi-branch params: {model_multi.count_params():,}")

# ── Load data for QAT ────────────────────────────────────────────────────────
# Minimal: load just enough data structures to run QAT
import csv
from sklearn.model_selection import train_test_split
from embedded_gauge_reading_tinyml.training import TrainingExample, _compute_edge_weights, _compute_fullframe_obb_params

ML_ROOT = PROJECT_ROOT
MANIFEST_PATH = ML_ROOT / "data" / "merged_geometry_board_manifest.csv"
AI_CENTERS_CSV = ML_ROOT / "data" / "ai_annotated_board_captures.csv"
ANNOTATE_30_CSV = ML_ROOT / "data" / "annotate_30" / "annotations.csv"
ANNOTATE_BATCH2_CSV = ML_ROOT / "data" / "annotate_batch2" / "annotations_batch2.csv"
CAPTURED_IMAGES_DIR = ML_ROOT / "data" / "captured_images_320"
YUV_LABELS_CSV = ML_ROOT / "data" / "capture_2026-06-07_labels_v2.csv"
GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO = 0.3076

IMAGE_HEIGHT, IMAGE_WIDTH = 320, 320
HEATMAP_SIZE = 40
SIGMA_PIXELS = 1.5
BATCH_SIZE = 8
EPOCHS_QAT = 30
QAT_LEARNING_RATE = 5e-5
CENTER_LOSS_WEIGHT = 3.0
SEED = 42
VAL_FRACTION = 0.20

def _obb_params_from_center_224(cx_224, cy_224, source_size=224):
    radius = source_size * GAUGE_INNER_DIAL_RADIUS_FRAME_RATIO
    return _compute_fullframe_obb_params(
        source_size, source_size, cx_224, cy_224, radius, radius, 0.0, IMAGE_HEIGHT, IMAGE_WIDTH)

def _is_original_capture(filename):
    name = Path(filename).name
    if any(tag in name for tag in (".gray.", "_preview", "_yuy2", "_glare_")):
        return False
    return True

# Duplicate data loaders from training script
def _load_manifest_examples(examples, seen):
    added = 0
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "captured_images" not in row["image_path"]: continue
            if row.get("quality_flag", "") not in ("clean", "manual"): continue
            fpath = str(ML_ROOT.parent / row["image_path"])
            fpath = fpath.replace("/captured_images/", "/captured_images_320/")
            if fpath.endswith(".jpg"): fpath = fpath[:-4] + ".png"
            if fpath in seen or not Path(fpath).exists(): continue
            seen.add(fpath)
            source_w, source_h = int(float(row["source_width"])), int(float(row["source_height"]))
            cx, cy = float(row["center_x_source"]), float(row["center_y_source"])
            radius = float(row.get("dial_radius_source", 0))
            if radius <= 0: continue
            obb = _compute_fullframe_obb_params(source_w, source_h, cx, cy, radius, radius, 0.0, IMAGE_HEIGHT, IMAGE_WIDTH)
            examples.append(TrainingExample(
                image_path=fpath, value=float(row.get("temperature_c", 0)),
                crop_box_xyxy=(0.0, 0.0, float(source_w), float(source_h)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb))
            added += 1
    return added

def _load_pxl_photo_examples(examples, seen):
    added = 0
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "PXL_" not in row["image_path"]: continue
            if row.get("quality_flag", "") not in ("clean", "manual"): continue
            fpath = str(ML_ROOT.parent / row["image_path"])
            if fpath in seen or not Path(fpath).exists(): continue
            seen.add(fpath)
            source_w, source_h = int(float(row["source_width"])), int(float(row["source_height"]))
            cx, cy = float(row["center_x_source"]), float(row["center_y_source"])
            radius = float(row.get("dial_radius_source", 0))
            if radius <= 0: continue
            obb = _compute_fullframe_obb_params(source_w, source_h, cx, cy, radius, radius, 0.0, IMAGE_HEIGHT, IMAGE_WIDTH)
            examples.append(TrainingExample(
                image_path=fpath, value=float(row.get("temperature_c", 0)),
                crop_box_xyxy=(0.0, 0.0, float(source_w), float(source_h)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb))
            added += 1
    return added

def _load_annotate_csv(csv_path, examples, seen, *, source_size=224):
    added = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            if not _is_original_capture(fname): continue
            fpath = str(CAPTURED_IMAGES_DIR / fname)
            if fpath in seen or not Path(fpath).exists(): continue
            seen.add(fpath)
            cx, cy = float(row["cx"]), float(row["cy"])
            obb = _obb_params_from_center_224(cx, cy, source_size)
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, float(source_size), float(source_size)),
                needle_unit_xy=(0.0, 0.0), obb_params=obb))
            added += 1
    return added

def _load_ai_annotated_examples(examples, seen):
    added = 0
    if not AI_CENTERS_CSV.exists(): return 0
    with open(AI_CENTERS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel_path = row["image_path"]
            if not _is_original_capture(rel_path): continue
            rel_path = rel_path.replace("captured_images/", "captured_images_320/")
            fpath = str(ML_ROOT / "data" / rel_path)
            if fpath in seen or not Path(fpath).exists(): continue
            seen.add(fpath)
            cx, cy = float(row["center_x"]), float(row["center_y"])
            obb = _obb_params_from_center_224(cx, cy, 224)
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, 224.0, 224.0),
                needle_unit_xy=(0.0, 0.0), obb_params=obb))
            added += 1
    return added

def _load_yuv_board_captures(examples, seen):
    added = 0
    if not YUV_LABELS_CSV.exists(): return 0
    with open(YUV_LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fpath = str(CAPTURED_IMAGES_DIR / row["image_path"])
            if fpath in seen or not Path(fpath).exists(): continue
            seen.add(fpath)
            cx, cy = float(row["center_x"]), float(row["center_y"])
            outer_r = float(row["outer_radius"])
            obb = _compute_fullframe_obb_params(320, 320, cx, cy, outer_r, outer_r, 0.0, IMAGE_HEIGHT, IMAGE_WIDTH)
            examples.append(TrainingExample(
                image_path=fpath, value=0.0,
                crop_box_xyxy=(0.0, 0.0, 320.0, 320.0),
                needle_unit_xy=(0.0, 0.0), obb_params=obb))
            added += 1
    return added

print("Loading examples...")
examples = []
seen = set()
n1 = _load_pxl_photo_examples(examples, seen)
n2 = _load_manifest_examples(examples, seen)
n3 = _load_yuv_board_captures(examples, seen)
n4 = _load_annotate_csv(ANNOTATE_30_CSV, examples, seen)
n5 = _load_annotate_csv(ANNOTATE_BATCH2_CSV, examples, seen)
n6 = _load_ai_annotated_examples(examples, seen)
print(f"  PXL:{n1} manifest:{n2} YUV:{n3} ann30:{n4} annb2:{n5} AI:{n6} = {len(examples)} total")

train_exs, temp_exs = train_test_split(examples, test_size=VAL_FRACTION * 2, random_state=SEED)
val_exs, test_exs = train_test_split(temp_exs, test_size=0.5, random_state=SEED)
print(f"  Train:{len(train_exs)} Val:{len(val_exs)} Test:{len(test_exs)}")

# Data pipeline (same as training script)
from embedded_gauge_reading_tinyml.heatmap_utils import make_gaussian_heatmap

def _make_heatmap_target(cx, cy, h, w, sigma=SIGMA_PIXELS):
    return make_gaussian_heatmap(h, w, cx, cy, sigma_pixels=sigma)

def _make_grid_targets(cx, cy, w, h, cos2t, sin2t, grid_h, grid_w):
    box_map = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    angle_map = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    cell_x = int(round(cx * (grid_w - 1)))
    cell_y = int(round(cy * (grid_h - 1)))
    cell_x = max(0, min(cell_x, grid_w - 1))
    cell_y = max(0, min(cell_y, grid_h - 1))
    box_map[cell_y, cell_x, :] = [w, h]
    angle_map[cell_y, cell_x, :] = [sin2t, cos2t]
    return box_map, angle_map

def _preprocess_colour(image, height, width):
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

def _load_sample(image_path, obb_params, weight):
    is_yuv = tf.strings.regex_full_match(image_path, ".*\\.yuv422$")
    def _load_yuv():
        raw = tf.io.read_file(image_path)
        yuyv = tf.io.decode_raw(raw, tf.uint8)
        yuyv = tf.reshape(yuyv, [320, 640])
        y = tf.cast(yuyv[:, 0::2], tf.float32)
        u = tf.cast(yuyv[:, 1::4], tf.float32) - 128.0
        v = tf.cast(yuyv[:, 3::4], tf.float32) - 128.0
        u = tf.repeat(u, 2, axis=1); v = tf.repeat(v, 2, axis=1)
        rgb = tf.stack([y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u], axis=-1)
        return tf.cast(tf.clip_by_value(rgb, 0, 255), tf.uint8)
    def _load_standard():
        return tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
    image = tf.cond(is_yuv, _load_yuv, _load_standard)
    image = tf.ensure_shape(image, [None, None, 3])
    image = _preprocess_colour(image, IMAGE_HEIGHT, IMAGE_WIDTH)
    obb = tf.cast(obb_params, tf.float32)
    cx, cy, w, h, cos2t, sin2t = obb[0], obb[1], obb[2], obb[3], obb[4], obb[5]
    def _gen_hm(cx_n, cy_n):
        return _make_heatmap_target(float(cx_n), float(cy_n), HEATMAP_SIZE, HEATMAP_SIZE).astype(np.float32)
    def _gen_grid(cx_n, cy_n, wn, hn, c2t, s2t):
        bm, am = _make_grid_targets(float(cx_n), float(cy_n), float(wn), float(hn), float(c2t), float(s2t), HEATMAP_SIZE, HEATMAP_SIZE)
        return bm, am
    heatmap = tf.numpy_function(_gen_hm, [cx, cy], tf.float32)
    heatmap = tf.ensure_shape(heatmap, [HEATMAP_SIZE, HEATMAP_SIZE])
    heatmap = heatmap[..., tf.newaxis]
    box, angle = tf.numpy_function(_gen_grid, [cx, cy, w, h, cos2t, sin2t], [tf.float32, tf.float32])
    box = tf.ensure_shape(box, [HEATMAP_SIZE, HEATMAP_SIZE, 2])
    angle = tf.ensure_shape(angle, [HEATMAP_SIZE, HEATMAP_SIZE, 2])
    targets = {"heatmap": heatmap, "box_size": box, "angle": angle}
    return image, targets, None

edge_weights = _compute_edge_weights(train_exs, strength=0.5)
def make_dataset(exs, shuffle, weights=None):
    paths = [ex.image_path for ex in exs]
    obb = [ex.obb_params for ex in exs]
    w_arr = weights.astype(np.float32) if weights is not None else np.ones(len(exs), dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(np.array(obb, dtype=np.float32)), tf.constant(w_arr)))
    if shuffle:
        ds = ds.shuffle(len(exs), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, o, w: _load_sample(p, o, w), num_parallel_calls=4)
    ds = ds.batch(BATCH_SIZE, drop_remainder=shuffle)
    return ds.prefetch(2)

train_ds = make_dataset(train_exs, shuffle=True, weights=edge_weights)
val_ds = make_dataset(val_exs, shuffle=False)

# ── Reparameterize ──────────────────────────────────────────────────────────

print("\n[1/3] Reparameterizing → fused single-conv model (no SE for tfmot compat)...")
model_fused = reparameterize_model(model_multi, use_se=False)  # SE blocks not supported by tfmot
params_fused = model_fused.count_params()
print(f"  Fused params: {params_fused:,} (Δ={model_multi.count_params()-params_fused:,})")
print(f"  Est. INT8: {params_fused/1024/1024:.2f}M → ~{params_fused*1.2/1024/1024:.1f} MB TFLite")

# ── QAT ──────────────────────────────────────────────────────────────────────

print("\n[2/3] QAT on fused model...")
try:
    model_fused._is_graph_network = True
except AttributeError:
    pass

qat_model = tfmot.quantization.keras.quantize_model(model_fused)

# Loss classes (same as training script)
class FocalHeatmapLoss(keras.losses.Loss):
    def __init__(self, alpha=2.0, gamma=4.0, reduction="sum_over_batch_size", name="focal_heat"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha; self.gamma = gamma
    def call(self, y_true, y_pred):
        pred = tf.sigmoid(y_pred); true = tf.cast(y_true, tf.float32)
        pos_mask = true > 0.5
        pos_loss = -tf.where(pos_mask, (1-pred)**self.alpha * tf.math.log(tf.clip_by_value(pred, 1e-7, 1.0)), tf.zeros_like(pred))
        neg_weight = (1-true)**self.gamma
        neg_loss = -tf.where(true < 0.5, neg_weight * pred**self.alpha * tf.math.log(tf.clip_by_value(1-pred, 1e-7, 1.0)), tf.zeros_like(pred))
        return tf.reduce_mean(pos_loss + neg_loss)
    def get_config(self): return {"alpha": self.alpha, "gamma": self.gamma, "name": self.name}

class MaskedHuberLoss(keras.losses.Loss):
    def __init__(self, delta=1.0/9.0, reduction="sum_over_batch_size", name="masked_huber"):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta
    def call(self, y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        quad = 0.5 * tf.square(diff); lin = self.delta * (diff - 0.5 * self.delta)
        huber = tf.where(diff <= self.delta, quad, lin)
        mask = tf.cast(tf.reduce_max(tf.abs(y_true), axis=-1, keepdims=True) > 0, tf.float32)
        masked = huber * mask
        return tf.reduce_sum(masked) / tf.maximum(tf.reduce_sum(mask), 1.0)
    def get_config(self): return {"delta": self.delta, "name": self.name}

qat_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=QAT_LEARNING_RATE,
            decay_steps=EPOCHS_QAT * max(1, len(train_exs) // BATCH_SIZE),
            alpha=1e-4,
        ),
        clipnorm=1.0,
    ),
    loss={"heatmap": FocalHeatmapLoss(alpha=2.0, gamma=4.0),
          "box_size": MaskedHuberLoss(delta=1.0/9.0),
          "angle": MaskedHuberLoss(delta=1.0/9.0)},
    loss_weights={"heatmap": 1.0, "box_size": 1.0, "angle": CENTER_LOSS_WEIGHT},
)

qat_callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.CSVLogger(str(RUN_DIR / "qat_log.csv")),
    keras.callbacks.ModelCheckpoint(str(RUN_DIR / "best_qat.keras"), monitor="val_loss", mode="min", save_best_only=True, verbose=1),
]

qat_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_QAT, callbacks=qat_callbacks, verbose=2)

# ── TFLite export ───────────────────────────────────────────────────────────

print("\n[3/3] Exporting TFLite int8...")
def representative_dataset():
    for ex in test_exs[:50]:
        if ex.image_path.endswith(".yuv422"): continue
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
tflite_path = RUN_DIR / "qarepvgg_pro_a1.5_int8.tflite"
tflite_path.write_bytes(tflite_model)
size_kb = len(tflite_model) / 1024
print(f"  TFLite int8: {tflite_path}")
print(f"  Size: {size_kb:.1f} KB ({size_kb/1024:.2f} MB)")

# ── Summary ──────────────────────────────────────────────────────────────────
summary = {
    "run_dir": str(RUN_DIR),
    "model": "qarepvgg_pro_a1.5",
    "params_fused": int(params_fused),
    "tflite_size_kb": float(size_kb),
    "tflite_under_2_5_mb": size_kb < 2560,
}
with open(RUN_DIR / "qat_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'✓' if size_kb < 2560 else '✗ WARNING:'} TFLite {size_kb/1024:.2f} MB under 2.5 MB")
print("=== Done ===")
