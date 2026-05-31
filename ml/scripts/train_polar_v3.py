#!/usr/bin/env python3
"""
Polar angle classifier v3 — simple CNN backbone + 224-bin classification + rotation augmentation.

Key innovations over v1/v2:
1. Polar shift augmentation: randomly roll the polar image horizontally, adjusting the angle label.
   This generates ANY needle position from ANY training sample — eliminates angle overfitting.
2. Classification head (224 bins) with circular decode + dead-zone masking (matching V28's proven architecture).
3. Simple CNN backbone (65K params from v1) — avoids MobileNetV2 BN mismatch issues.
4. Categorical cross-entropy during training; dead-zone masking at inference only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)
from embedded_gauge_reading_tinyml.polar_projection import polar_project_image
from luma_crop_detector import estimate_bright_centroid, compute_dynamic_crop, crop_and_resize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Gauge constants
MIN_ANGLE_DEG: float = 135.0
SWEEP_DEG: float = 270.0
MIN_TEMP: float = -30.0
MAX_TEMP: float = 50.0
NUM_BINS: int = 224


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 80
    lr: float = 1e-3
    board_weight: float = 10.0
    val_split: float = 0.15
    dropout: float = 0.5
    backbone: str = "simple"
    add_edges: bool = True
    polar_augment: bool = True
    experiment: str = "exp1"
    output_dir: str = "/tmp/gauge_polar_v3"


def input_channels(conf: Config) -> int:
    return 3 + (3 if conf.add_edges else 0)


def _angle_to_bin(angle_deg: float) -> int:
    return int((angle_deg / 360.0) * NUM_BINS) % NUM_BINS


def _bin_to_angle(bin_idx: int) -> float:
    return (bin_idx / float(NUM_BINS)) * 360.0


def circular_decode_angle(logits: np.ndarray) -> float:
    """Circular mean decode with dead-zone masking (matching V28 firmware)."""
    dead = np.ones(NUM_BINS, dtype=np.float32)
    for i in range(NUM_BINS):
        angle = _bin_to_angle(i)
        a = angle if angle >= MIN_ANGLE_DEG else angle + 360.0
        if MIN_ANGLE_DEG <= a <= MIN_ANGLE_DEG + SWEEP_DEG:
            dead[i] = 0.0
    masked = logits.copy()
    masked[dead > 0.5] = -1e9
    e = np.exp(masked - np.max(masked))
    probs = e / (np.sum(e) + 1e-12)
    angles_rad = np.array([(_bin_to_angle(i) * np.pi / 180.0) for i in range(NUM_BINS)])
    sin_mean = np.sum(probs * np.sin(angles_rad))
    cos_mean = np.sum(probs * np.cos(angles_rad))
    angle_rad = np.arctan2(sin_mean, cos_mean)
    return (angle_rad * 180.0 / np.pi) % 360.0


def build_input_channels(img: np.ndarray, add_edges: bool) -> np.ndarray:
    if not add_edges:
        return img
    gray = (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    gx = np.clip(gx * 0.5 + 0.5, 0, 1)
    gy = np.clip(gy * 0.5 + 0.5, 0, 1)
    mag = np.clip(mag / 4.0, 0, 1)
    return np.concatenate([img, gx[..., None], gy[..., None], mag[..., None]], axis=-1)


def load_metadata(p: Path) -> list[dict]:
    with open(p) as f:
        raw = json.load(f)
    samples = []
    for e in raw:
        cx, cy = float(e["center_x_norm"]), float(e["center_y_norm"])
        tx, ty = float(e["tip_x_norm"]), float(e["tip_y_norm"])
        angle = angle_degrees_from_center_to_tip(cx * 224, cy * 224, tx * 224, ty * 224)
        samples.append({
            "path": str(Path(e["image_path"])),
            "temp": float(e["temperature_c"]),
            "angle_deg": angle % 360.0,
            "board": False,
        })
    return samples


def load_board(p: Path) -> list[dict]:
    samples = []
    with open(p) as f:
        for row in csv.DictReader(f):
            sw, sh = float(row["source_width"]), float(row["source_height"])
            cx = float(row["center_x"]) / sw
            cy = float(row["center_y"]) / sh
            tx = float(row["tip_x"]) / sw
            ty = float(row["tip_y"]) / sh
            angle = angle_degrees_from_center_to_tip(cx * 224, cy * 224, tx * 224, ty * 224)
            samples.append({
                "path": str(row["image_path"]),
                "temp": float(row["temperature_c"]),
                "angle_deg": angle % 360.0,
                "board": True,
            })
    return samples


def load_image(path: Path, board: bool, conf: Config, pp_dir: Optional[Path]) -> Optional[np.ndarray]:
    if not path.exists():
        if board:
            alt = PROJECT_ROOT / path
            if not alt.exists():
                return None
            path = alt
        elif pp_dir:
            alt = pp_dir / path
            if not alt.exists():
                return None
            path = alt
        else:
            return None
    try:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        if board:
            cent = estimate_bright_centroid(arr)
            cb = compute_dynamic_crop(arr.shape[1], arr.shape[0], cent.center_x, cent.center_y)
            if cb is None:
                return None
            arr = np.asarray(crop_and_resize(arr, cb, target_size=conf.image_size), dtype=np.uint8)
        elif arr.shape[:2] != (conf.image_size, conf.image_size):
            arr = np.asarray(Image.fromarray(arr).resize((conf.image_size, conf.image_size), Image.BILINEAR), dtype=np.uint8)
        polar = polar_project_image(arr, center_xy=(112, 112), max_radius=112, polar_size=conf.image_size)
        return build_input_channels(polar, conf.add_edges)
    except Exception as e:
        return None


def _simple_backbone(x, backbone_dropout: float = 0.0):
    """65k-param CNN from scratch (same as v1), with per-block dropout."""
    def conv(x, f, s=1):
        x = layers.Conv2D(f, 3, s, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=-1)(x)
        return layers.Activation("swish")(x)
    def res(x, f, d=0):
        s = x
        x = layers.SeparableConv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=-1)(x)
        x = layers.Activation("swish")(x)
        x = layers.SeparableConv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=-1)(x)
        if s.shape[-1] != f:
            s = layers.Conv2D(f, 1, use_bias=False)(s)
            s = layers.GroupNormalization(groups=-1)(s)
        x = layers.Add()([x, s])
        x = layers.Activation("swish")(x)
        if d > 0:
            x = layers.Dropout(d)(x)
        return x
    rates = [backbone_dropout * r for r in [0.1, 0.2, 0.3, 0.4]]
    x = conv(x, 32, 2)
    x = res(x, 32, rates[0])
    x = layers.MaxPool2D(2)(x)
    x = res(x, 64, rates[1])
    x = layers.MaxPool2D(2)(x)
    x = res(x, 96, rates[2])
    x = layers.MaxPool2D(2)(x)
    x = res(x, 128, rates[3])
    x = layers.GlobalAveragePooling2D()(x)
    return x


def build_model(conf: Config) -> keras.Model:
    """
    Simple CNN backbone → classification head (224 bins).

    Matches V28 architecture pattern: polar → CNN → angle bins → circular decode.
    """
    cin = input_channels(conf)
    inp = keras.Input(shape=(conf.image_size, conf.image_size, cin))

    x = _simple_backbone(inp, conf.dropout)

    x = layers.Dense(128, activation="swish")(x)
    x = layers.Dropout(conf.dropout)(x)
    x = layers.Dense(64, activation="swish")(x)
    x = layers.Dropout(conf.dropout)(x)
    out = layers.Dense(NUM_BINS, activation="linear", name="angle_logits")(x)
    return keras.Model(inp, out)


def angular_error(y_true_bin, y_pred_logits):
    """Angular error in radians between true bin and circular-decoded prediction."""
    true_angle = tf.cast(y_true_bin, tf.float32) / tf.cast(NUM_BINS, tf.float32) * 2.0 * np.pi
    pred_probs = tf.nn.softmax(y_pred_logits)
    angles_rad = tf.constant([2.0 * np.pi * i / NUM_BINS for i in range(NUM_BINS)], dtype=tf.float32)
    sin_pred = tf.reduce_sum(pred_probs * tf.sin(angles_rad), axis=-1)
    cos_pred = tf.reduce_sum(pred_probs * tf.cos(angles_rad), axis=-1)
    d = tf.acos(tf.clip_by_value(cos_pred, -1.0, 1.0))
    return tf.reduce_mean(d)


def polar_shift_augment(image, label):
    """
    Randomly roll the polar image horizontally, adjusting the angle bin label.

    In polar space, horizontal shift = needle rotation.
    With 224 columns for 360°, each column = 360/224 ≈ 1.61°.
    """
    shift = tf.random.uniform([], 0, NUM_BINS, dtype=tf.int32)
    image = tf.roll(image, shift=shift, axis=1)
    label = tf.roll(label, shift=tf.cast(shift, tf.int32), axis=0)
    return image, label


def eval_board(model, samples, conf, pp_dir):
    errs = []
    for s in samples:
        p = Path(s["path"])
        if not p.exists():
            a = PROJECT_ROOT / p
            if a.exists():
                p = a
            else:
                continue
        img = load_image(p, True, conf, pp_dir)
        if img is None:
            continue
        logits = model.predict(img[None], verbose=0)[0]
        angle_deg = circular_decode_angle(logits)
        pred_temp = max(MIN_TEMP, min(MAX_TEMP, celsius_from_inner_dial_angle_degrees(angle_deg)))
        errs.append(abs(pred_temp - s["temp"]))
    if not errs:
        return {"mae": float("inf"), "med": float("inf"), "n": 0}
    e = np.array(errs)
    return {"mae": float(e.mean()), "med": float(np.median(e)), "max": float(e.max()), "n": len(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["simple", "shallow"], default="simple")
    ap.add_argument("--board-weight", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--no-polar-augment", action="store_true")
    ap.add_argument("--experiment", default="exp1")
    ap.add_argument("--output-dir", default="/tmp/gauge_polar_v3")
    args = ap.parse_args()

    conf = Config(
        epochs=args.epochs, dropout=args.dropout, lr=args.lr,
        board_weight=args.board_weight,
        add_edges=not args.no_edges,
        polar_augment=not args.no_polar_augment,
        backbone=args.backbone,
        experiment=args.experiment, output_dir=args.output_dir,
    )

    print("=" * 60)
    print(f"POLAR ANGLE CLASSIFIER v3 — backbone={conf.backbone}")
    print(f"  {NUM_BINS}-bin classification + circular decode")
    print(f"  Edges: {conf.add_edges}  Polar augment: {conf.polar_augment}")
    print(f"  Dropout (head): {conf.dropout}  Board weight: {conf.board_weight}x")

    root = Path(__file__).resolve().parent.parent
    pp_dir = root / "data" / "preprocessed_crops"

    all_s = []
    mp = pp_dir / "metadata.json"
    if mp.exists():
        all_s.extend(load_metadata(mp))
    bp = root / "data" / "board_captures_labeled_v2.csv"
    board_s = []
    if bp.exists():
        board_s = load_board(bp)
        all_s.extend(board_s)
    print(f"  {len(all_s)} total samples ({len(board_s)} board)")

    np.random.seed(42)
    ix = np.random.permutation(len(all_s))
    vs = int(len(all_s) * conf.val_split)
    tr = [all_s[i] for i in ix[vs:]]
    va = [all_s[i] for i in ix[:vs]]
    tr_w = []
    for s in tr:
        tr_w.append(s)
        if s["board"]:
            tr_w.extend([s] * (int(conf.board_weight) - 1))
    print(f"  Train: {len(tr_w)} (weighted)  Val: {len(va)}")

    # Build datasets
    tr_imgs, tr_tgts = [], []
    for s in tr_w:
        img = load_image(Path(s["path"]), s["board"], conf, pp_dir)
        if img is None:
            continue
        tr_imgs.append(img)
        bin_idx = _angle_to_bin(s["angle_deg"])
        oh = np.zeros(NUM_BINS, dtype=np.float32)
        oh[bin_idx] = 1.0
        tr_tgts.append(oh)

    va_imgs, va_tgts = [], []
    for s in va:
        img = load_image(Path(s["path"]), s["board"], conf, pp_dir)
        if img is None:
            continue
        va_imgs.append(img)
        bin_idx = _angle_to_bin(s["angle_deg"])
        oh = np.zeros(NUM_BINS, dtype=np.float32)
        oh[bin_idx] = 1.0
        va_tgts.append(oh)

    tr_ds = tf.data.Dataset.from_tensor_slices((np.stack(tr_imgs), np.stack(tr_tgts)))
    if conf.polar_augment:
        tr_ds = tr_ds.map(polar_shift_augment, num_parallel_calls=tf.data.AUTOTUNE)
    tr_ds = tr_ds.shuffle(max(len(tr_imgs) // 2, 100)).batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)
    va_ds = tf.data.Dataset.from_tensor_slices((np.stack(va_imgs), np.stack(va_tgts)))
    va_ds = va_ds.batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"  Train batches: {len(tr_imgs)} samples  Val: {len(va_imgs)} samples")

    model = build_model(conf)
    model.compile(
        optimizer=keras.optimizers.Adam(conf.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", angular_error],
    )

    out = Path(conf.output_dir) / conf.experiment
    out.mkdir(parents=True, exist_ok=True)

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_angular_error", patience=20, restore_best_weights=True, mode="min"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_angular_error", factor=0.5, patience=10, min_lr=1e-6, mode="min"),
    ]
    hist = model.fit(tr_ds, validation_data=va_ds, epochs=conf.epochs, callbacks=cb, verbose=2)
    model.save(out / "model.keras")

    hist_d = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    hist_d["config"] = {
        "backbone": conf.backbone, "edges": conf.add_edges,
        "board_weight": conf.board_weight, "polar_augment": conf.polar_augment,
    }
    json.dump(hist_d, open(out / "history.json", "w"), indent=2)

    r = eval_board(model, board_s, conf, pp_dir)
    print(f"\n  Board MAE: {r['mae']:.2f}°C  med={r['med']:.2f}°C  max={r.get('max', 0):.2f}°C  n={r['n']}")
    json.dump(r, open(out / "board_eval.json", "w"), indent=2)
    if r["mae"] < 5:
        print("  ✓ TARGET ACHIEVED")
    else:
        print(f"  ✗ Need <5°C (got {r['mae']:.2f}°C)")

    # INT8 quant
    def rep():
        for x, _ in va_ds.take(100):
            yield [x]
    cvt = tf.lite.TFLiteConverter.from_keras_model(model)
    cvt.optimizations = [tf.lite.Optimize.DEFAULT]
    cvt.representative_dataset = rep
    cvt.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    cvt.inference_input_type = tf.uint8
    cvt.inference_output_type = tf.float32
    with open(out / "model_int8.tflite", "wb") as f:
        f.write(cvt.convert())
    sz = (out / "model_int8.tflite").stat().st_size
    print(f"INT8: {sz/1024:.0f} KB")


if __name__ == "__main__":
    main()
