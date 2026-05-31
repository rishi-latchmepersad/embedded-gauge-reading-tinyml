#!/usr/bin/env python3
"""
Polar gauge reader v5 — deeper backbone, cosine decay, strong augmentation.

Hypothesis: the v1 simple backbone (65K params) lacks capacity to reach <5°C.
v5 adds a deeper variant with ~150K params and trains with cosine LR schedule
for up to 200 epochs with patience=30 early stopping.

The polar projection is kept because:
1. It converts needle detection to a 1D column-finding problem
2. V28 achieved 0.34°C with this approach
3. The needle signal in polar space, though weak, is linearly separable
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
MIN_TEMP: float = -30.0
MAX_TEMP: float = 50.0


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 200
    lr: float = 1e-3
    board_weight: float = 10.0
    val_split: float = 0.15
    dropout: float = 0.25
    backbone: str = "medium"
    add_edges: bool = True
    experiment: str = "exp1"
    output_dir: str = "/tmp/gauge_polar_v5"


def input_channels(conf: Config) -> int:
    return 3 + (3 if conf.add_edges else 0)


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
        ar = math.radians(angle)
        samples.append({
            "path": str(Path(e["image_path"])),
            "temp": float(e["temperature_c"]),
            "sin": math.sin(ar), "cos": math.cos(ar),
            "angle": angle, "board": False,
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
            ar = math.radians(angle)
            samples.append({
                "path": str(row["image_path"]),
                "temp": float(row["temperature_c"]),
                "sin": math.sin(ar), "cos": math.cos(ar),
                "angle": angle, "board": True,
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


def _medium_backbone(x, dropout: float = 0.0):
    """~150K params — deeper than simple (65K), deeper res blocks."""
    def conv(x, f, s=1):
        x = layers.Conv2D(f, 3, s, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=-1, epsilon=1e-5)(x)
        return layers.Activation("swish")(x)
    def res(x, f, d=0):
        s = x
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=-1, epsilon=1e-5)(x)
        x = layers.Activation("swish")(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.GroupNormalization(groups=-1, epsilon=1e-5)(x)
        if s.shape[-1] != f:
            s = layers.Conv2D(f, 1, use_bias=False)(s)
            s = layers.GroupNormalization(groups=-1, epsilon=1e-5)(s)
        x = layers.Add()([x, s])
        x = layers.Activation("swish")(x)
        if d > 0:
            x = layers.Dropout(d)(x)
        return x
    rates = [dropout * r for r in [0.0, 0.1, 0.15, 0.2, 0.25]]
    x = conv(x, 32, 2)
    x = res(x, 32, rates[0])
    x = layers.MaxPool2D(2)(x)
    x = res(x, 64, rates[1])
    x = layers.MaxPool2D(2)(x)
    x = res(x, 96, rates[2])
    x = layers.MaxPool2D(2)(x)
    x = res(x, 128, rates[3])
    x = layers.MaxPool2D(2)(x)
    x = res(x, 160, rates[4])
    x = layers.GlobalAveragePooling2D()(x)
    return x


def build_model(conf: Config) -> keras.Model:
    cin = input_channels(conf)
    inp = keras.Input(shape=(conf.image_size, conf.image_size, cin))
    x = _medium_backbone(inp, conf.dropout)
    x = layers.Dense(128, activation="swish", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(conf.dropout)(x)
    x = layers.Dense(64, activation="swish", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(conf.dropout)(x)
    out = layers.Dense(2, activation="tanh", name="angle")(x)
    return keras.Model(inp, out)


def sort_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    dot = y_true[:, 0] * y_pred[:, 0] + y_true[:, 1] * y_pred[:, 1]
    return tf.reduce_mean(tf.acos(tf.clip_by_value(dot, -1.0, 1.0)))


def sincos_to_temp(s, c):
    a = math.degrees(math.atan2(s, c)) % 360
    return max(MIN_TEMP, min(MAX_TEMP, celsius_from_inner_dial_angle_degrees(a)))


@tf.function
def augment_polar(image, label):
    rgb = image[..., :3]
    rgb = tf.image.random_brightness(rgb, 0.15)
    rgb = tf.image.random_contrast(rgb, 0.7, 1.3)
    rgb = tf.image.random_saturation(rgb, 0.7, 1.3)
    rgb = tf.image.random_hue(rgb, 0.05)
    image = tf.concat([rgb, image[..., 3:]], axis=-1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    noise = tf.random.normal(tf.shape(image), 0.0, 0.02)
    return tf.clip_by_value(image + noise, 0.0, 1.0), label


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
        o = model.predict(img[None], verbose=0)[0]
        t = sincos_to_temp(float(o[0]), float(o[1]))
        errs.append(abs(t - s["temp"]))
    if not errs:
        return {"mae": float("inf"), "med": float("inf"), "n": 0}
    e = np.array(errs)
    return {"mae": float(e.mean()), "med": float(np.median(e)), "max": float(e.max()), "n": len(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["medium", "simple", "shallow"], default="medium")
    ap.add_argument("--board-weight", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--experiment", default="exp1")
    ap.add_argument("--output-dir", default="/tmp/gauge_polar_v5")
    args = ap.parse_args()

    conf = Config(
        epochs=args.epochs, dropout=args.dropout, lr=args.lr,
        board_weight=args.board_weight,
        add_edges=not args.no_edges,
        backbone=args.backbone,
        experiment=args.experiment, output_dir=args.output_dir,
    )

    print("=" * 60)
    print(f"POLAR GAUGE READER v5 — backbone={conf.backbone} (~150K params)")
    print(f"  sin/cos regression + cosine LR decay + augmentation")
    print(f"  Edges: {conf.add_edges}  Dropout: {conf.dropout}")
    print(f"  Board weight: {conf.board_weight}x  L2 reg: 1e-4")

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
    print(f"  {len(all_s)} samples ({len(board_s)} board)")

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
    print(f"  Train: {len(tr_w)}  Val: {len(va)}")

    tr_imgs, tr_tgts = [], []
    for s in tr_w:
        img = load_image(Path(s["path"]), s["board"], conf, pp_dir)
        if img is None:
            continue
        tr_imgs.append(img)
        tr_tgts.append([s["sin"], s["cos"]])

    va_imgs, va_tgts = [], []
    for s in va:
        img = load_image(Path(s["path"]), s["board"], conf, pp_dir)
        if img is None:
            continue
        va_imgs.append(img)
        va_tgts.append([s["sin"], s["cos"]])

    tr_ds = tf.data.Dataset.from_tensor_slices((np.stack(tr_imgs), np.stack(tr_tgts)))
    if not args.no_augment:
        tr_ds = tr_ds.map(augment_polar, num_parallel_calls=tf.data.AUTOTUNE)
    tr_ds = tr_ds.shuffle(max(len(tr_imgs) // 2, 100)).batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)
    va_ds = tf.data.Dataset.from_tensor_slices((np.stack(va_imgs), np.stack(va_tgts)))
    va_ds = va_ds.batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"  Train samples: {len(tr_imgs)}  Val: {len(va_imgs)}")

    model = build_model(conf)
    model.compile(optimizer=keras.optimizers.Adam(conf.lr), loss="mse", metrics=["mae", sort_fn])

    out = Path(conf.output_dir) / conf.experiment
    out.mkdir(parents=True, exist_ok=True)

    # Cosine decay with warm restarts — helps escape local minima
    lr_sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=conf.lr,
        decay_steps=conf.epochs * max(len(tr_imgs) // conf.batch_size, 10),
        alpha=1e-4,
    )
    model.compile(optimizer=keras.optimizers.Adam(lr_sched), loss="mse", metrics=["mae", sort_fn])

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_sort_fn", patience=30, restore_best_weights=True, mode="min"),
    ]
    hist = model.fit(tr_ds, validation_data=va_ds, epochs=conf.epochs, callbacks=cb, verbose=2)
    model.save(out / "model.keras")

    hist_d = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    hist_d["config"] = {"backbone": conf.backbone, "edges": conf.add_edges, "board_weight": conf.board_weight}
    json.dump(hist_d, open(out / "history.json", "w"), indent=2)

    r = eval_board(model, board_s, conf, pp_dir)
    print(f"\n  Board MAE: {r['mae']:.2f}°C  med={r['med']:.2f}°C  max={r.get('max', 0):.2f}°C  n={r['n']}")
    json.dump(r, open(out / "board_eval.json", "w"), indent=2)
    if r["mae"] < 5:
        print("  ✓ TARGET ACHIEVED")
    else:
        print(f"  ✗ Need <5°C (got {r['mae']:.2f}°C)")

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
