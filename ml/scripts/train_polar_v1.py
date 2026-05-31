#!/usr/bin/env python3
"""
Polar gauge reader — projects crop to polar, feeds to CNN for angle regression.

In polar space the needle is a vertical line at column = angle_column.
This makes angle detection much easier: columns are linearly related to angle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 80
    lr: float = 1e-3
    board_weight: float = 10.0
    val_split: float = 0.15
    use_augment: bool = True
    dropout: float = 0.25
    backbone: str = "simple"  # simple, mobilenetv2, shallow
    add_edges: bool = True
    output_dir: str = "/tmp/gauge_polar"
    experiment: str = "exp1"


def build_input_channels(img: np.ndarray, add_edges: bool) -> np.ndarray:
    """Optionally stack Sobel edge channels onto polar image."""
    if not add_edges:
        return img
    gray = (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    # Normalize edges to ~[0,1]
    gx = np.clip(gx * 0.5 + 0.5, 0, 1)
    gy = np.clip(gy * 0.5 + 0.5, 0, 1)
    mag = np.clip(mag / 4.0, 0, 1)
    return np.concatenate([img, gx[..., None], gy[..., None], mag[..., None]], axis=-1)


def input_channels(conf: Config) -> int:
    return 3 + (3 if conf.add_edges else 0)


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
        # Polar project (center = center of crop)
        polar = polar_project_image(arr, center_xy=(112, 112), max_radius=112, polar_size=conf.image_size)
        return build_input_channels(polar, conf.add_edges)
    except Exception as e:
        print(f"  ! {path}: {e}")
        return None


def build_dataset(samples, conf: Config, augment: bool, pp_dir: Optional[Path]):
    imgs, tgts = [], []
    for s in samples:
        img = load_image(Path(s["path"]), s["board"], conf, pp_dir)
        if img is None:
            continue
        imgs.append(img)
        tgts.append([s["sin"], s["cos"]])
    n = len(imgs)
    if n == 0:
        raise ValueError("No images loaded!")
    ds = tf.data.Dataset.from_tensor_slices((np.stack(imgs), np.stack(tgts)))
    if augment:
        ds = ds.shuffle(max(n // 2, 100))
    ds = ds.batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, n


def _simple_backbone(x, conf: Config):
    """65k-param CNN from scratch."""
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

    x = conv(x, 32, 2)
    x = res(x, 32, 0.02)
    x = layers.MaxPool2D(2)(x)
    x = res(x, 64, 0.04)
    x = layers.MaxPool2D(2)(x)
    x = res(x, 96, 0.06)
    x = layers.MaxPool2D(2)(x)
    x = res(x, 128, 0.08)
    x = layers.GlobalAveragePooling2D()(x)
    return x


def _shallow_backbone(x, conf: Config):
    """Minimal CNN — polar space is simpler, few layers should suffice."""
    ch = input_channels(conf)
    x = layers.Conv2D(32, 7, 2, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(64, 5, 2, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(128, 3, 2, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=-1)(x)
    x = layers.Activation("swish")(x)
    x = layers.GlobalAveragePooling2D()(x)
    return x


def build_model(conf: Config) -> keras.Model:
    cin = input_channels(conf)
    inp = keras.Input(shape=(conf.image_size, conf.image_size, cin))

    if conf.backbone == "simple":
        feat = _simple_backbone(inp, conf)
    elif conf.backbone == "mobilenetv2":
        feat = _mobilenetv2_backbone(inp, conf)
    elif conf.backbone == "shallow":
        feat = _shallow_backbone(inp, conf)
    else:
        raise ValueError(f"Unknown backbone: {conf.backbone}")

    x = layers.Dense(128, activation="swish")(feat)
    x = layers.Dropout(conf.dropout)(x)
    x = layers.Dense(64, activation="swish")(x)
    x = layers.Dropout(conf.dropout)(x)
    out = layers.Dense(2, activation="tanh", name="angle")(x)
    return keras.Model(inp, out)


def sort_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    dot = y_true[:, 0] * y_pred[:, 0] + y_true[:, 1] * y_pred[:, 1]
    return tf.reduce_mean(tf.acos(tf.clip_by_value(dot, -1.0, 1.0)))


def sincos_to_temp(s, c):
    a = math.degrees(math.atan2(s, c)) % 360
    return max(-30, min(50, celsius_from_inner_dial_angle_degrees(a)))


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
        return {"mae": float("inf"), "n": 0}
    e = np.array(errs)
    return {"mae": float(e.mean()), "med": float(np.median(e)), "n": len(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["simple", "mobilenetv2", "shallow"], default="simple")
    ap.add_argument("--board-weight", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--experiment", default="exp1")
    ap.add_argument("--output-dir", default="/tmp/gauge_polar")
    args = ap.parse_args()

    conf = Config(
        epochs=args.epochs, dropout=args.dropout, lr=args.lr,
        board_weight=args.board_weight, use_augment=not args.no_augment,
        add_edges=not args.no_edges, backbone=args.backbone,
        experiment=args.experiment, output_dir=args.output_dir,
    )

    print("=" * 60)
    print(f"POLAR GAUGE READER — backbone={conf.backbone} edges={conf.add_edges}")
    print(f"  Board weight: {conf.board_weight}x  Dropout: {conf.dropout}")
    print(f"  Input channels: {input_channels(conf)}")

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

    train_ds, _ = build_dataset(tr_w, conf, conf.use_augment, pp_dir)
    val_ds, _ = build_dataset(va, conf, False, pp_dir)

    model = build_model(conf)
    model.compile(optimizer=keras.optimizers.Adam(conf.lr), loss="mse", metrics=["mae", sort_fn])
    out = Path(conf.output_dir) / conf.experiment
    out.mkdir(parents=True, exist_ok=True)

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_sort_fn", patience=15, restore_best_weights=True, mode="min"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_sort_fn", factor=0.5, patience=8, min_lr=1e-6, mode="min"),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=conf.epochs, callbacks=cb, verbose=1)
    model.save(out / "model.keras")

    hist_d = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    hist_d["config"] = {"backbone": conf.backbone, "edges": conf.add_edges, "board_weight": conf.board_weight}
    json.dump(hist_d, open(out / "history.json", "w"), indent=2)

    r = eval_board(model, board_s, conf, pp_dir)
    print(f"\nBoard MAE: {r['mae']:.2f}°C  n={r['n']}")
    json.dump(r, open(out / "board_eval.json", "w"), indent=2)
    if r["mae"] < 5:
        print("  ✓ TARGET ACHIEVED")
    else:
        print(f"  ✗ Need <5°C (got {r['mae']:.2f}°C)")

    # INT8 quant
    def rep():
        for x, _ in val_ds.take(100):
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
