#!/usr/bin/env python3
"""
Direct gauge reader v3b — MobileNetV2 tiny (alpha=0.35) with moderate improvements.

Fixes from v3a (translation augmentation destroyed val performance):
1. NO translation — roll+crop+resize creates domain gap, model couldn't converge
2. Gaussian blur (p=0.5, kernel=3) to simulate board capture softness
3. Slightly stronger color jitter than v2 (but less than v3a)
4. Cosine LR decay for better convergence
5. Board weight 12x (up from 10x in v2)
6. Dropout 0.4 (middle ground between v2's 0.3 and v3a's 0.5)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
from luma_crop_detector import estimate_bright_centroid, compute_dynamic_crop, crop_and_resize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 150
    lr: float = 3e-4
    board_weight: float = 12.0
    val_split: float = 0.15
    dropout: float = 0.4
    experiment: str = "exp1"
    output_dir: str = "/tmp/gauge_direct_v3b"


def load_metadata(p: Path) -> list[dict]:
    with open(p) as f:
        raw = json.load(f)
    samples = []
    for e in raw:
        samples.append({
            "path": str(Path(e["image_path"])),
            "temp": float(e["temperature_c"]),
            "board": False,
        })
    return samples


def load_board(p: Path) -> list[dict]:
    samples = []
    with open(p) as f:
        for row in csv.DictReader(f):
            samples.append({
                "path": str(row["image_path"]),
                "temp": float(row["temperature_c"]),
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
        return arr.astype(np.float32) / 255.0
    except Exception as e:
        return None


def build_model(conf: Config) -> keras.Model:
    inp = keras.Input(shape=(conf.image_size, conf.image_size, 3))
    base = keras.applications.MobileNetV2(
        input_shape=(conf.image_size, conf.image_size, 3),
        include_top=False,
        weights="imagenet",
        alpha=0.35,
    )
    base.trainable = False
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="swish", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(conf.dropout)(x)
    x = layers.Dense(64, activation="swish", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(conf.dropout)(x)
    out = layers.Dense(1, activation="linear", name="temp_norm")(x)
    return keras.Model(inp, out)


@tf.function
def augment_image(image, label):
    rgb = image[..., :3]
    # Color jitter (slightly stronger than v2)
    rgb = tf.image.random_brightness(rgb, 0.2)
    rgb = tf.image.random_contrast(rgb, 0.6, 1.4)
    rgb = tf.image.random_saturation(rgb, 0.6, 1.4)
    rgb = tf.image.random_hue(rgb, 0.06)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    # Gaussian blur (p=0.5, 3x3 kernel)
    def apply_blur(x):
        return tf.nn.depthwise_conv2d(
            x[tf.newaxis, ...], 
            tf.ones((3, 3, 3, 1), dtype=tf.float32) / 9.0,
            [1, 1, 1, 1], "SAME"
        )[0]
    if tf.random.uniform([]) < 0.5:
        rgb = apply_blur(rgb)
    # Gaussian noise
    noise = tf.random.normal(tf.shape(rgb), 0.0, 0.02)
    rgb = tf.clip_by_value(rgb + noise, 0.0, 1.0)
    return tf.clip_by_value(rgb, 0.0, 1.0), label


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
        o = float(model.predict(img[None], verbose=0)[0, 0])
        errs.append(abs(o - s["temp"]))
    if not errs:
        return {"mae": float("inf"), "med": float("inf"), "n": 0}
    e = np.array(errs)
    return {"mae": float(e.mean()), "med": float(np.median(e)), "max": float(e.max()), "n": len(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-weight", type=float, default=12.0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--experiment", default="exp1")
    ap.add_argument("--output-dir", default="/tmp/gauge_direct_v3b")
    args = ap.parse_args()

    conf = Config(
        epochs=args.epochs, dropout=args.dropout, lr=args.lr,
        board_weight=args.board_weight,
        experiment=args.experiment, output_dir=args.output_dir,
    )

    print("=" * 60)
    print(f"DIRECT GAUGE READER v3b — MobileNetV2 tiny + moderate aug")
    print(f"  Dropout: {conf.dropout}  Board weight: {conf.board_weight}x  LR: {conf.lr}")
    print(f"  Augment: {not args.no_augment} (color+noise+blur, NO translation)")

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
        tr_tgts.append(s["temp"])

    va_imgs, va_tgts = [], []
    for s in va:
        img = load_image(Path(s["path"]), s["board"], conf, pp_dir)
        if img is None:
            continue
        va_imgs.append(img)
        va_tgts.append(s["temp"])

    tr_ds = tf.data.Dataset.from_tensor_slices((np.stack(tr_imgs), np.array(tr_tgts, dtype=np.float32)))
    if not args.no_augment:
        tr_ds = tr_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    tr_ds = tr_ds.shuffle(max(len(tr_imgs) // 2, 100)).batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)
    va_ds = tf.data.Dataset.from_tensor_slices((np.stack(va_imgs), np.array(va_tgts, dtype=np.float32)))
    va_ds = va_ds.batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"  Train samples: {len(tr_imgs)}  Val: {len(va_imgs)}")

    model = build_model(conf)
    lr_sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=conf.lr,
        decay_steps=conf.epochs * max(len(tr_imgs) // conf.batch_size, 10),
        alpha=1e-4,
    )
    model.compile(optimizer=keras.optimizers.Adam(lr_sched), loss="mse", metrics=["mae"])

    out = Path(conf.output_dir) / conf.experiment
    out.mkdir(parents=True, exist_ok=True)

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_mae", patience=30, restore_best_weights=True, mode="min"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=10, min_lr=1e-6, mode="min"),
    ]
    hist = model.fit(tr_ds, validation_data=va_ds, epochs=conf.epochs, callbacks=cb, verbose=2)
    model.save(out / "model.keras")

    hist_d = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    hist_d["config"] = {"board_weight": conf.board_weight}
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
