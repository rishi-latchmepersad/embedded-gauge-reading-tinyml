#!/usr/bin/env python3
"""
Sweep logits v2 — fewer bins (30), no backbone unfreezing, longer head training.

Key changes from v1:
1. 30 bins instead of 90 (wider 9.7°C per bin, Gaussian sigma=1.5 smooths boundaries)
2. No backbone unfreezing — Phase 2 removed (was destroying generalization)
3. Cosine LR decay for better convergence
4. Train head only for 100 epochs with patience=30
5. Keep both v1/v1b augmentation variants
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge.processing import (
    GaugeSpec,
    value_to_fraction,
    fraction_to_value,
)
from embedded_gauge_reading_tinyml.models import build_mobilenetv2_sweep_logits_model
from luma_crop_detector import estimate_bright_centroid, compute_dynamic_crop, crop_and_resize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GAUGE_SPEC = GaugeSpec(
    gauge_id="littlegood_home_temp_gauge_c",
    min_angle_rad=math.radians(135.0),
    sweep_rad=math.radians(270.0),
    min_value=-30.0,
    max_value=50.0,
    units="C",
    direction="clockwise",
)

VALUE_MIN: float = -30.0
VALUE_MAX: float = 50.0
NUM_BINS: int = 30
SOFT_TARGET_SIGMA: float = 1.5
DECODE_TOPK: int = 5


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 100
    lr: float = 5e-4
    board_weight: float = 10.0
    val_split: float = 0.15
    dropout: float = 0.3
    variant: Literal["v1", "v1b"] = "v1"
    experiment: str = "exp1"
    output_dir: str = "/tmp/gauge_sweep_logits_v2"


def fraction_to_soft_target(
    fraction: float,
    *,
    num_bins: int = NUM_BINS,
    sigma_bins: float = SOFT_TARGET_SIGMA,
) -> np.ndarray:
    f = min(max(float(fraction), 0.0), 1.0)
    center = f * float(num_bins - 1)
    indices = np.arange(num_bins, dtype=np.float32)
    distances = np.abs(indices - np.float32(center))
    target = np.exp(-0.5 * (distances / np.float32(sigma_bins)) ** 2)
    total = float(np.sum(target))
    if total > 0.0:
        target /= np.float32(total)
    return target.astype(np.float32)


def decode_sweep_logits(
    logits: np.ndarray,
    *,
    num_bins: int = NUM_BINS,
    topk: int = DECODE_TOPK,
) -> float:
    flat = np.asarray(logits, dtype=np.float32).reshape(-1)
    probs = np.exp(flat - np.max(flat))
    probs /= np.sum(probs)
    indices = np.argsort(probs)[-topk:]
    fraction = float(np.sum(probs[indices] * (indices / float(num_bins - 1))) / np.sum(probs[indices]))
    return fraction_to_value(fraction, GAUGE_SPEC)


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
    except Exception:
        return None


def augment_v1(image, label):
    rgb = image[..., :3]
    rgb = tf.image.random_brightness(rgb, 0.15)
    rgb = tf.image.random_contrast(rgb, 0.7, 1.3)
    rgb = tf.image.random_saturation(rgb, 0.7, 1.3)
    rgb = tf.image.random_hue(rgb, 0.05)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    noise = tf.random.normal(tf.shape(rgb), 0.0, 0.02)
    rgb = tf.clip_by_value(rgb + noise, 0.0, 1.0)
    return tf.clip_by_value(rgb, 0.0, 1.0), label


def augment_v1b(image, label):
    rgb = image[..., :3]
    rgb = tf.image.random_brightness(rgb, 0.2)
    rgb = tf.image.random_contrast(rgb, 0.6, 1.4)
    rgb = tf.image.random_saturation(rgb, 0.6, 1.4)
    rgb = tf.image.random_hue(rgb, 0.06)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    def apply_blur(x):
        return tf.nn.depthwise_conv2d(
            x[tf.newaxis, ...],
            tf.ones((3, 3, 3, 1), dtype=tf.float32) / 9.0,
            [1, 1, 1, 1], "SAME"
        )[0]
    if tf.random.uniform([]) < 0.5:
        rgb = apply_blur(rgb)
    noise = tf.random.normal(tf.shape(rgb), 0.0, 0.02)
    rgb = tf.clip_by_value(rgb + noise, 0.0, 1.0)
    return tf.clip_by_value(rgb, 0.0, 1.0), label


def make_targets(temps: np.ndarray) -> np.ndarray:
    targets = []
    for t in temps:
        f = value_to_fraction(float(t), GAUGE_SPEC)
        targets.append(fraction_to_soft_target(f))
    return np.stack(targets, axis=0)


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
        pred = decode_sweep_logits(logits)
        errs.append(abs(pred - s["temp"]))
    if not errs:
        return {"mae": float("inf"), "med": float("inf"), "n": 0}
    e = np.array(errs)
    return {"mae": float(e.mean()), "med": float(np.median(e)), "max": float(e.max()), "p90": float(np.percentile(e, 90)), "n": len(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v1", "v1b"], default="v1")
    ap.add_argument("--board-weight", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--experiment", default="exp1")
    ap.add_argument("--output-dir", default="/tmp/gauge_sweep_logits_v2")
    args = ap.parse_args()

    conf = Config(
        variant=args.variant,
        epochs=args.epochs,
        dropout=args.dropout,
        lr=args.lr,
        board_weight=args.board_weight,
        experiment=args.experiment,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print(f"SWEEP LOGITS v2 — 30 bins, no unfreeze")
    print(f"  Variant: {conf.variant}  Dropout: {conf.dropout}  Board weight: {conf.board_weight}x")
    print(f"  Epochs: {conf.epochs}  LR: {conf.lr}  CosineDecay")

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
    print(f"  Train (weighted): {len(tr_w)}  Val: {len(va)}")

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

    tr_targets = make_targets(np.array(tr_tgts, dtype=np.float32))
    va_targets = make_targets(np.array(va_tgts, dtype=np.float32))

    aug_fn = augment_v1 if conf.variant == "v1" else augment_v1b
    tr_ds = tf.data.Dataset.from_tensor_slices((np.stack(tr_imgs), tr_targets))
    if not args.no_augment:
        tr_ds = tr_ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    tr_ds = tr_ds.shuffle(max(len(tr_imgs) // 2, 100)).batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)
    va_ds = tf.data.Dataset.from_tensor_slices((np.stack(va_imgs), va_targets))
    va_ds = va_ds.batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"  Train images: {len(tr_imgs)}  Val: {len(va_imgs)}")

    model = build_mobilenetv2_sweep_logits_model(
        conf.image_size, conf.image_size,
        num_bins=NUM_BINS,
        head_dropout=conf.dropout,
    )

    out = Path(conf.output_dir) / conf.experiment
    out.mkdir(parents=True, exist_ok=True)

    lr_sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=conf.lr,
        decay_steps=conf.epochs * max(len(tr_imgs) // conf.batch_size, 10),
        alpha=1e-4,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_sched),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, mode="min"),
    ]
    model.fit(tr_ds, validation_data=va_ds, epochs=conf.epochs, callbacks=cb, verbose=2)

    model.save(out / "model.keras")

    r = eval_board(model, board_s, conf, pp_dir)
    print(f"\n  Board MAE: {r['mae']:.2f}°C  med={r['med']:.2f}°C  max={r.get('max', 0):.2f}°C  p90={r.get('p90', 0):.2f}°C  n={r['n']}")
    with open(out / "board_eval.json", "w") as f:
        json.dump(r, f, indent=2)
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
    tflite_path = out / "model_int8.tflite"
    with open(tflite_path, "wb") as f:
        f.write(cvt.convert())
    sz = tflite_path.stat().st_size
    print(f"  INT8: {sz/1024:.0f} KB")

    with open(out / "config.json", "w") as f:
        json.dump({
            "variant": conf.variant,
            "dropout": conf.dropout,
            "board_weight": conf.board_weight,
            "num_bins": NUM_BINS,
            "epochs": int(conf.epochs),
            "lr": conf.lr,
            "board_mae": r["mae"],
            "board_med": r["med"],
        }, f, indent=2)


if __name__ == "__main__":
    main()
