#!/usr/bin/env python3
"""
Polar angle classifier v2 — MobileNetV2 backbone + 224-bin classification + circular decode.

Pipeline: luma crop → polar projection → MobileNetV2 → 224 angle bins → circular decode → temperature.
Matches the V28 architecture pattern that achieved 0.34°C MAE on hard cases.
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

# Gauge constants (matching polar_vote_circular_v28)
MIN_ANGLE_DEG: float = 135.0
SWEEP_DEG: float = 270.0
MIN_TEMP: float = -30.0
MAX_TEMP: float = 50.0
NUM_BINS: int = 224


@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 60
    lr: float = 3e-4
    board_weight: float = 10.0
    val_split: float = 0.15
    use_augment: bool = True
    dropout: float = 0.3
    add_edges: bool = False
    add_vote_prior: bool = False
    backbone_trainable_pct: float = 0.3
    experiment: str = "exp1"
    output_dir: str = "/tmp/gauge_polar_v2"


def input_channels(conf: Config) -> int:
    return 3 + (3 if conf.add_edges else 0) + (1 if conf.add_vote_prior else 0)


def _angle_to_bin(angle_deg: float) -> int:
    return int((angle_deg / 360.0) * NUM_BINS) % NUM_BINS


def _bin_to_angle(bin_idx: int) -> float:
    return (bin_idx / float(NUM_BINS)) * 360.0


def _circular_decode(logits: np.ndarray) -> float:
    """Circular mean decode with dead-zone masking (matching V28 firmware)."""
    # Mask dead-zone bins (outside gauge sweep: 0-135 and 405-360 wrap)
    dead = np.ones(NUM_BINS, dtype=np.float32)
    for i in range(NUM_BINS):
        angle = _bin_to_angle(i)
        # Check if angle is within the gauge sweep [135, 405] (wrapping)
        a = angle if angle >= MIN_ANGLE_DEG else angle + 360.0
        if MIN_ANGLE_DEG <= a <= MIN_ANGLE_DEG + SWEEP_DEG:
            dead[i] = 0.0
    masked = logits.copy()
    masked[dead > 0.5] = -1e9

    # Softmax
    e = np.exp(masked - np.max(masked))
    probs = e / (np.sum(e) + 1e-12)

    # Circular mean
    angles_rad = np.array([(_bin_to_angle(i) * np.pi / 180.0) for i in range(NUM_BINS)])
    sin_mean = np.sum(probs * np.sin(angles_rad))
    cos_mean = np.sum(probs * np.cos(angles_rad))
    angle_rad = np.arctan2(sin_mean, cos_mean)
    angle_deg = (angle_rad * 180.0 / np.pi) % 360.0
    return angle_deg


def _temp_from_angle_deg(angle_deg: float) -> float:
    temp = celsius_from_inner_dial_angle_degrees(angle_deg)
    return max(MIN_TEMP, min(MAX_TEMP, temp))


def _build_vote_prior(polar_img: np.ndarray) -> np.ndarray:
    """Build a simple vote-prior channel from polar image darkness + edges."""
    gray = polar_img[..., 0] * 0.299 + polar_img[..., 1] * 0.587 + polar_img[..., 2] * 0.114
    # Dark pixels are candidate needle locations
    dark = 1.0 - gray
    # Angular edge evidence
    gx = cv2.Sobel(dark.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dark.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx**2 + gy**2)
    # Combine: dark pixels with strong edges are good needle candidates
    vote = dark * np.clip(edge / (np.max(edge) + 1e-6), 0, 1)
    return np.clip(vote, 0, 1).astype(np.float32)


def _build_input(polar_img: np.ndarray, conf: Config) -> np.ndarray:
    channels = [polar_img]
    if conf.add_edges:
        gray = (polar_img[..., 0] * 0.299 + polar_img[..., 1] * 0.587 + polar_img[..., 2] * 0.114).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        channels.extend([
            np.clip(gx * 0.5 + 0.5, 0, 1),
            np.clip(gy * 0.5 + 0.5, 0, 1),
            np.clip(mag / 4.0, 0, 1),
        ])
    if conf.add_vote_prior:
        channels.append(_build_vote_prior(polar_img)[..., None])
    return np.concatenate(channels, axis=-1)


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
        return _build_input(polar, conf)
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
        # One-hot target for angle bin classification
        bin_idx = _angle_to_bin(s["angle_deg"])
        one_hot = np.zeros(NUM_BINS, dtype=np.float32)
        one_hot[bin_idx] = 1.0
        tgts.append(one_hot)
    n = len(imgs)
    if n == 0:
        raise ValueError("No images loaded!")
    ds = tf.data.Dataset.from_tensor_slices((np.stack(imgs), np.stack(tgts)))
    if augment:
        ds = ds.shuffle(max(n // 2, 100))
    ds = ds.batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, n


def build_model(conf: Config) -> keras.Model:
    cin = input_channels(conf)
    inp = keras.Input(shape=(conf.image_size, conf.image_size, cin))

    if conf.use_augment:
        x = keras.Sequential([
            layers.RandomRotation(10.0 / 360.0, fill_mode="reflect"),
            layers.RandomZoom(0.1, 0.1, fill_mode="reflect"),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.15),
        ])(inp)
    else:
        x = inp

    # MobileNetV2 backbone — use first 3 channels (polar RGB)
    base = keras.applications.MobileNetV2(
        input_shape=(conf.image_size, conf.image_size, 3),
        include_top=False, weights="imagenet",
    )
    base.trainable = False

    # Take only RGB channels for backbone
    rgb = x[..., :3]
    features = base(rgb, training=False)
    features = layers.GlobalAveragePooling2D()(features)

    # If we have extra channels, concatenate their pooled features
    if cin > 3:
        extra = x[..., 3:]
        extra_pool = layers.GlobalAveragePooling2D()(extra)
        features = layers.Concatenate()([features, extra_pool])

    # Classification head
    x = layers.Dense(256, activation="swish")(features)
    x = layers.Dropout(conf.dropout)(x)
    x = layers.Dense(128, activation="swish")(x)
    x = layers.Dropout(conf.dropout)(x)
    out = layers.Dense(NUM_BINS, activation="linear", name="angle_logits")(x)

    return keras.Model(inp, out)


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
        angle_deg = _circular_decode(logits)
        pred_temp = _temp_from_angle_deg(angle_deg)
        errs.append(abs(pred_temp - s["temp"]))
    if not errs:
        return {"mae": float("inf"), "n": 0}
    e = np.array(errs)
    return {"mae": float(e.mean()), "med": float(np.median(e)), "max": float(e.max()), "n": len(e)}


def circular_crossentropy(y_true, y_pred):
    """Cross-entropy with dead-zone masking."""
    # Mask dead-zone bins
    dead = tf.constant([0.0] * NUM_BINS, dtype=tf.float32)
    # Build mask: 1 for valid bins, 0 for dead zones
    mask = tf.py_function(_build_dead_zone_mask, [], tf.float32)
    mask.set_shape([NUM_BINS])
    masked_pred = y_pred + (1.0 - mask) * (-1e9)
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=masked_pred)
    return tf.reduce_mean(ce)


def _build_dead_zone_mask():
    dead = np.zeros(NUM_BINS, dtype=np.float32)
    for i in range(NUM_BINS):
        angle = _bin_to_angle(i)
        a = angle if angle >= MIN_ANGLE_DEG else angle + 360.0
        if MIN_ANGLE_DEG <= a <= MIN_ANGLE_DEG + SWEEP_DEG:
            dead[i] = 1.0
    return tf.constant(dead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-weight", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--add-edges", action="store_true", default=True)
    ap.add_argument("--add-vote-prior", action="store_true")
    ap.add_argument("--experiment", default="exp1")
    ap.add_argument("--output-dir", default="/tmp/gauge_polar_v2")
    args = ap.parse_args()

    conf = Config(
        epochs=args.epochs, dropout=args.dropout, lr=args.lr,
        board_weight=args.board_weight, use_augment=not args.no_augment,
        add_edges=args.add_edges and not args.no_edges,
        add_vote_prior=args.add_vote_prior,
        experiment=args.experiment, output_dir=args.output_dir,
    )

    print("=" * 60)
    print(f"POLAR ANGLE CLASSIFIER v2 — MobileNetV2 backbone")
    print(f"  Input: {conf.image_size}x{conf.image_size}x{input_channels(conf)}")
    print(f"  Bins: {NUM_BINS}, Sweep: {SWEEP_DEG}°, Range: [{MIN_TEMP}°C, {MAX_TEMP}°C]")
    print(f"  Board weight: {conf.board_weight}x  Dropout: {conf.dropout}")

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
    model.compile(
        optimizer=keras.optimizers.Adam(conf.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    out = Path(conf.output_dir) / conf.experiment
    out.mkdir(parents=True, exist_ok=True)

    # Phase 1: frozen backbone
    print("\n=== Phase 1: Frozen backbone ===")
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(str(out / "checkpoint_p1.keras"), monitor="val_accuracy", save_best_only=True),
    ]
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=conf.epochs // 2, callbacks=cb, verbose=1)

    # Phase 2: unfreeze last 30% of backbone
    print("\n=== Phase 2: Fine-tune backbone ===")
    n_layers = len(model.layers)
    for layer in model.layers:
        if hasattr(layer, '_name') and 'mobilenetv2' in layer._name.lower():
            base = layer
            unfreeze_from = int(len(base.layers) * (1 - conf.backbone_trainable_pct))
            for l in base.layers[:unfreeze_from]:
                l.trainable = False
            for l in base.layers[unfreeze_from:]:
                l.trainable = True
            print(f"  Unfroze last {len(base.layers) - unfreeze_from}/{len(base.layers)} layers")
            break

    model.compile(
        optimizer=keras.optimizers.Adam(conf.lr * 0.1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    cb2 = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(str(out / "checkpoint.keras"), monitor="val_accuracy", save_best_only=True),
    ]
    hist2 = model.fit(train_ds, validation_data=val_ds, epochs=conf.epochs // 2 + 20, callbacks=cb2, verbose=1)

    model.save(out / "model.keras")

    hist_d = {k: [float(v) for v in vs] for k, vs in hist2.history.items()}
    hist_d["config"] = {"backbone": "mobilenetv2", "edges": conf.add_edges, "vote_prior": conf.add_vote_prior}
    json.dump(hist_d, open(out / "history.json", "w"), indent=2)

    r = eval_board(model, board_s, conf, pp_dir)
    print(f"\n{'='*60}")
    print(f"Board MAE: {r['mae']:.2f}°C  n={r['n']}  max={r.get('max',0):.2f}°C")
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
