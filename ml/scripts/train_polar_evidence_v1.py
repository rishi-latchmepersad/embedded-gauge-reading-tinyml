#!/usr/bin/env python3
"""
Polar evidence v1 — Classical-inspired geometry CNN.

Learns three heads:
  1. Center: predicts dial center (cx, cy) in [0, 1] normalized coords.
  2. Polar evidence: per-angle evidence logits via center-conditioned soft binning.
  3. Confidence: scalar [0, 1] indicating estimate reliability.

All decode happens outside the model using GaugeSpec:
  polar_evidence → softmax → circular mean → angle → GaugeSpec → Celsius.

Key design:
- Local, center-conditioned polar transform (not old full-frame polar warp).
- Frozen backbone, photometric augmentation only, board oversampling.
- Gauge-agnostic: swap GaugeSpec for different gauges without retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge.processing import (
    load_gauge_specs,
    GaugeSpec,
    angle_rad_to_fraction,
    fraction_to_value,
)
from embedded_gauge_reading_tinyml.models import build_mobilenetv2_polar_evidence_model
from luma_crop_detector import estimate_bright_centroid, compute_dynamic_crop, crop_and_resize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GAUGE_ID: str = "littlegood_home_temp_gauge_c"
INPUT_SIZE: int = 224
NUM_ANGLES: int = 180
SIGMA_ANGLES: float = 3.0  # Gaussian sigma in angle bins for target smoothing


@dataclass
class Config:
    image_size: int = INPUT_SIZE
    batch_size: int = 32
    epochs: int = 100
    lr: float = 3e-4
    board_weight: float = 10.0
    val_split: float = 0.15
    dropout: float = 0.3
    experiment: str = "exp1"
    output_dir: str = "/tmp/gauge_polar_evidence"


def load_metadata(p: Path) -> list[dict]:
    with open(p) as f:
        raw = json.load(f)
    samples = []
    for e in raw:
        samples.append({
            "path": str(Path(e["image_path"])),
            "center_x_norm": float(e["center_x_norm"]),
            "center_y_norm": float(e["center_y_norm"]),
            "tip_x_norm": float(e["tip_x_norm"]),
            "tip_y_norm": float(e["tip_y_norm"]),
            "temperature_c": float(e["temperature_c"]),
            "board": False,
        })
    return samples


def load_board(p: Path, input_size: int) -> list[dict]:
    samples = []
    with open(p) as f:
        for row in csv.DictReader(f):
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            tx = float(row["tip_x"])
            ty = float(row["tip_y"])
            samples.append({
                "path": str(row["image_path"]),
                "center_x_norm": cx / (input_size - 1),
                "center_y_norm": cy / (input_size - 1),
                "tip_x_norm": tx / (input_size - 1),
                "tip_y_norm": ty / (input_size - 1),
                "temperature_c": float(row["temperature_c"]),
                "board": True,
            })
    return samples


def load_image(path: Path, board: bool, conf: Config, pp_dir: Path | None) -> np.ndarray | None:
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
            arr = np.asarray(
                Image.fromarray(arr).resize((conf.image_size, conf.image_size), Image.BILINEAR),
                dtype=np.uint8,
            )
        return arr.astype(np.float32) / 255.0
    except Exception:
        return None


def compute_angle_rad(cx_norm: float, cy_norm: float, tx_norm: float, ty_norm: float) -> float:
    """Compute needle angle in radians from normalized center/tip coords."""
    dx = tx_norm - cx_norm
    dy = ty_norm - cy_norm
    return math.atan2(dy, dx)


def make_angle_distribution(angle_rad: float, num_bins: int, sigma_bins: float) -> np.ndarray:
    """Create a Gaussian-smoothed target distribution over angle bins."""
    bin_centers = np.linspace(-np.pi, np.pi, num_bins + 1, dtype=np.float32)[:num_bins]
    diff = angle_rad - bin_centers
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    dist = np.exp(-(diff ** 2) / (2.0 * sigma_bins ** 2))
    dist = dist / np.sum(dist)
    return dist


@tf.function
def augment_image(image, center, angle_dist, confidence):
    rgb = image[..., :3]
    rgb = tf.image.random_brightness(rgb, 0.15)
    rgb = tf.image.random_contrast(rgb, 0.7, 1.3)
    rgb = tf.image.random_saturation(rgb, 0.7, 1.3)
    rgb = tf.image.random_hue(rgb, 0.05)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    noise = tf.random.normal(tf.shape(image), 0.0, 0.02)
    rgb = tf.clip_by_value(rgb + noise, 0.0, 1.0)
    return rgb, center, angle_dist, confidence


def eval_board(
    model, board_samples: list[dict], conf: Config, pp_dir: Path | None, spec: GaugeSpec
) -> dict:
    """Evaluate on board captures via polar evidence decode."""
    errs = []
    for s in board_samples:
        p = Path(s["path"])
        if not p.exists():
            alt = PROJECT_ROOT / p
            if alt.exists():
                p = alt
            else:
                continue
        img = load_image(p, True, conf, pp_dir)
        if img is None:
            continue
        out = model.predict(img[None], verbose=0)
        center = out["center"][0]
        evidence = out["polar_evidence"][0]
        conf_val = float(out["confidence"][0, 0])

        # Decode: softmax → circular mean → angle → temperature
        probs = tf.nn.softmax(evidence, axis=-1).numpy()
        bin_centers = np.linspace(-np.pi, np.pi, NUM_ANGLES + 1, dtype=np.float32)[:NUM_ANGLES]
        sin_sum = np.sum(probs * np.sin(bin_centers))
        cos_sum = np.sum(probs * np.cos(bin_centers))
        angle_rad = math.atan2(sin_sum, cos_sum)

        fraction = angle_rad_to_fraction(angle_rad, spec, strict=False)
        pred_temp = fraction_to_value(fraction, spec)
        errs.append(abs(pred_temp - s["temperature_c"]))

    if not errs:
        return {"mae": float("inf"), "med": float("inf"), "max": float("inf"), "n": 0}
    e = np.array(errs)
    return {
        "mae": float(e.mean()), "med": float(np.median(e)), "max": float(e.max()),
        "p90": float(np.percentile(e, 90)) if len(e) > 1 else float(e.max()),
        "n": len(e), "under_5c": int(np.sum(e < 5.0)),
        "under_5c_rate": float(np.mean(e < 5.0)),
    }


def main():
    ap = argparse.ArgumentParser(description="Train polar evidence CNN")
    ap.add_argument("--board-weight", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-angles", type=int, default=NUM_ANGLES)
    ap.add_argument("--sigma-bins", type=float, default=SIGMA_ANGLES)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--experiment", default="exp1")
    ap.add_argument("--output-dir", default="/tmp/gauge_polar_evidence")
    args = ap.parse_args()

    conf = Config(
        epochs=args.epochs, dropout=args.dropout, lr=args.lr,
        board_weight=args.board_weight,
        experiment=args.experiment, output_dir=args.output_dir,
    )

    print("=" * 60)
    print("POLAR EVIDENCE v1 — Classical-inspired geometry CNN")
    print(f"  MobileNetV2 alpha=0.35, {args.num_angles} angle bins")
    print(f"  Dropout: {conf.dropout}  Board weight: {conf.board_weight}x  LR: {conf.lr}")
    print(f"  Augment: {not args.no_augment}  Sigma bins: {args.sigma_bins}")

    specs = load_gauge_specs()
    if GAUGE_ID not in specs:
        print(f"ERROR: Gauge '{GAUGE_ID}' not found.")
        sys.exit(1)
    spec = specs[GAUGE_ID]
    print(f"  Gauge: {spec.gauge_id} ({spec.min_value}–{spec.max_value} {spec.units})")

    root = Path(__file__).resolve().parent.parent
    pp_dir = root / "data" / "preprocessed_crops"

    all_samples = []
    mp = pp_dir / "metadata.json"
    if mp.exists():
        all_samples.extend(load_metadata(mp))
    bp = root / "data" / "board_captures_labeled_v2.csv"
    board_samples = []
    if bp.exists():
        board_samples = load_board(bp, conf.image_size)
        all_samples.extend(board_samples)
    print(f"  {len(all_samples)} samples ({len(board_samples)} board)")

    np.random.seed(42)
    ix = np.random.permutation(len(all_samples))
    vs = int(len(all_samples) * conf.val_split)
    train_samples = [all_samples[i] for i in ix[vs:]]
    val_samples = [all_samples[i] for i in ix[:vs]]

    train_weighted = []
    for s in train_samples:
        train_weighted.append(s)
        if s["board"]:
            train_weighted.extend([s] * (int(conf.board_weight) - 1))
    print(f"  Train: {len(train_weighted)} (weighted)  Val: {len(val_samples)}")

    # Precompute targets
    def prepare_data(samples_list: list[dict]) -> tuple | None:
        images, centers, angle_dists, confs = [], [], [], []
        for s in samples_list:
            img = load_image(Path(s["path"]), s["board"], conf, pp_dir)
            if img is None:
                continue
            angle_rad = compute_angle_rad(
                s["center_x_norm"], s["center_y_norm"],
                s["tip_x_norm"], s["tip_y_norm"],
            )
            angle_dist = make_angle_distribution(angle_rad, args.num_angles, args.sigma_bins)
            images.append(img)
            centers.append([s["center_x_norm"], s["center_y_norm"]])
            angle_dists.append(angle_dist)
            confs.append([1.0])
        if not images:
            return None
        return (
            np.stack(images),
            np.array(centers, dtype=np.float32),
            np.array(angle_dists, dtype=np.float32),
            np.array(confs, dtype=np.float32),
        )

    print("Loading training data...")
    train_data = prepare_data(train_weighted)
    if train_data is None:
        print("ERROR: No training data.")
        sys.exit(1)
    train_imgs, train_cent, train_dist, train_conf = train_data
    print(f"  Train: {train_imgs.shape}")

    print("Loading validation data...")
    val_data = prepare_data(val_samples)
    if val_data is not None:
        val_imgs, val_cent, val_dist, val_conf = val_data
        print(f"  Val: {val_imgs.shape}")
    else:
        val_imgs = val_cent = val_dist = val_conf = None

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_imgs, {"center": train_cent, "polar_evidence": train_dist, "confidence": train_conf})
    )
    if not args.no_augment:
        def aug_map(img, t):
            aug_img, c, d, conf = augment_image(
                img, t["center"], t["polar_evidence"], t["confidence"]
            )
            return aug_img, {"center": c, "polar_evidence": d, "confidence": conf}
        train_ds = train_ds.map(aug_map, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = (
        train_ds.shuffle(max(len(train_imgs) // 2, 100))
        .batch(conf.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    if val_data is not None:
        val_ds = tf.data.Dataset.from_tensor_slices(
            (val_imgs, {"center": val_cent, "polar_evidence": val_dist, "confidence": val_conf})
        )
        val_ds = val_ds.batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_ds = None

    model = build_mobilenetv2_polar_evidence_model(
        image_height=conf.image_size, image_width=conf.image_size,
        num_angles=args.num_angles, alpha=0.35,
        pretrained=True, backbone_trainable=False,
        head_dropout=conf.dropout,
    )
    print(f"  Model params: {model.count_params():,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=conf.lr),
        loss={
            "center": "mse",
            "polar_evidence": "categorical_crossentropy",
            "confidence": "binary_crossentropy",
        },
        loss_weights={
            "center": 1.0,
            "polar_evidence": 5.0,
            "confidence": 0.1,
        },
        metrics={
            "center": ["mae"],
            "polar_evidence": ["mae"],
        },
    )

    out = Path(conf.output_dir) / conf.experiment
    out.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True, mode="min"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, mode="min"
        ),
    ]

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=conf.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(out / "model.keras")

    hist_d = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    hist_d["config"] = {
        "board_weight": conf.board_weight,
        "num_angles": args.num_angles,
        "sigma_bins": args.sigma_bins,
    }
    json.dump(hist_d, open(out / "history.json", "w"), indent=2)

    if board_samples:
        r = eval_board(model, board_samples, conf, pp_dir, spec)
        print(f"\n  Board MAE: {r['mae']:.2f}C  med={r['med']:.2f}C  max={r.get('max', 0):.2f}C"
              f"  p90={r.get('p90', 0):.2f}C  n={r['n']}")
        print(f"  Under 5C: {r.get('under_5c', 0)}/{r['n']} ({r.get('under_5c_rate', 0)*100:.0f}%)")
        json.dump(r, open(out / "board_eval.json", "w"), indent=2)
        if r["mae"] < 5:
            print("  TARGET ACHIEVED")
        else:
            print(f"  Need <5C (got {r['mae']:.2f}C)")

    def rep():
        for i in range(min(100, len(train_imgs))):
            yield [train_imgs[i:i+1] * 255.0]

    cvt = tf.lite.TFLiteConverter.from_keras_model(model)
    cvt.optimizations = [tf.lite.Optimize.DEFAULT]
    cvt.representative_dataset = rep
    cvt.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    cvt.inference_input_type = tf.uint8
    cvt.inference_output_type = tf.float32
    with open(out / "model_int8.tflite", "wb") as f:
        f.write(cvt.convert())
    sz = (out / "model_int8.tflite").stat().st_size
    print(f"  INT8: {sz/1024:.0f} KB")


if __name__ == "__main__":
    main()
