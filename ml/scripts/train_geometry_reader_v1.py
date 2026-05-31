#!/usr/bin/env python3
"""
Geometry reader v1 — MobileNetV2 tiny (alpha=0.35) keypoint heatmap reader.

Predicts raw center/tip heatmaps from 224x224 luma crops. All gauge decode
(angle → temperature) happens outside the model using GaugeSpec, making the
model gauge-agnostic and reusable across different gauge geometries.

Key features:
- No Celsius head in the model (pure geometry: center + tip heatmaps)
- Gauge-agnostic: swap GaugeSpec for different gauges without retraining
- Frozen backbone, photometric augmentation only, board oversampling
- External decode: soft-argmax → angle → GaugeSpec → temperature
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
from tensorflow.keras import layers
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge.processing import (
    load_gauge_specs,
    GaugeSpec,
    angle_rad_to_fraction,
    fraction_to_value,
)
from embedded_gauge_reading_tinyml.heatmap_utils import (
    HeatmapConfig,
    generate_center_tip_heatmaps,
    softargmax_2d,
)
from embedded_gauge_reading_tinyml.models import build_mobilenetv2_geometry_reader
from embedded_gauge_reading_tinyml.heatmap_losses import (
    weighted_heatmap_mse_loss,
    softargmax_coordinate_loss,
)
from luma_crop_detector import estimate_bright_centroid, compute_dynamic_crop, crop_and_resize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GAUGE_ID: str = "littlegood_home_temp_gauge_c"
INPUT_SIZE: int = 224
HEATMAP_SIZE: int = 28
SIGMA_PIXELS: float = 2.5


@dataclass
class Config:
    image_size: int = INPUT_SIZE
    heatmap_size: int = HEATMAP_SIZE
    batch_size: int = 32
    epochs: int = 100
    lr: float = 3e-4
    board_weight: float = 10.0
    val_split: float = 0.15
    dropout: float = 0.3
    experiment: str = "exp1"
    output_dir: str = "/tmp/gauge_geometry_reader"


def load_metadata(p: Path) -> list[dict]:
    """Load phone-crop metadata with CVAT keypoint labels."""
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
            "board": False,
        })
    return samples


def load_board(p: Path, input_size: int) -> list[dict]:
    """Load board captures with synthetic keypoint labels from inverse mapping."""
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
    """Load and preprocess an image: luma crop for board captures, direct for phone crops."""
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
            arr = np.asarray(
                crop_and_resize(arr, cb, target_size=conf.image_size), dtype=np.uint8
            )
        elif arr.shape[:2] != (conf.image_size, conf.image_size):
            arr = np.asarray(
                Image.fromarray(arr).resize((conf.image_size, conf.image_size), Image.BILINEAR),
                dtype=np.uint8,
            )
        return arr.astype(np.float32) / 255.0
    except Exception:
        return None


@tf.function
def augment_image(image, center_hm, tip_hm):
    """Photometric augmentation: color jitter + Gaussian noise (no geometric transforms)."""
    rgb = image[..., :3]
    rgb = tf.image.random_brightness(rgb, 0.15)
    rgb = tf.image.random_contrast(rgb, 0.7, 1.3)
    rgb = tf.image.random_saturation(rgb, 0.7, 1.3)
    rgb = tf.image.random_hue(rgb, 0.05)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    noise = tf.random.normal(tf.shape(image), 0.0, 0.02)
    rgb = tf.clip_by_value(rgb + noise, 0.0, 1.0)
    return rgb, center_hm, tip_hm


def decode_heatmaps_to_temp(
    center_hm: np.ndarray,
    tip_hm: np.ndarray,
    spec: GaugeSpec,
    input_size: int = INPUT_SIZE,
) -> float:
    """Decode heatmaps to Celsius via soft-argmax → angle → GaugeSpec."""
    cx_norm, cy_norm = softargmax_2d(center_hm)
    tx_norm, ty_norm = softargmax_2d(tip_hm)
    cx = cx_norm / (center_hm.shape[1] - 1) * (input_size - 1)
    cy = cy_norm / (center_hm.shape[0] - 1) * (input_size - 1)
    tx = tx_norm / (tip_hm.shape[1] - 1) * (input_size - 1)
    ty = ty_norm / (tip_hm.shape[0] - 1) * (input_size - 1)
    dx = tx - cx
    dy = ty - cy
    angle_rad = math.atan2(dy, dx)
    fraction = angle_rad_to_fraction(angle_rad, spec, strict=False)
    temp = fraction_to_value(fraction, spec)
    return temp


def eval_board(
    model, board_samples: list[dict], conf: Config, pp_dir: Path | None, spec: GaugeSpec
) -> dict:
    """Evaluate board replay: predict heatmaps, decode to Celsius, compute errors."""
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
        center_hm = out["center_heatmap"][0, :, :, 0]
        tip_hm = out["tip_heatmap"][0, :, :, 0]
        try:
            pred_temp = decode_heatmaps_to_temp(center_hm, tip_hm, spec)
        except (ValueError, ZeroDivisionError):
            continue
        errs.append(abs(pred_temp - s["temperature_c"]))

    if not errs:
        return {"mae": float("inf"), "med": float("inf"), "max": float("inf"), "n": 0}
    e = np.array(errs)
    return {
        "mae": float(e.mean()),
        "med": float(np.median(e)),
        "max": float(e.max()),
        "p90": float(np.percentile(e, 90)) if len(e) > 1 else float(e.max()),
        "n": len(e),
        "under_5c": int(np.sum(e < 5.0)),
        "under_5c_rate": float(np.mean(e < 5.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-weight", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--heatmap-size", type=int, default=HEATMAP_SIZE)
    ap.add_argument("--sigma", type=float, default=SIGMA_PIXELS)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--experiment", default="exp1")
    ap.add_argument("--output-dir", default="/tmp/gauge_geometry_reader")
    args = ap.parse_args()

    conf = Config(
        epochs=args.epochs, dropout=args.dropout, lr=args.lr,
        board_weight=args.board_weight, heatmap_size=args.heatmap_size,
        experiment=args.experiment, output_dir=args.output_dir,
    )

    print("=" * 60)
    print("GEOMETRY READER v1 — Gauge-agnostic keypoint heatmaps")
    print(f"  MobileNetV2 alpha=0.35, {conf.heatmap_size}x{conf.heatmap_size} heatmaps")
    print(f"  Dropout: {conf.dropout}  Board weight: {conf.board_weight}x  LR: {conf.lr}")
    print(f"  Augment: {not args.no_augment}  Sigma: {args.sigma}px")

    # Load gauge spec
    specs = load_gauge_specs()
    if GAUGE_ID not in specs:
        print(f"ERROR: Gauge '{GAUGE_ID}' not found in calibration TOML. Available: {list(specs.keys())}")
        sys.exit(1)
    spec = specs[GAUGE_ID]
    print(f"  Gauge: {spec.gauge_id} ({spec.min_value}–{spec.max_value} {spec.units})")

    # Load data
    root = Path(__file__).resolve().parent.parent
    pp_dir = root / "data" / "preprocessed_crops"

    all_samples = []
    mp = pp_dir / "metadata.json"
    if mp.exists():
        all_samples.extend(load_metadata(mp))
    bp = root / "data" / "board_captures_labeled_v2.csv"
    board_samples_list = []
    if bp.exists():
        board_samples_list = load_board(bp, conf.image_size)
        all_samples.extend(board_samples_list)
    print(f"  {len(all_samples)} samples ({len(board_samples_list)} board)")

    # Train/val split
    np.random.seed(42)
    ix = np.random.permutation(len(all_samples))
    vs = int(len(all_samples) * conf.val_split)
    train_samples = [all_samples[i] for i in ix[vs:]]
    val_samples = [all_samples[i] for i in ix[:vs]]

    # Board oversampling
    train_weighted = []
    for s in train_samples:
        train_weighted.append(s)
        if s["board"]:
            train_weighted.extend([s] * (int(conf.board_weight) - 1))
    print(f"  Train: {len(train_weighted)} (weighted)  Val: {len(val_samples)}")

    # Build heatmap config
    hm_config = HeatmapConfig(
        heatmap_height=conf.heatmap_size,
        heatmap_width=conf.heatmap_size,
        sigma_pixels=args.sigma,
    )

    # Load images and generate heatmap targets
    def prepare_data(samples_list: list[dict], augment: bool):
        images, centers, tips = [], [], []
        for s in samples_list:
            img = load_image(Path(s["path"]), s["board"], conf, pp_dir)
            if img is None:
                continue
            c_hm, t_hm = generate_center_tip_heatmaps(
                s["center_x_norm"], s["center_y_norm"],
                s["tip_x_norm"], s["tip_y_norm"],
                config=hm_config,
            )
            images.append(img)
            centers.append(c_hm.astype(np.float32))
            tips.append(t_hm.astype(np.float32))
        if not images:
            return None, None, None
        return (
            np.stack(images),
            np.stack(centers)[..., None],
            np.stack(tips)[..., None],
        )

    print("Loading training data...")
    train_imgs, train_c, train_t = prepare_data(train_weighted, augment=False)
    if train_imgs is None:
        print("ERROR: No training data loaded.")
        sys.exit(1)
    print(f"  Train: {train_imgs.shape}")

    print("Loading validation data...")
    val_imgs, val_c, val_t = prepare_data(val_samples, augment=False)
    if val_imgs is not None:
        print(f"  Val: {val_imgs.shape}")
    else:
        print("  Val: empty — eval on train split only")

    # Build dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_c, train_t))
    if not args.no_augment:
        train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = (
        train_ds.shuffle(max(len(train_imgs) // 2, 100))
        .batch(conf.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    if val_imgs is not None:
        val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_c, val_t))
        val_ds = val_ds.batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_ds = None

    # Build model
    model = build_mobilenetv2_geometry_reader(
        image_height=conf.image_size,
        image_width=conf.image_size,
        heatmap_size=conf.heatmap_size,
        alpha=0.35,
        pretrained=True,
        backbone_trainable=False,
    )
    print(f"  Model params: {model.count_params():,}")

    # Compile with weighted MSE + softargmax coordinate loss per heatmap
    def center_loss(y_true, y_pred):
        return weighted_heatmap_mse_loss(y_true, y_pred) + softargmax_coordinate_loss(
            y_true, y_pred
        )

    def tip_loss(y_true, y_pred):
        return weighted_heatmap_mse_loss(y_true, y_pred) + 0.5 * softargmax_coordinate_loss(
            y_true, y_pred
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=conf.lr),
        loss={
            "center_heatmap": center_loss,
            "tip_heatmap": tip_loss,
        },
        loss_weights={
            "center_heatmap": 1.0,
            "tip_heatmap": 2.0,
        },
        metrics={
            "center_heatmap": [keras.metrics.MeanAbsoluteError(name="mae")],
            "tip_heatmap": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )

    # Train
    out = Path(conf.output_dir) / conf.experiment
    out.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True, mode="min"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, mode="min"
        ),
        keras.callbacks.ModelCheckpoint(
            str(out / "best.keras"), monitor="val_loss", save_best_only=True, mode="min"
        ),
    ]

    validation_data = val_ds if val_ds is not None else None
    hist = model.fit(
        train_ds,
        validation_data=validation_data,
        epochs=conf.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Save final model
    model.save(out / "model.keras")

    # Save history
    hist_d = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    hist_d["config"] = {
        "board_weight": conf.board_weight,
        "heatmap_size": conf.heatmap_size,
        "sigma_pixels": args.sigma,
    }
    json.dump(hist_d, open(out / "history.json", "w"), indent=2)

    # Board replay evaluation
    if board_samples_list:
        r = eval_board(model, board_samples_list, conf, pp_dir, spec)
        print(f"\n  Board MAE: {r['mae']:.2f}C  med={r['med']:.2f}C  max={r.get('max', 0):.2f}C"
              f"  p90={r.get('p90', 0):.2f}C  n={r['n']}")
        print(f"  Under 5C: {r.get('under_5c', 0)}/{r['n']} ({r.get('under_5c_rate', 0)*100:.0f}%)")
        json.dump(r, open(out / "board_eval.json", "w"), indent=2)
        if r["mae"] < 5:
            print("  \x1b[32mTARGET ACHIEVED\x1b[0m")
        else:
            print(f"  \x1b[33mNeed <5C (got {r['mae']:.2f}C)\x1b[0m")

    # INT8 export
    def rep():
        for x in train_imgs[:100]:
            yield [x[None] * 255.0]

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
