#!/usr/bin/env python3
"""
Center selector v1 — Hybrid localizer: CNN picks & refines center, classical
polar vote + GaugeSpec decode stays in firmware/postprocess.

Learns only two things:
  1. center_logits: which of 4 classical center hypotheses is best
  2. center_offset: sub-pixel residual from the best hypothesis

Training uses distance-based soft targets: each hypothesis gets a score
inversely proportional to its distance from the true (annotated) center.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from embedded_gauge_reading_tinyml.gauge.processing import (
    load_gauge_specs,
    GaugeSpec,
    angle_rad_to_fraction,
    fraction_to_value,
)
from embedded_gauge_reading_tinyml.models import build_mobilenetv2_center_selector
from embedded_gauge_reading_tinyml.hybrid_localizer import (
    compute_all_hypotheses,
    needle_angle_from_polar_vote,
    estimate_dial_radius,
    estimate_bright_centroid_on_crop,
    compute_crop_center,
    compute_image_center,
)
from luma_crop_detector import estimate_bright_centroid, compute_dynamic_crop, crop_and_resize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GAUGE_ID: str = "littlegood_home_temp_gauge_c"
INPUT_SIZE: int = 224
NUM_HYPOTHESES: int = 4
SOFT_TARGET_TEMPERATURE: float = 0.05  # lower = sharper soft targets

# Confidence gating thresholds (tuned empirically; adjustable via CLI)
CONFIDENCE_LOGIT_MARGIN: float = 1.5  # min difference between top-2 logits
CONFIDENCE_MAX_ENTROPY: float = 1.0   # max acceptable entropy (nats)
CONFIDENCE_MAX_OFFSET: float = 5.0    # max acceptable offset magnitude (pixels)
OFFSET_SCALE: float = 10.0            # scale tanh [-1,1] to pixel offset range


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
    output_dir: str = "/tmp/gauge_center_selector"
    logit_margin: float = CONFIDENCE_LOGIT_MARGIN
    max_entropy: float = CONFIDENCE_MAX_ENTROPY
    max_offset: float = CONFIDENCE_MAX_OFFSET


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
        return arr
    except Exception:
        return None


def make_soft_targets(
    hypotheses_norm: np.ndarray, true_cx: float, true_cy: float, temperature: float
) -> np.ndarray:
    """Convert hypothesis distances to soft classification targets.

    Each hypothesis gets a score = exp(-distance² / temperature²).
    Then softmax-normalized so they sum to 1.

    Args:
        hypotheses_norm: (4, 2) hypothesis centers in [0, 1] normalized coords.
        true_cx, true_cy: Ground truth center in [0, 1].
        temperature: Soft target sharpness.

    Returns:
        (4,) soft target distribution.
    """
    distances = np.sqrt(
        (hypotheses_norm[:, 0] - true_cx) ** 2 +
        (hypotheses_norm[:, 1] - true_cy) ** 2
    )
    scores = np.exp(-distances / temperature)
    return scores / np.sum(scores)


def compute_confidence(
    logits: np.ndarray,
    offset: np.ndarray,
    *,
    logit_margin_threshold: float = CONFIDENCE_LOGIT_MARGIN,
    max_entropy: float = CONFIDENCE_MAX_ENTROPY,
    max_offset: float = CONFIDENCE_MAX_OFFSET,
    offset_scale: float = OFFSET_SCALE,
) -> tuple[float, bool]:
    """Compute confidence score and high/low confidence gate from model outputs.

    Confidence is derived from three signals:
      1. Logit margin: difference between top-2 hypothesis logits
      2. Entropy: uncertainty of softmax distribution over hypotheses
      3. Offset magnitude: how far the residual refinement is from the chosen hypothesis

    Args:
        logits: (num_hypotheses,) unnormalized hypothesis scores.
        offset: (2,) tanh-scaled residual from chosen hypothesis.
        logit_margin_threshold: min acceptable margin between top-2 logits.
        max_entropy: max acceptable entropy (nats) for softmax distribution.
        max_offset: max acceptable offset magnitude in pixels.
        offset_scale: scale factor to convert tanh [-1,1] to pixels.

    Returns:
        (confidence_score, is_high_confidence) where confidence_score is in [0, 1]
        and is_high_confidence indicates whether to use the predicted center
        or fall back to classical multi-hypothesis search.
    """
    # Logit margin: how much the winning hypothesis stands out
    sorted_logits = np.sort(logits)[::-1]  # descending
    logit_margin = sorted_logits[0] - sorted_logits[1]

    # Entropy of softmax distribution
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    # Offset magnitude in pixel space
    offset_px = offset * offset_scale
    offset_mag = np.sqrt(offset_px[0] ** 2 + offset_px[1] ** 2)

    # Normalize each signal to [0, 1] confidence contribution
    # Higher margin = more confident (sigmoid-like scaling)
    margin_conf = min(logit_margin / logit_margin_threshold, 1.0)
    # Lower entropy = more confident
    entropy_conf = max(1.0 - entropy / max_entropy, 0.0)
    # Lower offset = more confident
    offset_conf = max(1.0 - offset_mag / max_offset, 0.0)

    # Weighted combination (margin is most important)
    confidence = 0.5 * margin_conf + 0.3 * entropy_conf + 0.2 * offset_conf

    # High confidence if all three signals pass their individual gates
    is_high = (
        logit_margin >= logit_margin_threshold
        and entropy <= max_entropy
        and offset_mag <= max_offset
    )

    return float(confidence), bool(is_high)


def decode_center_with_confidence(
    logits: np.ndarray,
    offset: np.ndarray,
    hypotheses_px: np.ndarray,
    *,
    offset_scale: float = OFFSET_SCALE,
) -> tuple[float, float, float, bool]:
    """Decode model outputs into a refined center with confidence gate.

    Args:
        logits: (num_hypotheses,) unnormalized hypothesis scores.
        offset: (2,) tanh-scaled residual from chosen hypothesis.
        hypotheses_px: (num_hypotheses, 2) classical center hypotheses in pixels.
        offset_scale: scale factor to convert tanh [-1,1] to pixels.

    Returns:
        (refined_cx, refined_cy, confidence_score, is_high_confidence)
    """
    best_hyp_idx = int(np.argmax(logits))
    best_hx, best_hy = hypotheses_px[best_hyp_idx]

    # Scale offset from tanh [-1, 1] to pixel offset
    refined_cx = best_hx + offset[0] * offset_scale
    refined_cy = best_hy + offset[1] * offset_scale

    confidence, is_high = compute_confidence(logits, offset, offset_scale=offset_scale)

    return refined_cx, refined_cy, confidence, is_high


def compute_fast_hypotheses(
    image: np.ndarray,
) -> np.ndarray:
    """Compute 3 fast center hypotheses (skip slow rim search).

    Returns:
        (3, 2) array: [[bright_cx, bright_cy],
                       [crop_cx, crop_cy],
                       [image_cx, image_cy]]
    """
    height, width = image.shape[:2]
    h1 = estimate_bright_centroid_on_crop(image)
    h2 = compute_crop_center(width, height)
    h3 = compute_image_center(width, height)
    return np.array([
        [h1[0], h1[1]],
        [h2[0], h2[1]],
        [h3[0], h3[1]],
    ], dtype=np.float32)


def predict_and_decode(
    model, image_224: np.ndarray, hypotheses_px: np.ndarray, dial_radius: float, spec: GaugeSpec
) -> tuple[float, float, bool]:
    """Run the full hybrid pipeline on one board capture.

    1. CNN predicts center_logits and center_offset
    2. Select hypothesis via argmax over logits
    3. Add predicted offset (scaled from tanh [-1,1] to pixels)
    4. Run polar spoke vote around refined center
    5. Decode angle → temperature via GaugeSpec

    Returns:
        (predicted_temperature, confidence_score, is_high_confidence)
    """
    inp = image_224.astype(np.float32) / 255.0
    out = model.predict(inp[None], verbose=0)
    logits = out["center_logits"][0]
    offset = out["center_offset"][0]  # tanh [-1, 1]

    refined_cx, refined_cy, confidence, is_high = decode_center_with_confidence(
        logits, offset, hypotheses_px,
    )

    height, width = image_224.shape[:2]
    dial_r = dial_radius if dial_radius else estimate_dial_radius(height)
    angle_deg = needle_angle_from_polar_vote(image_224, refined_cx, refined_cy, dial_r)
    angle_rad = math.radians(angle_deg)
    fraction = angle_rad_to_fraction(angle_rad, spec, strict=False)
    temp = fraction_to_value(fraction, spec)
    return temp, confidence, is_high


def eval_board(
    model, board_samples: list[dict], conf: Config, pp_dir: Path | None, spec: GaugeSpec
) -> dict:
    """Evaluate hybrid pipeline on board captures with confidence gating."""
    errs = []
    confidences = []
    fallback_count = 0
    high_count = 0

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

        # Compute classical hypotheses on the 224x224 crop
        hypotheses_px = compute_all_hypotheses(img)
        dial_r = estimate_dial_radius(img.shape[0])

        pred_temp, confidence, is_high = predict_and_decode(
            model, img, hypotheses_px, dial_r, spec,
        )
        confidences.append(confidence)
        if is_high:
            high_count += 1
            errs.append(abs(pred_temp - s["temperature_c"]))
        else:
            # Fallback: use classical multi-hypothesis center search
            # For now, use the crop center as a simple fallback
            # In production, this would run the full classical baseline
            fallback_count += 1
            crop_cx, crop_cy = hypotheses_px[1]  # crop center hypothesis
            angle_deg = needle_angle_from_polar_vote(img, crop_cx, crop_cy, dial_r)
            angle_rad = math.radians(angle_deg)
            fraction = angle_rad_to_fraction(angle_rad, spec, strict=False)
            fallback_temp = fraction_to_value(fraction, spec)
            errs.append(abs(fallback_temp - s["temperature_c"]))

    if not errs:
        return {"mae": float("inf"), "med": float("inf"), "max": float("inf"), "n": 0}
    e = np.array(errs)
    return {
        "mae": float(e.mean()), "med": float(np.median(e)), "max": float(e.max()),
        "p90": float(np.percentile(e, 90)) if len(e) > 1 else float(e.max()),
        "n": len(e), "under_5c": int(np.sum(e < 5.0)),
        "under_5c_rate": float(np.mean(e < 5.0)),
        "high_confidence_count": high_count,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(board_samples) if board_samples else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Train center selector CNN")
    ap.add_argument("--board-weight", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--soft-temp", type=float, default=SOFT_TARGET_TEMPERATURE)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--fast-hypotheses", action="store_true",
                    help="Skip slow rim center search (use 3 hypotheses instead of 4)")
    ap.add_argument("--logit-margin", type=float, default=CONFIDENCE_LOGIT_MARGIN,
                    help="Min logit margin for high confidence")
    ap.add_argument("--max-entropy", type=float, default=CONFIDENCE_MAX_ENTROPY,
                    help="Max entropy for high confidence")
    ap.add_argument("--max-offset", type=float, default=CONFIDENCE_MAX_OFFSET,
                    help="Max offset magnitude (pixels) for high confidence")
    ap.add_argument("--experiment", default="exp1")
    ap.add_argument("--output-dir", default="/tmp/gauge_center_selector")
    args = ap.parse_args()

    conf = Config(
        epochs=args.epochs, dropout=args.dropout, lr=args.lr,
        board_weight=args.board_weight,
        experiment=args.experiment, output_dir=args.output_dir,
        logit_margin=args.logit_margin, max_entropy=args.max_entropy,
        max_offset=args.max_offset,
    )

    num_hyps = 3 if args.fast_hypotheses else NUM_HYPOTHESES
    print("=" * 60)
    print("CENTER SELECTOR v1 — Hybrid localizer")
    print(f"  MobileNetV2 alpha=0.35, {num_hyps} hypotheses{' (fast, no rim)' if args.fast_hypotheses else ''}")
    print(f"  Dropout: {conf.dropout}  Board weight: {conf.board_weight}x  LR: {conf.lr}")
    print(f"  Augment: {not args.no_augment}  Soft temp: {args.soft_temp}")
    print(f"  Confidence: margin>={conf.logit_margin}, entropy<={conf.max_entropy}, offset<={conf.max_offset}px")

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

    def prepare_data(samples_list: list[dict]) -> tuple | None:
        images, logit_tgts, offset_tgts = [], [], []
        total = len(samples_list)
        for i, s in enumerate(samples_list):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Loading {i+1}/{total}...", flush=True)
            img = load_image(Path(s["path"]), s["board"], conf, pp_dir)
            if img is None:
                continue

            # Compute classical hypotheses in pixel coords
            if args.fast_hypotheses:
                hyp_px = compute_fast_hypotheses(img)  # (3, 2) - skip rim search
            else:
                hyp_px = compute_all_hypotheses(img)  # (4, 2) pixel coords
            # Normalize to [0, 1] for comparison with true center
            h, w = img.shape[:2]
            hyp_norm = hyp_px / np.array([[w - 1, h - 1]], dtype=np.float32)

            true_cx = s["center_x_norm"]
            true_cy = s["center_y_norm"]

            # Soft target: closer hypotheses get higher probability
            soft_tgt = make_soft_targets(hyp_norm, true_cx, true_cy, args.soft_temp)

            # Find best hypothesis (closest to true)
            best_idx = int(np.argmax(soft_tgt))
            best_hx_norm, best_hy_norm = hyp_norm[best_idx]

            # Offset target: residual from best hypothesis to true center, in [-1, 1]
            offset_tgt = np.array([
                (true_cx - best_hx_norm),
                (true_cy - best_hy_norm),
            ], dtype=np.float32)

            images.append(img.astype(np.float32) / 255.0)
            logit_tgts.append(soft_tgt)
            offset_tgts.append(offset_tgt)

        if not images:
            return None
        return (
            np.stack(images),
            np.array(logit_tgts, dtype=np.float32),
            np.array(offset_tgts, dtype=np.float32),
        )

    print("Loading training data...")
    train_data = prepare_data(train_weighted)
    if train_data is None:
        print("ERROR: No training data.")
        sys.exit(1)
    train_imgs, train_logits, train_offsets = train_data
    print(f"  Train: {train_imgs.shape}")

    print("Loading validation data...")
    val_data = prepare_data(val_samples)
    if val_data is not None:
        val_imgs, val_logits, val_offsets = val_data
        print(f"  Val: {val_imgs.shape}")
    else:
        val_imgs = val_logits = val_offsets = None

    @tf.function
    def augment_image(image, logits, offsets):
        rgb = image[..., :3]
        rgb = tf.image.random_brightness(rgb, 0.15)
        rgb = tf.image.random_contrast(rgb, 0.7, 1.3)
        rgb = tf.image.random_saturation(rgb, 0.7, 1.3)
        rgb = tf.image.random_hue(rgb, 0.05)
        rgb = tf.clip_by_value(rgb, 0.0, 1.0)
        noise = tf.random.normal(tf.shape(image), 0.0, 0.02)
        rgb = tf.clip_by_value(rgb + noise, 0.0, 1.0)
        return rgb, logits, offsets

    def make_ds(imgs, logits, offsets, augment: bool):
        ds = tf.data.Dataset.from_tensor_slices(
            (imgs, {"center_logits": logits, "center_offset": offsets})
        )
        if augment:
            ds = ds.map(
                lambda i, t: (
                    augment_image(i, t["center_logits"], t["center_offset"])[0],
                    {"center_logits": t["center_logits"], "center_offset": t["center_offset"]},
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return ds

    train_ds = (
        make_ds(train_imgs, train_logits, train_offsets, augment=not args.no_augment)
        .shuffle(max(len(train_imgs) // 2, 100))
        .batch(conf.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    if val_imgs is not None:
        val_ds = make_ds(val_imgs, val_logits, val_offsets, augment=False)
        val_ds = val_ds.batch(conf.batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_ds = None

    model = build_mobilenetv2_center_selector(
        image_height=conf.image_size, image_width=conf.image_size,
        num_hypotheses=num_hyps, alpha=0.35,
        pretrained=True, backbone_trainable=False,
        head_dropout=conf.dropout,
    )
    print(f"  Model params: {model.count_params():,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=conf.lr),
        loss={
            "center_logits": "categorical_crossentropy",
            "center_offset": "mse",
        },
        loss_weights={
            "center_logits": 1.0,
            "center_offset": 5.0,
        },
        metrics={
            "center_logits": ["mae"],
            "center_offset": ["mae"],
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
    hist_d["config"] = {"board_weight": conf.board_weight, "soft_temp": args.soft_temp}
    json.dump(hist_d, open(out / "history.json", "w"), indent=2)

    # Board replay evaluation (full hybrid pipeline)
    if board_samples:
        r = eval_board(model, board_samples, conf, pp_dir, spec)
        print(f"\n  Board MAE: {r['mae']:.2f}C  med={r['med']:.2f}C  max={r.get('max', 0):.2f}C"
              f"  p90={r.get('p90', 0):.2f}C  n={r['n']}")
        print(f"  Under 5C: {r.get('under_5c', 0)}/{r['n']} ({r.get('under_5c_rate', 0)*100:.0f}%)")
        print(f"  High confidence: {r.get('high_confidence_count', 0)}  "
              f"Fallback: {r.get('fallback_count', 0)} ({r.get('fallback_rate', 0)*100:.0f}%)")
        print(f"  Mean confidence: {r.get('mean_confidence', 0):.3f}")
        json.dump(r, open(out / "board_eval.json", "w"), indent=2)
        if r["mae"] < 5:
            print("  TARGET ACHIEVED")
        else:
            print(f"  Need <5C (got {r['mae']:.2f}C)")

    # INT8 export with proper metadata for firmware handoff
    print("\nExporting INT8 TFLite model...")
    export_center_selector_int8(
        model, train_imgs, out,
        gauge_id=GAUGE_ID,
        num_hypotheses=num_hyps,
    )


def export_center_selector_int8(
    model,
    train_imgs: np.ndarray,
    output_dir: Path,
    *,
    gauge_id: str = GAUGE_ID,
    num_hypotheses: int = NUM_HYPOTHESES,
) -> dict:
    """Export center selector as INT8 TFLite with firmware metadata.

    Args:
        model: Trained Keras model with center_logits and center_offset outputs.
        train_imgs: Training images for representative dataset calibration.
        output_dir: Directory to write exported artifacts.
        gauge_id: Gauge identifier for metadata.
        num_hypotheses: Number of classical center hypotheses.

    Returns:
        Metadata dict with export details.
    """
    # Build representative dataset
    def rep():
        indices = np.linspace(0, len(train_imgs) - 1, num=min(100, len(train_imgs)), dtype=int)
        for i in indices:
            yield [train_imgs[i:i+1] * 255.0]

    # Convert to INT8 TFLite
    cvt = tf.lite.TFLiteConverter.from_keras_model(model)
    cvt.optimizations = [tf.lite.Optimize.DEFAULT]
    cvt.representative_dataset = rep
    cvt.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    cvt.inference_input_type = tf.uint8
    cvt.inference_output_type = tf.float32  # Keep outputs float32 for firmware flexibility
    tflite_path = output_dir / "model_int8.tflite"
    with open(tflite_path, "wb") as f:
        f.write(cvt.convert())

    # Inspect TFLite I/O tensors
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    # Build metadata for firmware handoff
    metadata = {
        "model_name": model.name,
        "model_type": "center_selector",
        "gauge_id": gauge_id,
        "num_hypotheses": num_hypotheses,
        "hypothesis_order": ["bright_centroid", "crop_center", "rim_center", "image_center"],
        "input": {
            "shape": list(input_details["shape"]),
            "dtype": str(input_details["dtype"]),
            "scale": float(input_details["quantization"][0]),
            "zero_point": int(input_details["quantization"][1]),
        },
        "outputs": {},
        "offset_scale": OFFSET_SCALE,
        "confidence_thresholds": {
            "logit_margin": CONFIDENCE_LOGIT_MARGIN,
            "max_entropy": CONFIDENCE_MAX_ENTROPY,
            "max_offset_px": CONFIDENCE_MAX_OFFSET,
        },
        "tflite_path": str(tflite_path),
        "tflite_size_bytes": tflite_path.stat().st_size,
    }

    for det in output_details:
        name = det["name"]
        metadata["outputs"][name] = {
            "shape": [int(x) for x in det["shape"]],
            "dtype": str(det["dtype"]),
            "scale": float(det["quantization"][0]),
            "zero_point": int(det["quantization"][1]),
        }

    # Write metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    sz_kb = tflite_path.stat().st_size / 1024
    print(f"  INT8: {sz_kb:.0f} KB")
    print(f"  Metadata: {metadata_path}")

    return metadata


if __name__ == "__main__":
    main()
