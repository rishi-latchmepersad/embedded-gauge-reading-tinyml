#!/usr/bin/env python3
"""
Center selector v2 — Baseline-aligned 5-way hypotheses with two-phase curriculum.

Key changes from v1:
1. 5 hypotheses: bright_centroid, crop_center, board_prior, rim_center, image_center
2. Two-phase training:
   - Phase 1: frozen backbone, logits-only, soft-target temp=0.12
   - Phase 2: add offset loss (logits=5.0, offset=1.0), optional unfreeze last block
3. Calibrated abstain gate from logits margin, entropy, offset magnitude
4. Photometric augmentation only, board oversampling 10x

Downstream reader stays classical: polar vote -> angle -> GaugeSpec -> temperature.
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
    compute_fast_hypotheses,
    needle_angle_from_polar_vote,
    estimate_dial_radius,
)
from luma_crop_detector import estimate_bright_centroid, compute_dynamic_crop, crop_and_resize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GAUGE_ID: str = "littlegood_home_temp_gauge_c"
INPUT_SIZE: int = 224
NUM_HYPOTHESES: int = 5
SOFT_TARGET_TEMPERATURE: float = 0.12  # softer than v1 (0.05) for easier learning

# Confidence gating thresholds (calibrated on validation)
CONFIDENCE_LOGIT_MARGIN: float = 2.0  # min difference between top-2 logits
CONFIDENCE_MAX_ENTROPY: float = 1.2   # max acceptable entropy (nats)
CONFIDENCE_MAX_OFFSET: float = 8.0    # max acceptable offset magnitude (pixels)
OFFSET_SCALE: float = 10.0            # scale tanh [-1,1] to pixel offset range


@dataclass
class Config:
    image_size: int = INPUT_SIZE
    batch_size: int = 32
    phase1_epochs: int = 40
    phase2_epochs: int = 30
    lr: float = 3e-4
    phase2_lr: float = 1e-4
    board_weight: float = 10.0
    val_split: float = 0.15
    dropout: float = 0.3
    experiment: str = "exp1"
    output_dir: str = "/tmp/gauge_center_selector_v2"
    logit_margin: float = CONFIDENCE_LOGIT_MARGIN
    max_entropy: float = CONFIDENCE_MAX_ENTROPY
    max_offset: float = CONFIDENCE_MAX_OFFSET
    unfreeze_last_block: bool = False


def load_metadata(p: Path) -> list[dict]:
    """Load preprocessed crop metadata."""
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
    """Load board captures with labeled center/tip."""
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
    """Load and preprocess one image."""
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
    Then softmax-normalized so they sum to 1.0.

    Args:
        hypotheses_norm: (N, 2) hypothesis centers in [0, 1] normalized coords.
        true_cx, true_cy: Ground truth center in [0, 1].
        temperature: Soft target sharpness (higher = softer).

    Returns:
        (N,) soft target distribution.
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

    Returns:
        (confidence_score, is_high_confidence) where confidence_score is in [0, 1].
    """
    sorted_logits = np.sort(logits)[::-1]
    logit_margin = sorted_logits[0] - sorted_logits[1]

    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    offset_px = offset * offset_scale
    offset_mag = np.sqrt(offset_px[0] ** 2 + offset_px[1] ** 2)

    margin_conf = min(logit_margin / logit_margin_threshold, 1.0)
    entropy_conf = max(1.0 - entropy / max_entropy, 0.0)
    offset_conf = max(1.0 - offset_mag / max_offset, 0.0)

    confidence = 0.5 * margin_conf + 0.3 * entropy_conf + 0.2 * offset_conf

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

    Returns:
        (refined_cx, refined_cy, confidence_score, is_high_confidence)
    """
    best_hyp_idx = int(np.argmax(logits))
    best_hx, best_hy = hypotheses_px[best_hyp_idx]

    refined_cx = best_hx + offset[0] * offset_scale
    refined_cy = best_hy + offset[1] * offset_scale

    confidence, is_high = compute_confidence(logits, offset, offset_scale=offset_scale)

    return refined_cx, refined_cy, confidence, is_high


def predict_and_decode(
    model, image_224: np.ndarray, hypotheses_px: np.ndarray, dial_radius: float, spec: GaugeSpec
) -> tuple[float, float, bool]:
    """Run the full hybrid pipeline on one board capture.

    Returns:
        (predicted_temperature, confidence_score, is_high_confidence)
    """
    inp = image_224.astype(np.float32) / 255.0
    out = model.predict(inp[None], verbose=0)
    logits = out["center_logits"][0]
    offset = out["center_offset"][0]

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
    model, board_samples: list[dict], conf: Config, pp_dir: Path | None, spec: GaugeSpec,
    use_fast_hypotheses: bool = False,
) -> dict:
    """Evaluate hybrid pipeline on board captures with confidence gating."""
    errs = []
    confidences = []
    fallback_count = 0
    high_count = 0
    top1_correct = 0
    total_evaluated = 0

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
        if use_fast_hypotheses:
            hypotheses_px = compute_fast_hypotheses(img)
        else:
            hypotheses_px = compute_all_hypotheses(img)
        dial_r = estimate_dial_radius(img.shape[0])

        # Check top-1 hypothesis accuracy
        h, w = img.shape[:2]
        hyp_norm = hypotheses_px / np.array([[w - 1, h - 1]], dtype=np.float32)
        true_cx, true_cy = s["center_x_norm"], s["center_y_norm"]
        distances = np.sqrt((hyp_norm[:, 0] - true_cx) ** 2 + (hyp_norm[:, 1] - true_cy) ** 2)
        best_hyp_idx = int(np.argmin(distances))

        pred_temp, confidence, is_high = predict_and_decode(
            model, img, hypotheses_px, dial_r, spec,
        )
        confidences.append(confidence)
        total_evaluated += 1

        if is_high:
            high_count += 1
            errs.append(abs(pred_temp - s["temperature_c"]))
            # Check if the model picked the right hypothesis
            inp = img.astype(np.float32) / 255.0
            out = model.predict(inp[None], verbose=0)
            pred_idx = int(np.argmax(out["center_logits"][0]))
            if pred_idx == best_hyp_idx:
                top1_correct += 1
        else:
            # Fallback: use crop center as simple fallback
            fallback_count += 1
            crop_cx, crop_cy = hypotheses_px[1]
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
        "fallback_rate": fallback_count / total_evaluated if total_evaluated else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "top1_accuracy": top1_correct / high_count if high_count else 0.0,
        "total_evaluated": total_evaluated,
    }


def export_center_selector_int8(
    model,
    train_imgs: np.ndarray,
    output_dir: Path,
    *,
    gauge_id: str = GAUGE_ID,
    num_hypotheses: int = NUM_HYPOTHESES,
) -> dict:
    """Export center selector as INT8 TFLite with firmware metadata."""
    def rep():
        indices = np.linspace(0, len(train_imgs) - 1, num=min(100, len(train_imgs)), dtype=int)
        for i in indices:
            yield [train_imgs[i:i+1] * 255.0]

    cvt = tf.lite.TFLiteConverter.from_keras_model(model)
    cvt.optimizations = [tf.lite.Optimize.DEFAULT]
    cvt.representative_dataset = rep
    cvt.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    cvt.inference_input_type = tf.uint8
    cvt.inference_output_type = tf.float32
    tflite_path = output_dir / "model_int8.tflite"
    with open(tflite_path, "wb") as f:
        f.write(cvt.convert())

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    metadata = {
        "model_name": model.name,
        "model_type": "center_selector_v2",
        "gauge_id": gauge_id,
        "num_hypotheses": num_hypotheses,
        "hypothesis_order": ["bright_centroid", "crop_center", "board_prior", "rim_center", "image_center"],
        "input": {
            "shape": [int(x) for x in input_details["shape"]],
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

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    sz_kb = tflite_path.stat().st_size / 1024
    print(f"  INT8: {sz_kb:.0f} KB")
    print(f"  Metadata: {metadata_path}")

    return metadata


def main():
    ap = argparse.ArgumentParser(description="Train center selector v2")
    ap.add_argument("--board-weight", type=float, default=10.0)
    ap.add_argument("--phase1-epochs", type=int, default=40)
    ap.add_argument("--phase2-epochs", type=int, default=30)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--phase2-lr", type=float, default=1e-4)
    ap.add_argument("--soft-temp", type=float, default=SOFT_TARGET_TEMPERATURE)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--fast-hypotheses", action="store_true",
                    help="Skip slow rim center search (use 4 hypotheses instead of 5)")
    ap.add_argument("--unfreeze-last-block", action="store_true",
                    help="Unfreeze last MobileNet block in phase 2")
    ap.add_argument("--logit-margin", type=float, default=CONFIDENCE_LOGIT_MARGIN)
    ap.add_argument("--max-entropy", type=float, default=CONFIDENCE_MAX_ENTROPY)
    ap.add_argument("--max-offset", type=float, default=CONFIDENCE_MAX_OFFSET)
    ap.add_argument("--experiment", default="exp1")
    ap.add_argument("--output-dir", default="/tmp/gauge_center_selector_v2")
    args = ap.parse_args()

    conf = Config(
        phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs,
        dropout=args.dropout, lr=args.lr, phase2_lr=args.phase2_lr,
        board_weight=args.board_weight,
        experiment=args.experiment, output_dir=args.output_dir,
        logit_margin=args.logit_margin, max_entropy=args.max_entropy,
        max_offset=args.max_offset, unfreeze_last_block=args.unfreeze_last_block,
    )

    num_hyps = 4 if args.fast_hypotheses else NUM_HYPOTHESES
    hyp_label = "fast (no rim)" if args.fast_hypotheses else "full 5-way"

    print("=" * 60)
    print("CENTER SELECTOR v2 — Baseline-aligned hybrid localizer")
    print(f"  MobileNetV2 alpha=0.35, {num_hyps} hypotheses ({hyp_label})")
    print(f"  Dropout: {conf.dropout}  Board weight: {conf.board_weight}x")
    print(f"  Phase 1: {conf.phase1_epochs} epochs, logits-only, LR={conf.lr}")
    print(f"  Phase 2: {conf.phase2_epochs} epochs, logits+offset, LR={conf.phase2_lr}")
    print(f"  Augment: {not args.no_augment}  Soft temp: {args.soft_temp}")
    print(f"  Unfreeze last block: {conf.unfreeze_last_block}")
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

            if args.fast_hypotheses:
                hyp_px = compute_fast_hypotheses(img)
            else:
                hyp_px = compute_all_hypotheses(img)
            h, w = img.shape[:2]
            hyp_norm = hyp_px / np.array([[w - 1, h - 1]], dtype=np.float32)

            true_cx = s["center_x_norm"]
            true_cy = s["center_y_norm"]

            soft_tgt = make_soft_targets(hyp_norm, true_cx, true_cy, args.soft_temp)

            best_idx = int(np.argmax(soft_tgt))
            best_hx_norm, best_hy_norm = hyp_norm[best_idx]

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

    print("Loading training data...", flush=True)
    train_data = prepare_data(train_weighted)
    if train_data is None:
        print("ERROR: No training data.")
        sys.exit(1)
    train_imgs, train_logits, train_offsets = train_data
    print(f"  Train: {train_imgs.shape}", flush=True)

    print("Loading validation data...", flush=True)
    val_data = prepare_data(val_samples)
    if val_data is not None:
        val_imgs, val_logits, val_offsets = val_data
        print(f"  Val: {val_imgs.shape}", flush=True)
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

    # -----------------------------------------------------------------------
    # Phase 1: Logits-only, frozen backbone
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1: Logits-only, frozen backbone", flush=True)
    print("=" * 60, flush=True)

    model = build_mobilenetv2_center_selector(
        image_height=conf.image_size, image_width=conf.image_size,
        num_hypotheses=num_hyps, alpha=0.35,
        pretrained=True, backbone_trainable=False,
        head_dropout=conf.dropout,
    )
    print(f"  Model params: {model.count_params():,}", flush=True)

    # Phase 1: only logits loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=conf.lr),
        loss={
            "center_logits": "categorical_crossentropy",
            "center_offset": "mse",  # included but zero-weighted
        },
        loss_weights={
            "center_logits": 1.0,
            "center_offset": 0.0,  # disabled in phase 1
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
            monitor="val_loss", patience=15, restore_best_weights=True, mode="min"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, mode="min"
        ),
    ]

    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=conf.phase1_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Board eval after phase 1
    if board_samples:
        r1 = eval_board(model, board_samples, conf, pp_dir, spec, args.fast_hypotheses)
        print(f"\n  Phase 1 Board MAE: {r1['mae']:.2f}C  med={r1['med']:.2f}C  max={r1.get('max', 0):.2f}C"
              f"  p90={r1.get('p90', 0):.2f}C  n={r1['n']}")
        print(f"  Under 5C: {r1.get('under_5c', 0)}/{r1['n']} ({r1.get('under_5c_rate', 0)*100:.0f}%)")
        print(f"  High confidence: {r1.get('high_confidence_count', 0)}  "
              f"Fallback: {r1.get('fallback_count', 0)} ({r1.get('fallback_rate', 0)*100:.0f}%)")
        print(f"  Top-1 accuracy (high conf): {r1.get('top1_accuracy', 0)*100:.0f}%")
        json.dump(r1, open(out / "board_eval_phase1.json", "w"), indent=2)

    # Save phase 1 checkpoint
    model.save(out / "model_phase1.keras")
    hist1_d = {k: [float(v) for v in vs] for k, vs in hist1.history.items()}
    hist1_d["config"] = {"board_weight": conf.board_weight, "soft_temp": args.soft_temp}
    json.dump(hist1_d, open(out / "history_phase1.json", "w"), indent=2)

    # -----------------------------------------------------------------------
    # Phase 2: Add offset loss, optionally unfreeze last block
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2: Logits + offset loss", flush=True)
    print("=" * 60, flush=True)

    if conf.unfreeze_last_block:
        backbone = getattr(model, "_mobilenet_backbone", None)
        if backbone is not None:
            # Unfreeze last ~20% of backbone layers
            total_layers = len(backbone.layers)
            unfreeze_from = int(total_layers * 0.8)
            for layer in backbone.layers[unfreeze_from:]:
                layer.trainable = True
            print(f"  Unfroze {total_layers - unfreeze_from}/{total_layers} backbone layers", flush=True)
            # Recompile to pick up trainable changes
            model._compiled_optimizer = keras.optimizers.Adam(learning_rate=conf.phase2_lr)

    # Phase 2: both logits and offset with new weights
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=conf.phase2_lr),
        loss={
            "center_logits": "categorical_crossentropy",
            "center_offset": "mse",
        },
        loss_weights={
            "center_logits": 5.0,  # prioritize selection
            "center_offset": 1.0,  # secondary refinement
        },
        metrics={
            "center_logits": ["mae"],
            "center_offset": ["mae"],
        },
    )

    callbacks2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, mode="min"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, mode="min"
        ),
    ]

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=conf.phase2_epochs,
        callbacks=callbacks2,
        verbose=2,
    )

    # Final board eval
    if board_samples:
        r2 = eval_board(model, board_samples, conf, pp_dir, spec, args.fast_hypotheses)
        print(f"\n  Phase 2 Board MAE: {r2['mae']:.2f}C  med={r2['med']:.2f}C  max={r2.get('max', 0):.2f}C"
              f"  p90={r2.get('p90', 0):.2f}C  n={r2['n']}")
        print(f"  Under 5C: {r2.get('under_5c', 0)}/{r2['n']} ({r2.get('under_5c_rate', 0)*100:.0f}%)")
        print(f"  High confidence: {r2.get('high_confidence_count', 0)}  "
              f"Fallback: {r2.get('fallback_count', 0)} ({r2.get('fallback_rate', 0)*100:.0f}%)")
        print(f"  Top-1 accuracy (high conf): {r2.get('top1_accuracy', 0)*100:.0f}%")
        json.dump(r2, open(out / "board_eval_phase2.json", "w"), indent=2)
        if r2["mae"] < 5:
            print("  TARGET ACHIEVED")
        else:
            print(f"  Need <5C (got {r2['mae']:.2f}C)")

    # Save final model
    model.save(out / "model.keras")
    hist2_d = {k: [float(v) for v in vs] for k, vs in hist2.history.items()}
    hist2_d["config"] = {"board_weight": conf.board_weight, "soft_temp": args.soft_temp}
    json.dump(hist2_d, open(out / "history_phase2.json", "w"), indent=2)

    # INT8 export
    print("\nExporting INT8 TFLite model...", flush=True)
    export_center_selector_int8(
        model, train_imgs, out,
        gauge_id=GAUGE_ID,
        num_hypotheses=num_hyps,
    )


if __name__ == "__main__":
    main()
