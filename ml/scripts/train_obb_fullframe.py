"""Full-frame OBB localizer training.

Fixes the degenerate "predict frame center" solution from train_obb_improved.py:
- Presents full-frame 224x224 images (not dial-centered crops)
- OBB targets in canvas space (cx,cy vary with photo composition)
- Equal per-parameter loss weights (no 4x center bias)
- Wider translation augmentation for positional variance
- Includes real board captures from the manifest
- Trains from scratch with unfrozen backbone
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.models import build_compact_obb_model
from embedded_gauge_reading_tinyml.training import (
    TrainingExample,
    _augment_image,
    _build_fullframe_obb_examples,
    _compute_edge_weights,
    _load_fullframe_obb_data,
    _preprocess_board_style,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.labels import summarize_label_sweep

ML_ROOT: Path = PROJECT_ROOT
RAW_DIR: Path = ML_ROOT / "data" / "raw"
LABELLED_DIR: Path = ML_ROOT / "data" / "labelled"
ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "training"
MANIFEST_PATH: Path = ML_ROOT / "data" / "merged_geometry_board_manifest.csv"

IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
SEED: int = 21
BATCH_SIZE: int = 32
EPOCHS: int = 80
LEARNING_RATE: float = 1e-3
VAL_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15
BOARD_STYLE_AUGMENT_PROB: float = 0.5


def _random_translate(image: tf.Tensor, max_shift_ratio: float = 0.20) -> tf.Tensor:
    """Apply random translation to simulate gauge at different frame positions."""
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    max_shift_px = tf.cast(h * max_shift_ratio, tf.int32)
    if max_shift_px <= 0:
        return image
    shift_y = tf.random.uniform([], -max_shift_px, max_shift_px + 1, dtype=tf.int32)
    shift_x = tf.random.uniform([], -max_shift_px, max_shift_px + 1, dtype=tf.int32)
    pad_h = tf.cast(h, tf.int32) + 2 * tf.abs(max_shift_px)
    pad_w = tf.cast(w, tf.int32) + 2 * tf.abs(max_shift_px)
    padded = tf.image.resize_with_pad(image, pad_h, pad_w)
    shifted = tf.image.crop_to_bounding_box(
        padded,
        max_shift_px + shift_y,
        max_shift_px + shift_x,
        tf.cast(h, tf.int32),
        tf.cast(w, tf.int32),
    )
    return shifted


def _random_rotate(image: tf.Tensor, max_deg: float = 5.0) -> tf.Tensor:
    """Apply mild random rotation with luma-zero (board black) fill."""
    angle_deg = tf.random.uniform([], -max_deg, max_deg)
    angle_rad = angle_deg * math.pi / 180.0
    c = tf.cos(angle_rad)
    s = tf.sin(angle_rad)
    # Construct 3x3 affine matrix in flat 8-param form: [a, b, c, d, e, f, g, h]
    # ProjectiveTransform uses [a, b, c, d, e, f, g, h] where:
    #   x_out = (a*x + b*y + c) / (g*x + h*y + 1)
    #   y_out = (d*x + e*y + f) / (g*x + h*y + 1)
    transforms = tf.stack([c, -s, 0.0, s, c, 0.0, 0.0, 0.0])  # affine, no perspective
    transforms = transforms[tf.newaxis, :]  # [1, 8]
    shape = tf.shape(image)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transforms,
        output_shape=[shape[0], shape[1]],
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]


def _build_board_capture_examples(manifest_path: Path) -> list[TrainingExample]:
    """Build TrainingExamples from the manifest for board captures (224x224 images).

    Uses needle center + dial radius from the manifest to approximate OBB targets.
    """
    examples: list[TrainingExample] = []
    seen: set[str] = set()

    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "captured_images" not in row["image_path"]:
                continue
            qf = row.get("quality_flag", "")
            if qf not in ("clean", "manual"):
                continue
            # Manifest paths are relative to repo root (ml/data/captured_images/...)
            fpath = str(Path(ML_ROOT).parent / row["image_path"])
            if fpath in seen:
                continue
            seen.add(fpath)

            source_w = float(row["source_width"])
            source_h = float(row["source_height"])
            cx = float(row["center_x_source"])
            cy = float(row["center_y_source"])
            radius = float(row.get("dial_radius_source", 0))
            if radius <= 0:
                continue

            # For board captures already at 224x224, no scale/pad mapping needed
            if source_w == IMAGE_WIDTH and source_h == IMAGE_HEIGHT:
                obb_params = np.array([
                    cx / IMAGE_WIDTH, cy / IMAGE_HEIGHT,
                    2.0 * radius / IMAGE_WIDTH, 2.0 * radius / IMAGE_HEIGHT,
                    1.0, 0.0,
                ], dtype=np.float32)
            else:
                # For non-standard sizes (unlikely for captures), map to canvas
                scale = min(IMAGE_WIDTH / source_w, IMAGE_HEIGHT / source_h)
                scaled_w = source_w * scale
                scaled_h = source_h * scale
                pad_x = (IMAGE_WIDTH - scaled_w) * 0.5
                pad_y = (IMAGE_HEIGHT - scaled_h) * 0.5
                canvas_cx = cx * scale + pad_x
                canvas_cy = cy * scale + pad_y
                canvas_d = 2.0 * radius * scale
                obb_params = np.array([
                    canvas_cx / IMAGE_WIDTH, canvas_cy / IMAGE_HEIGHT,
                    canvas_d / IMAGE_WIDTH, canvas_d / IMAGE_HEIGHT,
                    1.0, 0.0,
                ], dtype=np.float32)

            examples.append(TrainingExample(
                image_path=fpath,
                value=float(row.get("temperature_c", 0)),
                crop_box_xyxy=(0.0, 0.0, source_w, source_h),
                needle_unit_xy=(0.0, 0.0),
                obb_params=obb_params,
            ))

    return examples


class OBBEqualLoss(keras.losses.Loss):
    """Huber loss with equal weights on all 6 OBB parameters (no center bias)."""

    def __init__(self, delta: float = 0.05, reduction: str = "sum_over_batch_size", name: str = "obb_equal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        err = y_pred - y_true
        abs_err = tf.abs(err)
        quadratic = tf.minimum(abs_err, self.delta)
        linear = abs_err - quadratic
        huber = 0.5 * quadratic * quadratic + self.delta * linear
        return tf.reduce_mean(huber)

    def get_config(self) -> dict[str, Any]:
        return {"delta": self.delta}


def _augment_obb(image: tf.Tensor, y: dict, w: dict) -> tuple[tf.Tensor, dict, dict]:
    """Augment with translation, rotation, then standard photometric aug."""
    image = _random_translate(image, max_shift_ratio=0.20)
    image = _random_rotate(image, max_deg=5.0)
    image = _augment_image(image)
    return image, y, w


def main() -> None:
    run_name = f"obb_fullframe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ARTIFACTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Full-Frame OBB Training: {run_name}")
    print(f"{'='*60}\n")

    # --- Load CVAT-labelled dataset (phone photos) ---
    print("[DATA] Loading CVAT-labelled samples...")
    samples = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
    print(f"[DATA] Loaded {len(samples)} samples from CVAT zips.")

    specs = load_gauge_specs()
    spec = specs["littlegood_home_temp_gauge_c"]

    label_summary = summarize_label_sweep(samples, spec)
    print(f"[DATA] {label_summary}")

    # --- Build full-frame OBB examples (phone photos) ---
    print("[DATA] Building full-frame OBB examples from phone photos...")
    phone_examples, phone_dropped = _build_fullframe_obb_examples(
        samples, spec,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        strict_labels=False,
    )
    print(f"[DATA] Built {len(phone_examples)} phone-photo examples ({phone_dropped} dropped).")

    # --- Build board capture examples from manifest ---
    print("[DATA] Building board capture examples...")
    board_examples = _build_board_capture_examples(MANIFEST_PATH)
    print(f"[DATA] Built {len(board_examples)} board capture examples.")

    # --- Combine and split ---
    all_examples = phone_examples + board_examples
    # Deduplicate by image_path
    seen_paths: set[str] = set()
    deduped: list[TrainingExample] = []
    for ex in all_examples:
        if ex.image_path not in seen_paths:
            seen_paths.add(ex.image_path)
            deduped.append(ex)
    all_examples = deduped
    print(f"[DATA] Total unique examples: {len(all_examples)}")

    # Train/val/test split
    val_test_frac = VAL_FRACTION + TEST_FRACTION
    train_exs, val_test_exs = train_test_split(
        all_examples,
        test_size=val_test_frac,
        random_state=SEED,
        shuffle=True,
    )
    val_ratio = VAL_FRACTION / val_test_frac if val_test_frac > 0 else 0.5
    val_exs, test_exs = train_test_split(
        val_test_exs,
        test_size=1.0 - val_ratio,
        random_state=SEED,
        shuffle=True,
    )
    print(f"[DATA] Train={len(train_exs)}, Val={len(val_exs)}, Test={len(test_exs)}")

    # --- Edge weights ---
    all_split = train_exs + val_exs + test_exs
    weights = _compute_edge_weights(all_split, strength=0.75)

    def _to_arrays(exs):
        paths = np.array([e.image_path for e in exs])
        values = np.array([e.value for e in exs], dtype=np.float32)
        obb = np.array([e.obb_params for e in exs], dtype=np.float32)
        boxes = np.array([e.crop_box_xyxy for e in exs], dtype=np.float32)
        # Slice weights for this split
        start = all_split.index(exs[0]) if exs else 0
        end = start + len(exs)
        w = np.array(weights[start:end], dtype=np.float32)
        return paths, values, obb, boxes, w

    train_paths, train_vals, train_obb, train_boxes, train_weights = _to_arrays(train_exs)
    val_paths, val_vals, val_obb, val_boxes, val_weights = _to_arrays(val_exs)
    test_paths, test_vals, test_obb, test_boxes, test_weights = _to_arrays(test_exs)

    # --- Build TF datasets ---
    def _build_ds(paths, values, obb, boxes, weights_np, augment: bool):
        ds = tf.data.Dataset.from_tensor_slices((paths, values, obb, boxes, weights_np))
        ds = ds.map(
            lambda p, v, y, b, w: _load_fullframe_obb_data(
                p, v, y, b, IMAGE_HEIGHT, IMAGE_WIDTH, w
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if augment:
            ds = ds.map(
                _augment_obb,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = _build_ds(train_paths, train_vals, train_obb, train_boxes, train_weights, augment=True)
    val_ds = _build_ds(val_paths, val_vals, val_obb, val_boxes, val_weights, augment=False)
    test_ds = _build_ds(test_paths, test_vals, test_obb, test_boxes, test_weights, augment=False)

    print(f"[TRAIN] Datasets ready. Train batches: {len(train_exs) // BATCH_SIZE}")

    # --- Build model ---
    print("[MODEL] Building compact CNN OBB model...")
    model = build_compact_obb_model(
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        head_units=96,
        head_dropout=0.15,
    )

    loss_fn = OBBEqualLoss(delta=0.05)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss={"obb_params": loss_fn},
        metrics={
            "obb_params": [
                keras.metrics.MeanAbsoluteError(name="mae"),
                keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
        },
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_obb_params_mae", mode="min",
            patience=15, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_obb_params_mae", mode="min",
            factor=0.5, patience=5, min_lr=1e-7,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Evaluate on test set ---
    print("[EVAL] Evaluating on test set...")
    test_results = model.evaluate(test_ds, verbose=1, return_dict=True)
    print(f"[EVAL] Test: {test_results}")

    # --- Save ---
    model_path = run_dir / "model.keras"
    model.save(model_path)
    print(f"[SAVE] Model saved to {model_path}")

    metrics = {
        "config": {
            "model_family": "compact_obb_fullframe",
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "val_fraction": VAL_FRACTION,
            "test_fraction": TEST_FRACTION,
            "board_style_augment_prob": BOARD_STYLE_AUGMENT_PROB,
            "loss": "OBBEqualLoss(delta=0.05)",
            "phone_examples": len(phone_examples),
            "board_examples": len(board_examples),
            "train_examples": len(train_exs),
            "val_examples": len(val_exs),
            "test_examples": len(test_exs),
        },
        "label_summary": asdict(label_summary),
        "phone_dropped": phone_dropped,
        "test_metrics": test_results,
        "model_path": str(model_path),
    }

    (run_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"  Done: {run_name}")
    print(f"  Model: {model_path}")
    print(f"  Test: {test_results}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
