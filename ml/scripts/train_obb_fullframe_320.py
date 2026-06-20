"""Full-frame OBB localizer at 320x320 colour input.

Fork of train_obb_fullframe.py for the on-chip colour pipeline:
- 320x320 RGB input (matches DCMIPP YUV->RGB to NPU)
- Same compact OBB architecture (95K params, 6-value regressor)
- Uses existing _build_fullframe_obb_examples / _compute_fullframe_obb_params
  (they accept arbitrary canvas size via image_height/image_width params)
- RGB nearest-neighbor resize+pad instead of luma-only BT.601
- Adds YOLO OBB 320 dataset (DCMIPP-cropped, different preprocessing domain)
- Wider translation augmentation (25%) for positional robustness

Usage:
  cd ml && poetry run python scripts/train_obb_fullframe_320.py
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

# Limit GPU memory growth to avoid XLA OOM on 2GB GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

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
    _compute_fullframe_obb_params,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.labels import summarize_label_sweep

ML_ROOT: Path = PROJECT_ROOT
RAW_DIR: Path = ML_ROOT / "data" / "raw"
LABELLED_DIR: Path = ML_ROOT / "data" / "labelled"
ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "training"
MANIFEST_PATH: Path = ML_ROOT / "data" / "merged_geometry_board_manifest.csv"

IMAGE_HEIGHT: int = 320
IMAGE_WIDTH: int = 320
SEED: int = 21
BATCH_SIZE: int = 16
EPOCHS: int = 200
LEARNING_RATE: float = 1e-3
VAL_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15
TRANSLATION_MAX: float = 0.25





def _preprocess_colour(
    image: tf.Tensor, image_height: int, image_width: int
) -> tf.Tensor:
    """Resize-with-pad in RGB colour space (no luma conversion).

    Nearest-neighbor resize preserves dial geometry; zero-padding fills
    the rest. Normalizes to [0, 1].
    """
    image = tf.cast(image, tf.float32)

    crop_h = tf.shape(image)[0]
    crop_w = tf.shape(image)[1]
    scale = tf.minimum(
        tf.cast(image_height, tf.float32) / tf.cast(crop_h, tf.float32),
        tf.cast(image_width, tf.float32) / tf.cast(crop_w, tf.float32),
    )
    scaled_h = tf.cast(tf.cast(crop_h, tf.float32) * scale, tf.int32)
    scaled_w = tf.cast(tf.cast(crop_w, tf.float32) * scale, tf.int32)
    scaled_h = tf.maximum(scaled_h, 1)
    scaled_w = tf.maximum(scaled_w, 1)

    resized = tf.image.resize(
        image, [scaled_h, scaled_w], method="nearest"
    )

    pad_y = (image_height - scaled_h) // 2
    pad_x = (image_width - scaled_w) // 2
    pad_bottom = image_height - scaled_h - pad_y
    pad_right = image_width - scaled_w - pad_x
    padded = tf.pad(
        resized, [[pad_y, pad_bottom], [pad_x, pad_right], [0, 0]]
    )
    padded = tf.ensure_shape(padded, [image_height, image_width, 3])

    return padded / 255.0


def _load_fullframe_obb_data_colour(
    image_path: tf.Tensor,
    value: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load full image, RGB resize-with-pad, attach OBB targets."""
    image_bytes: tf.Tensor = tf.io.read_file(image_path)
    image: tf.Tensor = tf.io.decode_image(
        image_bytes, channels=3, expand_animations=False,
    )
    image = tf.ensure_shape(image, [None, None, 3])
    image = _preprocess_colour(image, image_height, image_width)
    obb_target = tf.cast(obb_params, tf.float32)
    return (
        image,
        {"obb_params": obb_target},
        {"obb_params": tf.cast(weight, tf.float32)},
    )


def _build_board_capture_examples_320(
    manifest_path: Path,
) -> list[TrainingExample]:
    """Build OBB examples from board capture rows in the manifest.

    Board captures are at their native resolution (mostly 224x224).
    OBB params are computed in 320x320 canvas space via
    _compute_fullframe_obb_params.
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
            # Manifest paths are relative to repo root
            fpath = str(Path(ML_ROOT).parent / row["image_path"])
            if fpath in seen:
                continue
            seen.add(fpath)

            source_w = int(float(row["source_width"]))
            source_h = int(float(row["source_height"]))
            cx = float(row["center_x_source"])
            cy = float(row["center_y_source"])
            radius = float(row.get("dial_radius_source", 0))
            if radius <= 0:
                continue

            obb_params = _compute_fullframe_obb_params(
                source_w, source_h,
                cx, cy, radius, radius,
                0.0,  # board captures: no rotation annotation
                IMAGE_HEIGHT, IMAGE_WIDTH,
            )

            examples.append(
                TrainingExample(
                    image_path=fpath,
                    value=float(row.get("temperature_c", 0)),
                    crop_box_xyxy=(
                        0.0, 0.0, float(source_w), float(source_h),
                    ),
                    needle_unit_xy=(0.0, 0.0),
                    obb_params=obb_params,
                )
            )

    return examples


class OBBEqualLoss(keras.losses.Loss):
    """Huber loss with equal weights on all 6 OBB parameters."""

    def __init__(
        self,
        delta: float = 0.05,
        reduction: str = "sum_over_batch_size",
        name: str = "obb_equal_loss",
    ):
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


def _random_translate(
    image: tf.Tensor, max_shift_ratio: float = TRANSLATION_MAX
) -> tf.Tensor:
    """Random translation to vary gauge position in frame."""
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    max_shift_px = tf.cast(h * max_shift_ratio, tf.int32)
    if max_shift_px <= 0:
        return image
    shift_y = tf.random.uniform(
        [], -max_shift_px, max_shift_px + 1, dtype=tf.int32
    )
    shift_x = tf.random.uniform(
        [], -max_shift_px, max_shift_px + 1, dtype=tf.int32
    )
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
    """Mild random rotation with zero fill."""
    angle_deg = tf.random.uniform([], -max_deg, max_deg)
    angle_rad = angle_deg * math.pi / 180.0
    c = tf.cos(angle_rad)
    s = tf.sin(angle_rad)
    transforms = tf.stack([c, -s, 0.0, s, c, 0.0, 0.0, 0.0])
    transforms = transforms[tf.newaxis, :]
    shape = tf.shape(image)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transforms,
        output_shape=[shape[0], shape[1]],
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]


def _augment_obb(
    image: tf.Tensor, y: dict, w: dict
) -> tuple[tf.Tensor, dict, dict]:
    """Augment with translation, rotation, then standard photometric aug."""
    image = _random_translate(image, max_shift_ratio=TRANSLATION_MAX)
    image = _random_rotate(image, max_deg=5.0)
    image = _augment_image(image)
    return image, y, w


def main() -> None:
    run_name = (
        f"obb_fullframe_320_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = ARTIFACTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Full-Frame OBB Training (320x320 colour): {run_name}")
    print(f"{'='*60}\n")

    # --- Load CVAT-labelled dataset (phone photos) ---
    print("[DATA] Loading CVAT-labelled samples...")
    samples = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
    print(f"[DATA] Loaded {len(samples)} samples from CVAT zips.")

    specs = load_gauge_specs()
    spec = specs["littlegood_home_temp_gauge_c"]

    label_summary = summarize_label_sweep(samples, spec)
    print(f"[DATA] {label_summary}")

    # --- Build full-frame OBB examples using training.py's own builder ---
    print("[DATA] Building full-frame OBB examples mapped to 320x320...")
    phone_examples, phone_dropped = _build_fullframe_obb_examples(
        samples, spec,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        strict_labels=False,
    )
    print(
        f"[DATA] Built {len(phone_examples)} phone-photo examples "
        f"({phone_dropped} dropped)."
    )

    # --- Build board capture examples ---
    print("[DATA] Building board capture examples...")
    board_examples = _build_board_capture_examples_320(MANIFEST_PATH)
    print(f"[DATA] Built {len(board_examples)} board capture examples.")

    # --- Combine and deduplicate ---
    all_examples = phone_examples + board_examples
    seen_paths: set[str] = set()
    deduped: list[TrainingExample] = []
    for ex in all_examples:
        if ex.image_path not in seen_paths:
            seen_paths.add(ex.image_path)
            deduped.append(ex)
    all_examples = deduped
    print(f"[DATA] Total unique examples: {len(all_examples)}")

    # --- Train/val/test split ---
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
    print(
        f"[DATA] Train={len(train_exs)}, Val={len(val_exs)}, "
        f"Test={len(test_exs)}"
    )

    # --- Edge weights ---
    all_split = train_exs + val_exs + test_exs
    weights = _compute_edge_weights(all_split, strength=0.75)

    def _to_arrays(exs):
        paths = np.array([e.image_path for e in exs])
        values = np.array([e.value for e in exs], dtype=np.float32)
        obb = np.array([e.obb_params for e in exs], dtype=np.float32)
        boxes = np.array([e.crop_box_xyxy for e in exs], dtype=np.float32)
        start = all_split.index(exs[0]) if exs else 0
        end = start + len(exs)
        w = np.array(weights[start:end], dtype=np.float32)
        return paths, values, obb, boxes, w

    train_paths, train_vals, train_obb, train_boxes, train_weights = (
        _to_arrays(train_exs)
    )
    val_paths, val_vals, val_obb, val_boxes, val_weights = _to_arrays(val_exs)
    test_paths, test_vals, test_obb, test_boxes, test_weights = _to_arrays(
        test_exs
    )

    # --- Build TF datasets (colour preprocessing) ---
    def _build_ds(paths, values, obb, boxes, weights_np, augment: bool):
        ds = tf.data.Dataset.from_tensor_slices(
            (paths, values, obb, boxes, weights_np)
        )
        ds = ds.map(
            lambda p, v, y, b, w: _load_fullframe_obb_data_colour(
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

    train_ds = _build_ds(
        train_paths, train_vals, train_obb, train_boxes,
        train_weights, augment=True,
    )
    val_ds = _build_ds(
        val_paths, val_vals, val_obb, val_boxes,
        val_weights, augment=False,
    )
    test_ds = _build_ds(
        test_paths, test_vals, test_obb, test_boxes,
        test_weights, augment=False,
    )

    print(
        f"[TRAIN] Datasets ready. "
        f"Train batches: {len(train_exs) // BATCH_SIZE}"
    )

    # --- Build model ---
    print("[MODEL] Building compact CNN OBB model (320x320)...")
    model = build_compact_obb_model(
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        head_units=96,
        head_dropout=0.15,
    )

    loss_fn = OBBEqualLoss(delta=0.05)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, clipnorm=1.0
        ),
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
            monitor="val_obb_params_mae",
            mode="min",
            patience=15,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_obb_params_mae",
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
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

    # --- Save Keras model ---
    model_path = run_dir / "model.keras"
    model.save(model_path)
    print(f"[SAVE] Model saved to {model_path}")

    # --- Detailed per-parameter error breakdown ---
    print("[EVAL] Per-parameter test error breakdown...")

    def _compute_per_param_errors(
        ds: tf.data.Dataset, param_names: list[str]
    ) -> dict:
        all_y_true = []
        all_y_pred = []
        for batch in ds:
            images, targets, _sample_weights = batch
            preds = model.predict(images, verbose=0)
            all_y_true.append(targets["obb_params"].numpy())
            all_y_pred.append(preds["obb_params"])
        y_true = np.concatenate(all_y_true, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)
        per_param = {}
        for i, name in enumerate(param_names):
            per_param[name] = float(
                np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
            )
        return per_param

    param_names = ["cx", "cy", "w", "h", "cos2t", "sin2t"]
    per_param = _compute_per_param_errors(test_ds, param_names)

    # Normalized -> pixel errors at 320x320
    cx_px = per_param["cx"] * IMAGE_WIDTH
    cy_px = per_param["cy"] * IMAGE_HEIGHT
    center_euclidean_px = math.sqrt(cx_px**2 + cy_px**2)
    print(
        f"[EVAL] Center MAE (normalized): cx={per_param['cx']:.4f}, "
        f"cy={per_param['cy']:.4f}"
    )
    print(
        f"[EVAL] Center MAE (pixels @320): cx={cx_px:.2f}px, "
        f"cy={cy_px:.2f}px, euclidean={center_euclidean_px:.2f}px"
    )
    for name in ["w", "h", "cos2t", "sin2t"]:
        print(f"[EVAL]   {name}: {per_param[name]:.4f}")

    # --- Export TFLite int8 ---
    print("\n[EXPORT] Converting to TFLite int8...")

    def representative_dataset():
        for batch in test_ds.take(min(10, max(1, len(test_exs) // BATCH_SIZE))):
            yield [batch[0].numpy().astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()
    tflite_path = run_dir / "model_int8.tflite"
    tflite_path.write_bytes(tflite_int8)
    print(
        f"[EXPORT] int8 TFLite: {len(tflite_int8) / 1024:.1f} KB "
        f"-> {tflite_path}"
    )

    # --- Save metrics ---
    metrics = {
        "config": {
            "model_family": "compact_obb_fullframe_320",
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "val_fraction": VAL_FRACTION,
            "test_fraction": TEST_FRACTION,
            "loss": "OBBEqualLoss(delta=0.05)",
            "phone_examples": len(phone_examples),
            "board_examples": len(board_examples),
            "train_examples": len(train_exs),
            "val_examples": len(val_exs),
            "test_examples": len(test_exs),
            "preprocessing": "colour_rgb_nearest_pad",
            "translation_max": TRANSLATION_MAX,
        },
        "label_summary": asdict(label_summary),
        "phone_dropped": phone_dropped,
        "test_metrics": test_results,
        "per_param_mae": per_param,
        "center_euclidean_px": float(f"{center_euclidean_px:.2f}"),
        "center_cx_px": float(f"{cx_px:.2f}"),
        "center_cy_px": float(f"{cy_px:.2f}"),
        "tflite_int8_kb": round(len(tflite_int8) / 1024, 1),
        "model_path": str(model_path),
    }

    (run_dir / "history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\n{'='*60}")
    print(f"  Done: {run_name}")
    print(f"  Model: {model_path}")
    print(f"  TFLite: {tflite_path}")
    print(
        f"  Test OBB MAE: "
        f"{test_results.get('obb_params_mae', 'N/A')}"
    )
    print(f"  Center px MAE @320: {center_euclidean_px:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
