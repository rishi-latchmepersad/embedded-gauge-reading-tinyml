"""Improved OBB localizer training with per-param loss weights.

Key improvements over the baseline longterm/retrain_v2:
1. Per-parameter Huber loss weights: center (cx,cy) weighted 2x vs size (w,h) 1x vs angle 0.5x
2. Longer training (50 epochs vs 30) — the model was still improving at epoch 30
3. Board-style augmentation mixed into the OBB data pipeline
4. Slightly smaller Huber delta (0.03) for tighter regression
5. Warm-start from retrain_v2 checkpoint
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.models import build_mobilenetv2_obb_model
from embedded_gauge_reading_tinyml.training import (
    TrainConfig,
    _build_training_examples,
    _compute_crop_box,
    _split_examples,
    _augment_image,
    _load_crop_and_preprocess_image,
    _load_crop_and_preprocess_image_board_style,
    _load_crop_with_obb_weight,
    _compute_edge_weights,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.labels import summarize_label_sweep

# Resolve paths
ML_ROOT: Path = PROJECT_ROOT
RAW_DIR: Path = ML_ROOT / "data" / "raw"
LABELLED_DIR: Path = ML_ROOT / "data" / "labelled"
ARTIFACTS_DIR: Path = ML_ROOT / "artifacts" / "training"

# --- Training hyperparameters ---
SEED: int = 21
IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
BATCH_SIZE: int = 8
LEARNING_RATE: float = 5e-7
WARMUP_EPOCHS: int = 10
VAL_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15
BOARD_STYLE_AUGMENT_PROB: float = 0.3

# Train from scratch (no warm-start) with needle pivot targets
INIT_MODEL_PATH: Path | None = None
EPOCHS: int = 100
LEARNING_RATE: float = 1e-4  # higher for from-scratch training


def _load_crop_with_obb_weight_and_board_style(
    image_path: tf.Tensor,
    value: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    board_style_prob: float,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image with optional board-style preprocessing."""
    use_board = tf.random.uniform([]) < board_style_prob
    image, _ = tf.cond(
        use_board,
        lambda: _load_crop_and_preprocess_image_board_style(
            image_path, value, crop_box_xyxy, image_height, image_width
        ),
        lambda: _load_crop_and_preprocess_image(
            image_path, value, crop_box_xyxy, image_height, image_width
        ),
    )
    obb_target = tf.cast(obb_params, tf.float32)
    return image, {"obb_params": obb_target}, {"obb_params": tf.cast(weight, tf.float32)}


class OBBPerParamLoss(keras.losses.Loss):
    """Huber loss with per-parameter weights on the 6-element OBB vector.

    Weight order: [cx, cy, w, h, cos2t, sin2t]
    """

    def __init__(
        self,
        param_weights: tuple[float, ...] = (2.0, 2.0, 1.0, 1.0, 0.5, 0.5),
        delta: float = 0.03,
        reduction: str = "sum_over_batch_size",
        name: str = "obb_per_param_loss",
    ):
        super().__init__(reduction=reduction, name=name)
        self.param_weights = np.array(param_weights, dtype=np.float32)
        self.delta = delta

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute weighted Huber loss across the 6 OBB parameters."""
        err = y_pred - y_true
        abs_err = tf.abs(err)
        quadratic = tf.minimum(abs_err, self.delta)
        linear = abs_err - quadratic
        huber = 0.5 * quadratic * quadratic + self.delta * linear
        # Apply per-param weights
        weighted = huber * self.param_weights[tf.newaxis, tf.newaxis, :]
        return tf.reduce_mean(weighted)

    def get_config(self) -> dict[str, Any]:
        return {
            "param_weights": self.param_weights.tolist(),
            "delta": self.delta,
        }


def _configure_mobilenet_backbone_trainability(
    backbone: keras.Model,
    *,
    trainable: bool,
    unfreeze_last_n: int = 0,
    freeze_batchnorm: bool = True,
) -> None:
    """Set MobileNetV2 backbone trainability (matching training.py pattern)."""
    backbone.trainable = trainable
    if trainable and unfreeze_last_n > 0:
        for layer in backbone.layers[:-unfreeze_last_n]:
            layer.trainable = False
    for layer in backbone.layers:
        if hasattr(layer, "moving_mean") or hasattr(layer, "moving_var"):
            if freeze_batchnorm:
                layer.trainable = False


def main() -> None:
    print("=" * 60)
    print("  Improved OBB Localizer Training")
    print("=" * 60)

    run_name = f"obb_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ARTIFACTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TRAIN] Run directory: {run_dir}")

    # --- Load dataset ---
    print("[TRAIN] Loading labelled dataset...")
    samples = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
    print(f"[TRAIN] Loaded {len(samples)} samples.")

    specs = load_gauge_specs()
    spec = specs["littlegood_home_temp_gauge_c"]

    label_summary = summarize_label_sweep(samples, spec)
    print(f"[TRAIN] Label summary: {label_summary}")

    # --- Build training examples ---
    print("[TRAIN] Building training examples...")
    examples, dropped = _build_training_examples(
        samples,
        spec,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        keypoint_heatmap_size=28,
        strict_labels=False,
        crop_pad_ratio=0.25,
    )
    print(f"[TRAIN] Built {len(examples)} examples ({dropped} dropped).")

    # --- Split ---
    config = TrainConfig(
        gauge_id="littlegood_home_temp_gauge_c",
        seed=SEED,
        test_fraction=TEST_FRACTION,
        val_fraction=VAL_FRACTION,
    )
    split = _split_examples(examples, config)
    print(
        f"[TRAIN] Split: train={len(split.train_examples)} "
        f"val={len(split.val_examples)} test={len(split.test_examples)}"
    )

    # --- Build TF datasets ---
    print("[TRAIN] Building TF datasets...")
    all_examples = split.train_examples + split.val_examples + split.test_examples
    weights = _compute_edge_weights(all_examples, strength=0.75)

    paths = np.array([e.image_path for e in all_examples])
    values = np.array([e.value for e in all_examples], dtype=np.float32)
    obb_targets = np.array([e.obb_params for e in all_examples], dtype=np.float32)
    # Replace dial center (obb_params[0,1]) with needle pivot (center_xy).
    # The OBB was previously trained on the dial center from corner markers,
    # but polar voting needs the needle pivot point. These differ by ~7 px.
    for i, ex in enumerate(all_examples):
        obb_targets[i, 0] = ex.center_xy[0] / IMAGE_WIDTH   # cx_norm
        obb_targets[i, 1] = ex.center_xy[1] / IMAGE_HEIGHT  # cy_norm
    boxes = np.array([e.crop_box_xyxy for e in all_examples], dtype=np.float32)
    weights_np = np.array(weights, dtype=np.float32)

    n_train = len(split.train_examples)
    n_val = len(split.val_examples)
    n_test = len(split.test_examples)

    train_paths = paths[:n_train]
    train_values = values[:n_train]
    train_obb = obb_targets[:n_train]
    train_boxes = boxes[:n_train]
    train_weights = weights_np[:n_train]

    val_paths = paths[n_train : n_train + n_val]
    val_values = values[n_train : n_train + n_val]
    val_obb = obb_targets[n_train : n_train + n_val]
    val_boxes = boxes[n_train : n_train + n_val]
    val_weights = weights_np[n_train : n_train + n_val]

    test_paths = paths[n_train + n_val :]
    test_values = values[n_train + n_val :]
    test_obb = obb_targets[n_train + n_val :]
    test_boxes = boxes[n_train + n_val :]
    test_weights = weights_np[n_train + n_val :]

    print(f"[TRAIN] Train={n_train}, Val={n_val}, Test={n_test}")

    # --- Build train dataset with board-style aug ---
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_values, train_obb, train_boxes, train_weights)
    )
    train_ds = train_ds.shuffle(buffer_size=max(n_train, 1), seed=SEED)
    train_ds = train_ds.map(
        lambda p, v, y, b, w: _load_crop_with_obb_weight_and_board_style(
            p, v, y, b, IMAGE_HEIGHT, IMAGE_WIDTH, w, BOARD_STYLE_AUGMENT_PROB
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.map(
        lambda img, y, w: (_augment_image(img), y, w),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # --- Build val dataset (no augmentation) ---
    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_paths, val_values, val_obb, val_boxes, val_weights)
    )
    val_ds = val_ds.map(
        lambda p, v, y, b, w: _load_crop_with_obb_weight(
            p, v, y, b, IMAGE_HEIGHT, IMAGE_WIDTH, w
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"[TRAIN] Datasets ready. Train batches: {n_train // BATCH_SIZE}")

    # --- Build model ---
    print("[TRAIN] Building model...")
    model = build_mobilenetv2_obb_model(
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        alpha=1.0,
        head_units=128,
        head_dropout=0.2,
    )

    # Warm-start from previous checkpoint (strip old optimizer state)
    if INIT_MODEL_PATH is not None and INIT_MODEL_PATH.exists():
        print(f"[TRAIN] Warm-starting from {INIT_MODEL_PATH}")
        prev_model = keras.models.load_model(str(INIT_MODEL_PATH), compile=False)
        model.set_weights(prev_model.get_weights())
        del prev_model
    else:
        print("[TRAIN] Training from scratch (random init)")

    # --- Two-stage training: warmup frozen backbone, then unfreeze ---
    backbone = getattr(model, "_mobilenet_backbone", None)
    if backbone is None:
        raise RuntimeError("MobileNetV2 backbone handle not found.")

    # Stage 1: frozen backbone
    print(f"[TRAIN] Stage 1: warmup {WARMUP_EPOCHS} epochs, backbone frozen")
    _configure_mobilenet_backbone_trainability(
        backbone, trainable=False, freeze_batchnorm=True
    )

    loss_fn = OBBPerParamLoss(
        param_weights=(2.0, 2.0, 1.0, 1.0, 0.5, 0.5),
        delta=0.03,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss={"obb_params": loss_fn},
        metrics={
            "obb_params": [
                keras.metrics.MeanAbsoluteError(name="obb_params_mae"),
                keras.metrics.RootMeanSquaredError(name="obb_params_rmse"),
            ],
        },
    )

    callbacks_stage1 = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_obb_params_mae", mode="min", factor=0.5, patience=3, min_lr=1e-7
        ),
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=callbacks_stage1,
        verbose=1,
    )

    # Stage 2: unfreeze backbone
    remaining_epochs = EPOCHS - WARMUP_EPOCHS
    print(f"[TRAIN] Stage 2: unfreeze backbone, {remaining_epochs} more epochs")
    _configure_mobilenet_backbone_trainability(
        backbone, trainable=True, freeze_batchnorm=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss={"obb_params": loss_fn},
        metrics={
            "obb_params": [
                keras.metrics.MeanAbsoluteError(name="obb_params_mae"),
                keras.metrics.RootMeanSquaredError(name="obb_params_rmse"),
            ],
        },
    )

    callbacks_stage2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_obb_params_mae",
            mode="min",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_obb_params_mae", mode="min", factor=0.5, patience=4, min_lr=1e-7
        ),
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=remaining_epochs,
        callbacks=callbacks_stage2,
        verbose=1,
    )

    # --- Evaluate on test set ---
    print("[TRAIN] Evaluating on test set...")
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_paths, test_values, test_obb, test_boxes, test_weights)
    )
    test_ds = test_ds.map(
        lambda p, v, y, b, w: _load_crop_with_obb_weight(
            p, v, y, b, IMAGE_HEIGHT, IMAGE_WIDTH, w
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_results = model.evaluate(test_ds, verbose=1, return_dict=True)
    print(f"[TRAIN] Test results: {test_results}")

    # --- Save artifacts ---
    model_path = run_dir / "model.keras"
    model.save(model_path)

    # Merge histories
    all_history = {}
    for key in history1.history:
        all_history[key] = list(history1.history[key]) + list(history2.history[key])

    metrics = {
        "config": {
            "model_family": "mobilenet_v2_obb",
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "warmup_epochs": WARMUP_EPOCHS,
            "val_fraction": VAL_FRACTION,
            "test_fraction": TEST_FRACTION,
            "board_style_augment_prob": BOARD_STYLE_AUGMENT_PROB,
            "per_param_weights": [2.0, 2.0, 1.0, 1.0, 0.5, 0.5],
            "huber_delta": 0.03,
            "init_model": str(INIT_MODEL_PATH) if INIT_MODEL_PATH is not None and INIT_MODEL_PATH.exists() else None,
        },
        "label_summary": asdict(label_summary),
        "test_metrics": test_results,
        "model_path": str(model_path),
    }

    (run_dir / "history.json").write_text(
        json.dumps(all_history, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("=" * 60)
    print(f"  Run: {run_name}")
    print(f"  Model: {model_path}")
    print(f"  Test metrics: {test_results}")
    print("=" * 60)


if __name__ == "__main__":
    main()
