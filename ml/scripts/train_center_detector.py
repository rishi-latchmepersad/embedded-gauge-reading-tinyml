"""Train a center-detection CNN with an auxiliary needle-colour head."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.models import build_center_detection_model
from embedded_gauge_reading_tinyml.training import (
    _augment_image,
    _build_training_examples,
    _load_crop_and_preprocess_image,
    _split_examples,
    LABELLED_DIR,
    ML_ROOT,
)
from embedded_gauge_reading_tinyml.presets import (
    DEFAULT_GAUGE_ID,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
)

IMAGE_HEIGHT: int = DEFAULT_IMAGE_HEIGHT  # 224
IMAGE_WIDTH: int = DEFAULT_IMAGE_WIDTH  # 224
BATCH_SIZE: int = DEFAULT_BATCH_SIZE  # 8
SEED: int = DEFAULT_SEED
GAUGE_ID: str = DEFAULT_GAUGE_ID
VAL_FRACTION: float = 0.1
TEST_FRACTION: float = 0.1
CROP_PAD_RATIO: float = 0.25
KEYPOINT_HEATMAP_SIZE: int = 28  # not used for this model but required by API
WARMUP_EPOCHS: int = 4
FINE_TUNE_EPOCHS: int = 30
LEARNING_RATE: float = 1e-4
RUN_DIR: Path = (
    ML_ROOT / "artifacts" / "training" / f"center_detector_v1_{datetime.now():%Y%m%d_%H%M%S}"
)
HEAD_UNITS: int = 128
HEAD_DROPOUT: float = 0.2
ALPHA: float = 1.0


def _load_center_example(
    image_path: tf.Tensor,
    center_xy: tf.Tensor,
    needle_colour: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load one image and return it with the center and colour targets."""
    image, _ = _load_crop_and_preprocess_image(
        image_path,
        tf.constant(0.0, dtype=tf.float32),
        crop_box_xyxy,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )
    targets: dict[str, tf.Tensor] = {
        "center_xy": center_xy,
        "needle_colour_head": needle_colour,
    }
    return image, targets


def _build_tf_dataset(
    examples: list[Any],
    *,
    training: bool,
) -> tf.data.Dataset:
    """Build a tf.data pipeline for center detection."""
    paths: np.ndarray = np.array([e.image_path for e in examples], dtype=str)
    center_xy: np.ndarray = np.array(
        [
            [e.center_xy[0] / IMAGE_WIDTH, e.center_xy[1] / IMAGE_HEIGHT]
            for e in examples
        ],
        dtype=np.float32,
    )
    # All current gauges have dark needles (label 0); the auxiliary head is
    # included for future multi-colour gauge support.
    needle_colour: np.ndarray = np.zeros((len(examples),), dtype=np.int32)
    boxes: np.ndarray = np.array([e.crop_box_xyxy for e in examples], dtype=np.float32)

    ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
        (paths, center_xy, needle_colour, boxes)
    )

    if training:
        ds = ds.shuffle(
            buffer_size=max(len(examples), 1),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda p, c, n, b: _load_center_example(p, c, n, b),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        ds = ds.map(
            lambda img, targets: (_augment_image(img), targets),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class UnfreezeBackboneCallback(keras.callbacks.Callback):
    """Unfreeze the MobileNetV2 backbone after a warmup phase."""

    def __init__(self, backbone: keras.Model, unfreeze_epoch: int = 4) -> None:
        super().__init__()
        self._backbone = backbone
        self._unfreeze_epoch = unfreeze_epoch
        self._unfrozen: bool = False

    def on_epoch_begin(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        if epoch == self._unfreeze_epoch and not self._unfrozen:
            self._backbone.trainable = True
            self._unfrozen = True
            print(
                f"\n[TRAIN] Unfreezing backbone at epoch {epoch + 1}. "
                "Keras will pick up trainable params automatically."
            )


def main() -> None:
    """Run the center-detector training pipeline."""
    print("=" * 60)
    print("Center Detector Training")
    print(f"  Run dir: {RUN_DIR}")
    print("=" * 60)

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset and build training examples ──────────────────────
    print("\n[1] Loading CVAT dataset...")
    samples = load_dataset()
    print(f"  Loaded {len(samples)} labelled samples.")

    spec = load_gauge_specs()[GAUGE_ID]
    examples, dropped = _build_training_examples(
        samples,
        spec,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        keypoint_heatmap_size=KEYPOINT_HEATMAP_SIZE,
        strict_labels=False,
        crop_pad_ratio=CROP_PAD_RATIO,
    )
    print(f"  Built {len(examples)} training examples ({dropped} dropped).")

    # ── 2. Split ────────────────────────────────────────────────────────
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _SimpleConfig:
        val_fraction: float
        test_fraction: float
        seed: int
        val_manifest: None = None
        test_manifest: None = None
        image_height: int = IMAGE_HEIGHT
        image_width: int = IMAGE_WIDTH

    split = _split_examples(
        examples,
        _SimpleConfig(val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED),
    )
    print(
        f"  Split: train={len(split.train_examples)}, "
        f"val={len(split.val_examples)}, "
        f"test={len(split.test_examples)}"
    )

    # ── 3. Build TF datasets ────────────────────────────────────────────
    print("\n[2] Building TF datasets...")
    train_ds = _build_tf_dataset(split.train_examples, training=True)
    val_ds = _build_tf_dataset(split.val_examples, training=False)

    # ── 4. Build model ──────────────────────────────────────────────────
    print("\n[3] Building model...")
    model = build_center_detection_model(
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        alpha=ALPHA,
        head_units=HEAD_UNITS,
        head_dropout=HEAD_DROPOUT,
    )
    backbone: keras.Model = getattr(model, "_mobilenet_backbone")
    backbone.trainable = False
    model.summary()

    # ── 5. Compile (warmup phase — backbone frozen) ─────────────────────
    print("\n[4] Warmup phase (backbone frozen)...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "center_xy": keras.losses.MeanSquaredError(),
            "needle_colour_head": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={
            "center_xy": 1.0,
            "needle_colour_head": 0.1,
        },
        metrics={
            "center_xy": [keras.metrics.MeanAbsoluteError(name="mean_absolute_error")],
            "needle_colour_head": ["accuracy"],
        },
    )

    # ── 6. Callbacks ────────────────────────────────────────────────────
    unfreeze_cb = UnfreezeBackboneCallback(backbone, unfreeze_epoch=WARMUP_EPOCHS)
    callbacks: list[keras.callbacks.Callback] = [
        unfreeze_cb,
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_center_xy_mean_absolute_error",
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_center_xy_mean_absolute_error",
            mode="min",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(RUN_DIR / "best_model.keras"),
            monitor="val_center_xy_mean_absolute_error",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(RUN_DIR / "training_log.csv")),
    ]
    # ── 7. Train ────────────────────────────────────────────────────────
    print("\n[5] Training...")
    total_epochs = WARMUP_EPOCHS + FINE_TUNE_EPOCHS
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 8. Save final model and metrics ─────────────────────────────────
    print("\n[6] Saving artifacts...")
    model.save(RUN_DIR / "final_model.keras")

    # Save history
    history_dict: dict[str, list[float]] = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    with open(RUN_DIR / "history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    # Save metrics summary
    best_epoch = int(np.argmin(history_dict.get("val_center_xy_mean_absolute_error", [float("inf")])))
    summary = {
        "run_dir": str(RUN_DIR),
        "gauge_id": GAUGE_ID,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "warmup_epochs": WARMUP_EPOCHS,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "head_units": HEAD_UNITS,
        "head_dropout": HEAD_DROPOUT,
        "alpha": ALPHA,
        "best_epoch": int(best_epoch),
        "best_val_center_xy_mae": float(
            history_dict.get("val_center_xy_mean_absolute_error", [float("inf")])[best_epoch]
        ),
        "train_samples": len(split.train_examples),
        "val_samples": len(split.val_examples),
        "test_samples": len(split.test_examples),
    }
    with open(RUN_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Run dir: {RUN_DIR}")
    print(f"  Best val center_xy MAE: {summary['best_val_center_xy_mae']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
