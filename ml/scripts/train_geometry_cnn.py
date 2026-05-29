#!/usr/bin/env python3
"""Train heatmap CNN with transfer learning on merged geometry + board capture data.

Uses the proven v4_112 architecture (MobileNetV2 backbone + progressive decoder)
with a two-phase training schedule:
  Phase 1: Frozen backbone, train decoder head only
  Phase 2: Unfreeze backbone, fine-tune everything at 10x lower LR

The merged manifest combines:
  - 344 geometry phone photos (human-labeled keypoints with crop boxes)
  - 76 board captures (224x224, synthetic keypoints from inverse temp mapping)

Evaluation uses the full inference pipeline:
  luma_crop → CNN → soft-argmax → angle → temperature

Usage:
    cd ml && poetry run python scripts/train_geometry_cnn.py --epochs 60 --augment
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)
from embedded_gauge_reading_tinyml.models_geometry import (
    build_mobilenetv2_geometry_heatmap_v4_112,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEATMAP_SIZE = 112
INPUT_SIZE = 224
SIGMA_PIXELS = 12.0  # wider Gaussian = stronger gradient signal; 66% pixels non-zero

DEFAULT_CROPS_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed_crops"
DEFAULT_OUTPUT_DIR = Path("/tmp/geometry_cnn_transfer")


# ---------------------------------------------------------------------------
# Data loading from pre-processed crops
# ---------------------------------------------------------------------------

def load_crops_metadata(crops_dir: Path) -> list[dict]:
    """Load metadata.json from preprocessed crops directory."""
    meta_path = crops_dir / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_generator(
    metadata: list[dict],
    crops_dir: Path,
    heatmap_size: int = HEATMAP_SIZE,
    augment: bool = True,
    apply_inner_mask: bool = True,
):
    """Return a generator that yields (image, targets) tuples from cached crops."""
    rng = np.random.default_rng(42)
    from embedded_gauge_reading_tinyml.heatmap_utils import HeatmapConfig, generate_center_tip_heatmaps

    heatmap_config = HeatmapConfig(
        heatmap_height=heatmap_size,
        heatmap_width=heatmap_size,
        sigma_pixels=SIGMA_PIXELS,
    )

    def gen():
        for sample in metadata:
            try:
                img_path = crops_dir / sample["image_path"]
                img = Image.open(img_path).convert("RGB")
                img_array = np.asarray(img, dtype=np.float32) / 255.0

                # Data augmentation: random brightness/contrast
                if augment:
                    rng_val = rng.random()
                    if rng_val < 0.3:
                        factor = float(rng.uniform(0.85, 1.15))
                        img_array = np.clip(img_array * factor, 0.0, 1.0)

                center_x_norm = sample["center_x_norm"]
                center_y_norm = sample["center_y_norm"]
                tip_x_norm = sample["tip_x_norm"]
                tip_y_norm = sample["tip_y_norm"]

                center_hm, tip_hm = generate_center_tip_heatmaps(
                    center_x_norm, center_y_norm,
                    tip_x_norm, tip_y_norm,
                    config=heatmap_config,
                )

                conf = np.array([1.0], dtype=np.float32)

                yield img_array, (center_hm.astype(np.float32), tip_hm.astype(np.float32), conf)
            except Exception as e:
                pass

    return gen


def load_arrays(
    metadata: list[dict],
    crops_dir: Path,
    heatmap_size: int = HEATMAP_SIZE,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all data into numpy arrays."""
    from embedded_gauge_reading_tinyml.heatmap_utils import HeatmapConfig, generate_center_tip_heatmaps

    heatmap_config = HeatmapConfig(
        heatmap_height=heatmap_size,
        heatmap_width=heatmap_size,
        sigma_pixels=SIGMA_PIXELS,
    )

    n = len(metadata)
    images = np.zeros((n, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    center_hms = np.zeros((n, heatmap_size, heatmap_size), dtype=np.float32)
    tip_hms = np.zeros((n, heatmap_size, heatmap_size), dtype=np.float32)

    rng = np.random.default_rng(42)

    for i, sample in enumerate(metadata):
        img = Image.open(crops_dir / sample["image_path"]).convert("RGB")
        img_array = np.asarray(img, dtype=np.float32) / 255.0

        if augment:
            if rng.random() < 0.3:
                factor = float(rng.uniform(0.85, 1.15))
                img_array = np.clip(img_array * factor, 0.0, 1.0)

        images[i] = img_array
        c_hm, t_hm = generate_center_tip_heatmaps(
            sample["center_x_norm"], sample["center_y_norm"],
            sample["tip_x_norm"], sample["tip_y_norm"],
            config=heatmap_config,
        )
        center_hms[i] = c_hm.astype(np.float32)
        tip_hms[i] = t_hm.astype(np.float32)

    return images, center_hms, tip_hms


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_two_phase(
    model: keras.Model,
    train_imgs: np.ndarray,
    train_targets: list[np.ndarray],
    val_imgs: np.ndarray,
    val_targets: list[np.ndarray],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    """Two-phase training: frozen backbone then fine-tune."""
    phase1_epochs = min(10, epochs // 4)
    print(f"Phase 1: Frozen backbone for {phase1_epochs} epochs")

    callbacks_phase1 = [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_phase1.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    history1 = model.fit(
        train_imgs, train_targets,
        validation_data=(val_imgs, val_targets),
        epochs=phase1_epochs, batch_size=batch_size,
        callbacks=callbacks_phase1, verbose=1,
    )

    phase2_epochs = epochs - phase1_epochs
    print(f"\nPhase 2: Fine-tuning backbone for {phase2_epochs} epochs")

    backbone = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for sublayer in layer.layers:
                sublayer.trainable = True
            backbone = layer
            break
    if backbone is not None:
        backbone.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr * 3),
        loss=["mse", "mse", "binary_crossentropy"],
        loss_weights=[1.0, 2.0, 0.1],
    )

    callbacks_phase2 = [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_phase2.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
        ),
    ]

    history2 = model.fit(
        train_imgs, train_targets,
        validation_data=(val_imgs, val_targets),
        epochs=phase2_epochs, batch_size=batch_size,
        callbacks=callbacks_phase2, initial_epoch=0, verbose=1,
    )

    _save_history(output_dir, history1, history2)


def _train_single_phase(
    model: keras.Model,
    train_imgs: np.ndarray,
    train_targets: list[np.ndarray],
    val_imgs: np.ndarray,
    val_targets: list[np.ndarray],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    """Single-phase training from scratch (no pretrained backbone).

    Everything is trainable from epoch 1 because there is no transfer
    learning to preserve.  Higher learning rate is used since the backbone
    starts from random weights.
    """
    scratch_lr = lr * 3  # 9e-4 for random-init backbone
    print(f"Single-phase from scratch: {epochs} epochs, lr={scratch_lr}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=scratch_lr),
        loss=["mse", "mse", "binary_crossentropy"],
        loss_weights=[1.0, 2.0, 0.1],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        train_imgs, train_targets,
        validation_data=(val_imgs, val_targets),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1,
    )

    _save_history(output_dir, history)


def _save_history(
    output_dir: Path,
    *histories: keras.callbacks.History,
) -> None:
    """Save training history to JSON."""
    combined: dict[str, list[float]] = {}
    for i, h in enumerate(histories):
        prefix = f"phase_{i+1}_" if len(histories) > 1 else ""
        for k, vals in h.history.items():
            combined[f"{prefix}{k}"] = [float(v) for v in vals]
    (output_dir / "training_history.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    print(f"Saved training history to {output_dir / 'training_history.json'}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    crops_dir: Path,
    output_dir: Path,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 3e-4,
    augment: bool = True,
    pretrained: bool = True,
) -> None:
    """Full training loop with two-phase schedule.

    When ``pretrained=False``, the backbone is initialized randomly (no ImageNet
    weights) and the model is trained in a single phase with everything
    unfrozen from the start.  This avoids BatchNorm statistics mismatch when
    the target domain (gauge crops) is far from ImageNet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-processed crops
    all_metadata = load_crops_metadata(crops_dir)
    train_meta = [m for m in all_metadata if m["split"] == "train"]
    val_meta = [m for m in all_metadata if m["split"] == "val"]
    print(f"Loaded {len(all_metadata)} crops ({len(train_meta)} train, {len(val_meta)} val)")

    # Build model
    model = build_mobilenetv2_geometry_heatmap_v4_112(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        alpha=0.35,
        backbone_frozen=pretrained,  # freeze only if using ImageNet weights
        pretrained=pretrained,
    )

    # Compile: losses as list matching output order [center, tip, confidence]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=["mse", "mse", "binary_crossentropy"],
        loss_weights=[1.0, 2.0, 0.1],
    )

    # Load all data into memory
    print("Loading training data...")
    train_imgs, train_c, train_t = load_arrays(train_meta, crops_dir, augment=augment)
    print(f"  Train: {train_imgs.shape}")
    print("Loading validation data...")
    val_imgs, val_c, val_t = load_arrays(val_meta, crops_dir, augment=False)
    print(f"  Val: {val_imgs.shape}")

    train_targets = [train_c, train_t, np.ones((len(train_meta), 1), dtype=np.float32)]
    val_targets = [val_c, val_t, np.ones((len(val_meta), 1), dtype=np.float32)]

    steps_per_epoch = max(1, len(train_meta) // batch_size)
    validation_steps = max(1, len(val_meta) // batch_size)

    print(f"\nModel parameters: {model.count_params():,}")

    if pretrained:
        _train_two_phase(
            model, train_imgs, train_targets, val_imgs, val_targets,
            output_dir, epochs, batch_size, lr,
        )
    else:
        _train_single_phase(
            model, train_imgs, train_targets, val_imgs, val_targets,
            output_dir, epochs, batch_size, lr,
        )

    # Save final model
    final_path = output_dir / "final.keras"
    model.save(str(final_path))
    print(f"\nSaved final model to {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train geometry heatmap CNN")
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained",
                        action="store_false", default=True,
                        help="Train from scratch (random init, no ImageNet weights)")
    args = parser.parse_args()

    train(
        crops_dir=args.crops_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment=args.augment,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
