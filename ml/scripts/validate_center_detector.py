"""Validate a trained center-detector model and visualise predictions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.training import (
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

matplotlib.use("Agg")

IMAGE_HEIGHT: int = DEFAULT_IMAGE_HEIGHT
IMAGE_WIDTH: int = DEFAULT_IMAGE_WIDTH
BATCH_SIZE: int = DEFAULT_BATCH_SIZE
SEED: int = DEFAULT_SEED
GAUGE_ID: str = DEFAULT_GAUGE_ID
VAL_FRACTION: float = 0.1
TEST_FRACTION: float = 0.1
CROP_PAD_RATIO: float = 0.25
KEYPOINT_HEATMAP_SIZE: int = 28


def _load_test_set() -> list[Any]:
    """Reproduce the same test split used during training."""
    samples = load_dataset()
    spec = load_gauge_specs()[GAUGE_ID]
    examples, _ = _build_training_examples(
        samples,
        spec,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        keypoint_heatmap_size=KEYPOINT_HEATMAP_SIZE,
        strict_labels=False,
        crop_pad_ratio=CROP_PAD_RATIO,
    )

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
    return split.test_examples


def _to_normalized_xy(
    examples: list[Any],
) -> np.ndarray:
    """Convert TrainingExample center_xy (pixel coords) to normalized [0, 1]."""
    return np.array(
        [
            [e.center_xy[0] / IMAGE_WIDTH, e.center_xy[1] / IMAGE_HEIGHT]
            for e in examples
        ],
        dtype=np.float32,
    )


def _compute_mae_pixels(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
) -> dict[str, float]:
    """Compute MAE metrics in both normalized and pixel units."""
    abs_diff = np.abs(pred_norm - target_norm)
    cx_mae_norm = float(np.mean(abs_diff[:, 0]))
    cy_mae_norm = float(np.mean(abs_diff[:, 1]))
    # Euclidean distance in normalized coords
    euclidean_norm = float(np.mean(np.sqrt(np.sum(abs_diff ** 2, axis=-1))))
    # Convert to pixel units
    cx_mae_px = cx_mae_norm * IMAGE_WIDTH
    cy_mae_px = cy_mae_norm * IMAGE_HEIGHT
    euclidean_px = euclidean_norm * math.sqrt(IMAGE_WIDTH ** 2 + IMAGE_HEIGHT ** 2) / math.sqrt(2)
    return {
        "center_x_mae_norm": cx_mae_norm,
        "center_y_mae_norm": cy_mae_norm,
        "euclidean_mae_norm": euclidean_norm,
        "center_x_mae_px": cx_mae_px,
        "center_y_mae_px": cy_mae_px,
        "euclidean_mae_px": euclidean_px,
    }


def _load_image_for_display(
    example: Any,
) -> np.ndarray:
    """Load an image the same way as training, return uint8 for display."""
    image_bytes = tf.io.read_file(example.image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, [None, None, 3])
    # Use the same crop+pad as training
    from embedded_gauge_reading_tinyml.training import _crop_image_with_xyxy
    box = tf.constant(example.crop_box_xyxy, dtype=tf.float32)
    image = _crop_image_with_xyxy(image, box)
    # Use resize_with_pad for display (same as training preprocess)
    image = tf.image.resize_with_pad(image, IMAGE_HEIGHT, IMAGE_WIDTH)
    return tf.cast(image, tf.uint8).numpy()


def _create_montage(
    examples: list[Any],
    pred_xy: np.ndarray,
    target_xy: np.ndarray,
    save_dir: Path,
    *,
    max_samples: int = 64,
) -> None:
    """Overlay predicted (green) and ground-truth (red) centers on test images."""
    n = min(len(examples), max_samples)
    cols = 8
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_flat = axes.flatten() if rows > 1 else (axes if cols > 1 else [axes])

    for i in range(n):
        ax = axes_flat[i]
        img = _load_image_for_display(examples[i])
        ax.imshow(img)

        # Ground truth (red) and predicted (green) in pixel coords
        gt_x = target_xy[i, 0] * IMAGE_WIDTH
        gt_y = target_xy[i, 1] * IMAGE_HEIGHT
        pr_x = pred_xy[i, 0] * IMAGE_WIDTH
        pr_y = pred_xy[i, 1] * IMAGE_HEIGHT

        ax.plot(gt_x, gt_y, "ro", markersize=6, markeredgecolor="white", markeredgewidth=0.5, label="GT")
        ax.plot(pr_x, pr_y, "go", markersize=4, markeredgecolor="white", markeredgewidth=0.5, label="Pred")

        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    save_path = save_dir / "montage.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Montage saved to {save_path}")


def main() -> None:
    """Run center-detector validation."""
    parser = argparse.ArgumentParser(description="Validate a trained center detector.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved .keras model (final or best).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Maximum number of test samples to visualise (default: 64).",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml/tmp") / f"center_validate_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Validating center detector: {model_path}")
    print(f"  Output dir: {out_dir}")
    print("=" * 60)

    # ── 1. Load model ───────────────────────────────────────────────────
    print("\n[1] Loading model...")
    model: keras.Model = keras.models.load_model(model_path)
    print(f"  Model loaded: {model.name}")

    # ── 2. Load test set ────────────────────────────────────────────────
    print("\n[2] Loading test set...")
    test_examples = _load_test_set()
    print(f"  Test samples: {len(test_examples)}")

    # ── 3. Run inference ────────────────────────────────────────────────
    print("\n[3] Running inference...")
    target_xy = _to_normalized_xy(test_examples)

    all_preds: list[np.ndarray] = []
    for start in range(0, len(test_examples), BATCH_SIZE):
        batch = test_examples[start : start + BATCH_SIZE]
        batch_images = []
        batch_boxes = np.array([e.crop_box_xyxy for e in batch], dtype=np.float32)
        for ex, box in zip(batch, batch_boxes):
            img, _ = _load_crop_and_preprocess_image(
                ex.image_path,
                np.float32(0.0),
                box,
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
            )
            batch_images.append(img.numpy())
        batch_array = np.stack(batch_images, axis=0)
        outputs = model.predict(batch_array, verbose=0)
        pred_center = outputs["center_xy"] if isinstance(outputs, dict) else outputs[0]
        all_preds.append(pred_center)

    pred_xy = np.concatenate(all_preds, axis=0)

    # ── 4. Compute metrics ──────────────────────────────────────────────
    print("\n[4] Metrics:")
    metrics = _compute_mae_pixels(pred_xy, target_xy)
    for key, val in metrics.items():
        print(f"  {key}: {val:.6f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 5. Visualise ────────────────────────────────────────────────────
    print("\n[5] Generating visualisation...")
    _create_montage(
        test_examples,
        pred_xy,
        target_xy,
        out_dir,
        max_samples=args.max_samples,
    )

    # Also generate a subset with filename mapping for reference
    results: list[dict[str, Any]] = []
    for i, ex in enumerate(test_examples[:args.max_samples]):
        results.append(
            {
                "image_path": ex.image_path,
                "gt_cx_norm": float(target_xy[i, 0]),
                "gt_cy_norm": float(target_xy[i, 1]),
                "pred_cx_norm": float(pred_xy[i, 0]),
                "pred_cy_norm": float(pred_xy[i, 1]),
                "cx_error_px": abs(float(pred_xy[i, 0] - target_xy[i, 0])) * IMAGE_WIDTH,
                "cy_error_px": abs(float(pred_xy[i, 1] - target_xy[i, 1])) * IMAGE_HEIGHT,
            }
        )
    with open(out_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
