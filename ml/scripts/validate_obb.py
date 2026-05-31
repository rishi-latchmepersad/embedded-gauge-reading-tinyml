"""Validate an OBB model against the labelled test set.

Loads a trained OBB model checkpoint, reproduces the same train/test split
used during training (seed=21, test_fraction=0.15), runs inference on the
test set, and reports per-param and overall MAE/RMSE plus mean IoU.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from embedded_gauge_reading_tinyml.board_crop_compare import (
    load_rgb_image,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs, needle_value
from embedded_gauge_reading_tinyml.labels import summarize_label_sweep
from embedded_gauge_reading_tinyml.models import (
    SpatialSoftArgmax2D,
    GaugeValueFromKeypoints,
    GaugeValueFromNeedleDirection,
    OrderedCornerBox,
    CornerKeypointsToBox,
)
from embedded_gauge_reading_tinyml.presets import (
    DEFAULT_CROP_PAD_RATIO,
    DEFAULT_GAUGE_ID,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_KEYPOINT_HEATMAP_SIZE,
    DEFAULT_SEED,
    DEFAULT_STRICT_LABELS,
    DEFAULT_TEST_FRACTION,
)
from embedded_gauge_reading_tinyml.training import (
    _build_training_examples,
    _compute_crop_box,
    _map_point_to_resized_crop_xy,
    _split_examples,
    TrainConfig,
)

# Resolve paths
ML_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = ML_ROOT / "data" / "raw"
LABELLED_DIR: Path = ML_ROOT / "data" / "labelled"


@dataclass(frozen=True)
class OBBMetrics:
    """Per-parameter and aggregate metrics for OBB predictions."""

    param_names: list[str]
    param_mae: np.ndarray
    param_rmse: np.ndarray
    overall_mae: float
    overall_rmse: float
    mean_iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an OBB model against the labelled test set."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained OBB model checkpoint (.keras).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit visualizations to this many test samples (for quick checks).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible train/test split (default: 21).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Test fraction (default: 0.15).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations. Defaults to tmp/obb_validate_<timestamp>/.",
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default=DEFAULT_GAUGE_ID,
        help="Gauge ID (default: littlegood_home_temp_gauge_c).",
    )
    return parser.parse_args()


def _build_obb_dataset_split(
    gauge_id: str,
    seed: int,
    test_fraction: float,
) -> tuple[
    list, list, list
]:
    """Load the labelled dataset and reproduce the train/test split used in training.

    Returns (all_examples, test_examples, all_samples).
    """
    print(f"[VALIDATE] Loading labelled dataset from {LABELLED_DIR}...")
    samples = load_dataset(labelled_dir=LABELLED_DIR, raw_dir=RAW_DIR)
    print(f"[VALIDATE] Loaded {len(samples)} labelled samples.")

    specs = load_gauge_specs()
    spec = specs[gauge_id]

    label_summary = summarize_label_sweep(samples, spec)
    print(f"[VALIDATE] Label summary: {label_summary}")

    print("[VALIDATE] Building training examples (same as training pipeline)...")
    examples, dropped = _build_training_examples(
        samples,
        spec,
        image_height=DEFAULT_IMAGE_HEIGHT,
        image_width=DEFAULT_IMAGE_WIDTH,
        keypoint_heatmap_size=DEFAULT_KEYPOINT_HEATMAP_SIZE,
        strict_labels=DEFAULT_STRICT_LABELS,
        crop_pad_ratio=DEFAULT_CROP_PAD_RATIO,
    )
    print(f"[VALIDATE] Built {len(examples)} examples ({dropped} dropped out-of-sweep).")

    # Build a minimal TrainConfig just to reuse _split_examples
    config = TrainConfig(
        gauge_id=gauge_id,
        seed=seed,
        test_fraction=test_fraction,
        val_fraction=0.15,
    )
    split = _split_examples(examples, config)
    print(
        f"[VALIDATE] Split: train={len(split.train_examples)}, "
        f"val={len(split.val_examples)}, test={len(split.test_examples)}"
    )

    return examples, split.test_examples, samples, spec


def _decode_obb_params(
    obb: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float, float]:
    """Decode a 6-param OBB vector back into pixel-space box parameters.

    Returns (cx, cy, w, h, angle_rad) where cx, cy are the center in pixels,
    w, h are the box width/height in pixels, and angle_rad is the rotation
    angle in radians.
    """
    cx_norm, cy_norm, w_norm, h_norm, cos2t, sin2t = obb
    # The angle is encoded as cos(2θ), sin(2θ) so θ = 0.5 * atan2(sin2t, cos2t)
    angle_rad = 0.5 * math.atan2(float(sin2t), float(cos2t))
    cx = float(cx_norm) * image_width
    cy = float(cy_norm) * image_height
    w = float(w_norm) * image_width
    h = float(h_norm) * image_height
    return cx, cy, w, h, angle_rad


def _obb_to_corners(
    cx: float, cy: float, w: float, h: float, angle_rad: float
) -> np.ndarray:
    """Convert (cx, cy, w, h, angle_rad) to four corner points (4, 2)."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    half_w = w / 2.0
    half_h = h / 2.0
    corners = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float32,
    )
    # Rotate corners
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    corners = corners @ rot.T
    corners[:, 0] += cx
    corners[:, 1] += cy
    return corners


def _axis_aligned_iou(
    pred_corners: np.ndarray, gt_corners: np.ndarray
) -> float:
    """Compute IoU of axis-aligned bounding boxes from corner points."""
    pred_min = pred_corners.min(axis=0)
    pred_max = pred_corners.max(axis=0)
    gt_min = gt_corners.min(axis=0)
    gt_max = gt_corners.max(axis=0)

    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)
    inter_wh = np.maximum(inter_max - inter_min, 0.0)
    inter_area = inter_wh[0] * inter_wh[1]

    pred_area = (pred_max[0] - pred_min[0]) * (pred_max[1] - pred_min[1])
    gt_area = (gt_max[0] - gt_min[0]) * (gt_max[1] - gt_min[1])
    union_area = pred_area + gt_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def _compute_rotated_iou(
    pred_corners: np.ndarray, gt_corners: np.ndarray
) -> float:
    """Compute rotated IoU using shapely if available, fall back to axis-aligned."""
    try:
        from shapely.geometry import Polygon

        pred_poly = Polygon(pred_corners)
        gt_poly = Polygon(gt_corners)
        if not pred_poly.is_valid or not gt_poly.is_valid:
            return _axis_aligned_iou(pred_corners, gt_corners)
        inter_area = pred_poly.intersection(gt_poly).area
        union_area = pred_poly.union(gt_poly).area
        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)
    except ImportError:
        return _axis_aligned_iou(pred_corners, gt_corners)


def _load_and_preprocess_image(
    image_path: Path,
    crop_box_xyxy: tuple[float, float, float, float],
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
) -> np.ndarray:
    """Load an image, crop it, and resize with pad (same as training pipeline)."""
    image = load_rgb_image(image_path)
    image = resize_with_pad_rgb(image, crop_box_xyxy, image_size=image_height)
    image = image.astype(np.float32) / 255.0
    return image


def _get_model_outputs(
    model: keras.Model,
    image_batch: np.ndarray,
) -> np.ndarray:
    """Run inference and return the obb_params predictions."""
    preds = model.predict(image_batch, verbose=0)
    if isinstance(preds, dict):
        return np.asarray(preds["obb_params"])
    return np.asarray(preds)


def _decode_to_corners(obb_params: np.ndarray) -> np.ndarray:
    """Decode a 6-param OBB vector (normalized) into four corner points."""
    cx_norm, cy_norm, w_norm, h_norm, cos2t, sin2t = obb_params
    angle_rad = 0.5 * math.atan2(float(sin2t), float(cos2t))
    cx = float(cx_norm) * DEFAULT_IMAGE_WIDTH
    cy = float(cy_norm) * DEFAULT_IMAGE_HEIGHT
    w = float(w_norm) * DEFAULT_IMAGE_WIDTH
    h = float(h_norm) * DEFAULT_IMAGE_HEIGHT
    return _obb_to_corners(cx, cy, w, h, angle_rad)


def _compute_metrics(
    preds: np.ndarray, targets: np.ndarray
) -> OBBMetrics:
    """Compute per-param and aggregate MAE, RMSE, and mean IoU."""
    param_names = ["cx_norm", "cy_norm", "w_norm", "h_norm", "cos(2θ)", "sin(2θ)"]

    abs_errors = np.abs(preds - targets)
    sq_errors = (preds - targets) ** 2

    param_mae = np.mean(abs_errors, axis=0)
    param_rmse = np.sqrt(np.mean(sq_errors, axis=0))
    overall_mae = float(np.mean(abs_errors))
    overall_rmse = float(np.sqrt(np.mean(sq_errors)))

    # Compute IoU (rotated via shapely if available, else axis-aligned)
    ious = []
    for i in range(len(preds)):
        pred_corners = _decode_to_corners(preds[i])
        gt_corners = _decode_to_corners(targets[i])
        iou = _compute_rotated_iou(pred_corners, gt_corners)
        ious.append(iou)
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return OBBMetrics(
        param_names=param_names,
        param_mae=param_mae,
        param_rmse=param_rmse,
        overall_mae=overall_mae,
        overall_rmse=overall_rmse,
        mean_iou=mean_iou,
    )


def _save_visualizations(
    test_examples: list,
    preds: np.ndarray,
    targets: np.ndarray,
    model: keras.Model,
    output_dir: Path,
    max_samples: int | None = None,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    image_width: int = DEFAULT_IMAGE_WIDTH,
) -> None:
    """Overlay predicted (green) and ground-truth (red) OBB boxes on images and save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    num_viz = len(test_examples) if max_samples is None else min(max_samples, len(test_examples))

    figs_per_row = 4
    num_rows = int(math.ceil(num_viz / figs_per_row))
    fig, axes = plt.subplots(num_rows, figs_per_row, figsize=(figs_per_row * 5, num_rows * 5))
    axes_flat = axes.flatten() if num_rows > 1 else [axes] if num_rows == 1 else []

    for idx in range(num_viz):
        example = test_examples[idx]
        image_path = Path(example.image_path)
        crop_box = example.crop_box_xyxy

        # Load and preprocess image for display (on the crop)
        image_raw = load_rgb_image(image_path)
        image_cropped = resize_with_pad_rgb(image_raw, crop_box, image_size=image_height)
        image_display = image_cropped.astype(np.float32) / 255.0

        # Get image dimensions from the crop
        crop_w = max(crop_box[2] - crop_box[0], 1.0)
        crop_h = max(crop_box[3] - crop_box[1], 1.0)
        scale = min(image_width / crop_w, image_height / crop_h)

        def _decode_to_crop_pixels(obb_params: np.ndarray) -> np.ndarray:
            """Decode OBB params to corners in the crop image pixel space."""
            cx_norm, cy_norm, w_norm, h_norm, cos2t, sin2t = obb_params
            angle_rad = 0.5 * math.atan2(float(sin2t), float(cos2t))
            cx = float(cx_norm) * image_width
            cy = float(cy_norm) * image_height
            w = float(w_norm) * image_width
            h = float(h_norm) * image_height
            return _obb_to_corners(cx, cy, w, h, angle_rad)

        pred_corners = _decode_to_crop_pixels(preds[idx])
        gt_corners = _decode_to_crop_pixels(targets[idx])

        if idx < len(axes_flat):
            ax = axes_flat[idx]
            ax.imshow(image_display)
            # Ground truth in red
            gt_poly = np.vstack([gt_corners, gt_corners[0:1]])
            ax.plot(gt_poly[:, 0], gt_poly[:, 1], "r-", linewidth=2, label="GT")
            # Prediction in green
            pred_poly = np.vstack([pred_corners, pred_corners[0:1]])
            ax.plot(pred_poly[:, 0], pred_poly[:, 1], "g-", linewidth=2, label="Pred")
            ax.set_title(f"Sample {idx}")
            ax.axis("off")

    # Hide unused axes
    for idx in range(num_viz, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    montage_path = output_dir / "validation_montage.png"
    fig.savefig(montage_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VALIDATE] Montage saved to {montage_path}")

    # Also save individual overlays
    for idx in range(num_viz):
        example = test_examples[idx]
        image_path = Path(example.image_path)
        crop_box = example.crop_box_xyxy

        image_raw = load_rgb_image(image_path)
        image_cropped = resize_with_pad_rgb(image_raw, crop_box, image_size=image_height)
        image_display = image_cropped.astype(np.float32) / 255.0

        def _decode_to_crop_pixels(obb_params):
            cx_norm, cy_norm, w_norm, h_norm, cos2t, sin2t = obb_params
            angle_rad = 0.5 * math.atan2(float(sin2t), float(cos2t))
            cx = float(cx_norm) * image_width
            cy = float(cy_norm) * image_height
            w = float(w_norm) * image_width
            h = float(h_norm) * image_height
            return _obb_to_corners(cx, cy, w, h, angle_rad)

        pred_corners = _decode_to_crop_pixels(preds[idx])
        gt_corners = _decode_to_crop_pixels(targets[idx])

        fig_i, ax_i = plt.subplots(1, 1, figsize=(6, 6))
        ax_i.imshow(image_display)
        gt_poly = np.vstack([gt_corners, gt_corners[0:1]])
        ax_i.plot(gt_poly[:, 0], gt_poly[:, 1], "r-", linewidth=2, label="GT")
        pred_poly = np.vstack([pred_corners, pred_corners[0:1]])
        ax_i.plot(pred_poly[:, 0], pred_poly[:, 1], "g-", linewidth=2, label="Pred")
        ax_i.legend()
        ax_i.axis("off")
        fname = f"sample_{idx:04d}_{image_path.name}.png"
        fig_i.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig_i)

    print(f"[VALIDATE] Individual overlays saved to {output_dir}/")


def _print_summary(metrics: OBBMetrics) -> None:
    """Print a clean summary table of all metrics."""
    print()
    print("=" * 60)
    print("  OBB Validation Summary")
    print("=" * 60)
    print(f"  {'Parameter':<15s} {'MAE':<12s} {'RMSE':<12s}")
    print("  " + "-" * 39)
    for name, mae, rmse in zip(
        metrics.param_names, metrics.param_mae, metrics.param_rmse
    ):
        print(f"  {name:<15s} {mae:<12.6f} {rmse:<12.6f}")
    print("  " + "-" * 39)
    print(f"  {'Overall':<15s} {metrics.overall_mae:<12.6f} {metrics.overall_rmse:<12.6f}")
    print(f"  {'Mean IoU':<15s} {metrics.mean_iou:<12.6f}")
    print("=" * 60)
    print()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("tmp") / f"obb_validate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    print(f"[VALIDATE] Loading model from {model_path}...")
    model = keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "preprocess_input": keras.applications.mobilenet_v2.preprocess_input,
            "SpatialSoftArgmax2D": SpatialSoftArgmax2D,
            "GaugeValueFromKeypoints": GaugeValueFromKeypoints,
            "GaugeValueFromNeedleDirection": GaugeValueFromNeedleDirection,
            "OrderedCornerBox": OrderedCornerBox,
            "CornerKeypointsToBox": CornerKeypointsToBox,
        },
    )
    print(f"[VALIDATE] Model loaded. Input shape: {model.input_shape}")

    # Build the dataset split (reproducing training split)
    examples, test_examples, samples, spec = _build_obb_dataset_split(
        gauge_id=args.gauge_id,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )

    # Collect ground-truth obb_params and images for the test set
    print(f"[VALIDATE] Running inference on {len(test_examples)} test samples...")

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for idx, example in enumerate(test_examples):
        image_path = Path(example.image_path)
        crop_box = example.crop_box_xyxy

        # Get ground truth obb_params
        gt_obb = example.obb_params
        if gt_obb is None:
            continue

        # Load and preprocess image
        image = _load_and_preprocess_image(
            image_path, crop_box, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH
        )
        image_batch = np.expand_dims(image, axis=0).astype(np.float32)

        # Run inference
        pred = _get_model_outputs(model, image_batch)

        all_preds.append(pred[0])
        all_targets.append(gt_obb)

        if (idx + 1) % 50 == 0:
            print(f"[VALIDATE] Processed {idx + 1}/{len(test_examples)}...")

    preds_array = np.array(all_preds, dtype=np.float32)
    targets_array = np.array(all_targets, dtype=np.float32)
    print(f"[VALIDATE] Inference complete: {len(preds_array)} samples.")

    # Compute metrics
    metrics = _compute_metrics(preds_array, targets_array)
    _print_summary(metrics)

    # Save visualizations
    print("[VALIDATE] Generating visualizations...")
    _save_visualizations(
        test_examples[:len(preds_array)],
        preds_array,
        targets_array,
        model,
        output_dir,
        max_samples=args.max_samples,
    )
    print(f"[VALIDATE] Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
