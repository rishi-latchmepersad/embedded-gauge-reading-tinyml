#!/usr/bin/env python3
"""
Validate polar voting spoke detector against ground-truth needle labels.

Loads CVAT-labelled gauge images, applies the fixed training crop,
runs the polar spoke-vote needle detector with configurable center
hypotheses, and reports angular error statistics (MAE, RMSE, % ≤ 2°/5°/10°).

Usage:
    python ml/scripts/validate_polar_voting.py --max-samples 50 \\
        --center-source ground_truth
    python ml/scripts/validate_polar_voting.py --center-source crop_center
    python ml/scripts/validate_polar_voting.py --center-source bright_centroid
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from embedded_gauge_reading_tinyml.board_crop_compare import (
    load_rgb_image,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import (
    GaugeSpec,
    load_gauge_specs,
)
from embedded_gauge_reading_tinyml.hybrid_localizer import (
    estimate_bright_centroid_on_crop,
    estimate_dial_radius,
    polar_spoke_vote as _polar_spoke_vote,
    rgb_to_luma,
    smooth_and_find_peak as _smooth_and_find_peak,
)
from embedded_gauge_reading_tinyml.presets import (
    DEFAULT_CROP_PAD_RATIO,
    DEFAULT_GAUGE_ID,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
)
from embedded_gauge_reading_tinyml.training import (
    TrainConfig,
    _build_training_examples,
    _split_examples,
)

# ---------------------------------------------------------------------------
# Polar voting wrapper
# ---------------------------------------------------------------------------


def _detect_needle_polar(
    image_224: NDArray[np.uint8],
    *,
    center_x: float,
    center_y: float,
    dial_radius: float,
    gauge_spec: GaugeSpec,
) -> float | None:
    """Run polar spoke-vote on a 224×224 crop and return the needle angle in degrees.

    Steps:
      1. Compute BT.601 luma from RGB.
      2. Compute Sobel edge magnitude via central differences.
      3. For each pixel in the annulus (30%–70% of dial radius):
           - Compute radial direction (from center to pixel).
           - Compute tangential alignment (cross product of gradient × radial).
           - Vote = edge_mag × tangential_alignment × darkness (255 − luma).
      4. Accumulate into 360 angle bins.
      5. Smooth with 3-bin boxcar and find the peak.
      6. Refine sub-bin via weighted average of the 3-neighbourhood.
      7. Gate on peak/mean confidence ≥ 1.25 and peak value ≥ 75.
      8. Validate that the angle falls inside the gauge sweep (with 6° margin).

    Returns:
        Best needle angle in degrees [0, 360), or None if no peak passes gates.
    """
    luma: NDArray[np.float32] = rgb_to_luma(image_224)
    votes: np.ndarray = _polar_spoke_vote(luma, center_x, center_y, dial_radius)
    angle_deg: float
    peak_val: float
    mean_val: float
    angle_deg, peak_val, mean_val = _smooth_and_find_peak(votes)

    # Confidence gate: peak-to-mean ratio (matches embedded C baseline).
    confidence: float = peak_val / max(mean_val, 1e-6)
    if confidence < 1.25 or peak_val < 75.0:
        return None

    # Sweep validation: reject angles outside the calibrated gauge arc.
    angle_rad: float = math.radians(angle_deg)
    shifted: float = (angle_rad - gauge_spec.min_angle_rad) % (2.0 * math.pi)
    if shifted > gauge_spec.sweep_rad + math.radians(6.0):
        return None

    return angle_deg


# ---------------------------------------------------------------------------
# Angular error helper
# ---------------------------------------------------------------------------


def _angular_error_deg(gt_deg: float, pred_deg: float) -> float:
    """Shortest absolute angular difference in [0, 180]."""
    diff: float = abs(pred_deg - gt_deg) % 360.0
    return min(diff, 360.0 - diff)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(errors_deg: list[float], n_total: int) -> dict[str, float]:
    """Return MAE, RMSE, and percentage within 2°/5°/10°."""
    if not errors_deg:
        return {"mae": 0.0, "rmse": 0.0, "pct_2": 0.0, "pct_5": 0.0, "pct_10": 0.0}

    arr: np.ndarray = np.array(errors_deg, dtype=np.float64)
    mae: float = float(np.mean(arr))
    rmse: float = float(np.sqrt(np.mean(arr ** 2)))
    pct_2: float = 100.0 * float(np.sum(arr <= 2.0)) / n_total
    pct_5: float = 100.0 * float(np.sum(arr <= 5.0)) / n_total
    pct_10: float = 100.0 * float(np.sum(arr <= 10.0)) / n_total
    return {"mae": mae, "rmse": rmse, "pct_2": pct_2, "pct_5": pct_5, "pct_10": pct_10}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _render_sample_plot(
    image_224: NDArray[np.uint8],
    *,
    center_x: float,
    center_y: float,
    gt_angle_deg: float,
    pred_angle_deg: float | None,
    votes: np.ndarray,
    sample_name: str,
    save_path: Path,
) -> None:
    """Save a two-panel figure: (image + needle overlays) and (vote histogram)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: image with center and needles ---
    ax_img.imshow(image_224)
    ax_img.plot(center_x, center_y, "yo", markersize=6, label="Center")

    # Ground truth needle (red, solid)
    gt_dx: float = 45.0 * math.cos(math.radians(gt_angle_deg))
    gt_dy: float = 45.0 * math.sin(math.radians(gt_angle_deg))
    ax_img.arrow(
        center_x, center_y, gt_dx, gt_dy,
        head_width=8, head_length=6, fc="red", ec="red", alpha=0.8,
        label=f"GT  {gt_angle_deg:.1f}°",
    )

    # Predicted needle (lime, dashed)
    if pred_angle_deg is not None:
        pred_dx: float = 45.0 * math.cos(math.radians(pred_angle_deg))
        pred_dy: float = 45.0 * math.sin(math.radians(pred_angle_deg))
        ax_img.arrow(
            center_x, center_y, pred_dx, pred_dy,
            head_width=8, head_length=6, fc="lime", ec="lime", alpha=0.8,
            linestyle="--",
            label=f"Pred {pred_angle_deg:.1f}°",
        )

    ax_img.legend(fontsize=8, loc="lower right")
    ax_img.set_title(sample_name, fontsize=9)
    ax_img.axis("off")

    # --- Right panel: polar vote histogram ---
    angles: np.ndarray = np.linspace(0, 360, len(votes), endpoint=False)
    ax_hist.bar(angles, votes, width=1.0, color="steelblue", alpha=0.7)
    ax_hist.axvline(gt_angle_deg, color="red", linewidth=2, label=f"GT {gt_angle_deg:.1f}°")
    if pred_angle_deg is not None:
        ax_hist.axvline(
            pred_angle_deg, color="lime", linewidth=2, linestyle="--",
            label=f"Pred {pred_angle_deg:.1f}°",
        )
    ax_hist.set_xlabel("Angle (deg)")
    ax_hist.set_ylabel("Vote")
    ax_hist.set_title("Polar Vote Histogram (360 bins, 3-bin boxcar)")
    ax_hist.legend(fontsize=8)

    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate polar voting spoke detector against ground-truth labels.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit test samples (default: all).",
    )
    parser.add_argument(
        "--center-source",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "crop_center", "bright_centroid"],
        help="Center hypothesis for polar voting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: tmp/polar_validate_<timestamp>).",
    )
    args = parser.parse_args()

    # Constants matching the training pipeline.
    gauge_id: str = DEFAULT_GAUGE_ID
    image_height: int = DEFAULT_IMAGE_HEIGHT   # 224
    image_width: int = DEFAULT_IMAGE_WIDTH     # 224
    crop_pad_ratio: float = DEFAULT_CROP_PAD_RATIO  # 0.25
    val_fraction: float = DEFAULT_VAL_FRACTION      # 0.15
    test_fraction: float = DEFAULT_TEST_FRACTION    # 0.15

    # ------------------------------------------------------------------
    # 1. Load gauge calibration parameters.
    # ------------------------------------------------------------------
    specs: dict[str, GaugeSpec] = load_gauge_specs()
    spec: GaugeSpec = specs[gauge_id]
    print(f"Gauge: {gauge_id}  sweep=[{spec.min_value:.0f}, {spec.max_value:.0f}] {spec.units}")

    # ------------------------------------------------------------------
    # 2. Load labelled dataset and build training examples.
    # ------------------------------------------------------------------
    print("Loading labelled dataset...")
    samples = load_dataset()
    print(f"Loaded {len(samples)} labelled samples.")

    config = TrainConfig(
        gauge_id=gauge_id,
        image_height=image_height,
        image_width=image_width,
        crop_pad_ratio=crop_pad_ratio,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    examples, dropped = _build_training_examples(
        samples,
        spec,
        image_height=image_height,
        image_width=image_width,
        keypoint_heatmap_size=28,
        strict_labels=False,
        crop_pad_ratio=crop_pad_ratio,
    )
    print(f"Built {len(examples)} training examples ({dropped} out-of-sweep dropped).")

    split = _split_examples(examples, config)
    test_examples = split.test_examples
    print(f"Test examples: {len(test_examples)}")

    if args.max_samples is not None:
        test_examples = test_examples[: args.max_samples]
        print(f"Limited to {len(test_examples)} test samples.")

    if not test_examples:
        print("No test examples to evaluate.")
        return

    # ------------------------------------------------------------------
    # 3. Set up output directory.
    # ------------------------------------------------------------------
    timestamp: str = time.strftime("%Y%m%d_%H%M%S")
    output_dir: Path = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"tmp/polar_validate_{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir: Path = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    print(f"Output: {output_dir.resolve()}")

    # ------------------------------------------------------------------
    # 4. Estimate dial radius for the 224×224 crop.
    # ------------------------------------------------------------------
    dial_radius: float = estimate_dial_radius(image_height)
    print(f"Dial radius estimate: {dial_radius:.1f} px")

    # ------------------------------------------------------------------
    # 5. Run evaluation over test samples.
    # ------------------------------------------------------------------
    errors_deg: list[float] = []
    sample_results: list[tuple[str, float, float | None, float | None]] = []

    for idx, example in enumerate(test_examples):
        sample_name: str = Path(example.image_path).name

        # 5a. Load the full-res image and apply the same crop+pad as training.
        image_full: np.ndarray = load_rgb_image(Path(example.image_path))
        image_224: np.ndarray = resize_with_pad_rgb(
            image_full,
            example.crop_box_xyxy,
            image_size=image_height,
        )

        # 5b. Ground truth angle (from the needle unit vector).
        gt_angle: float = math.atan2(
            example.needle_unit_xy[1], example.needle_unit_xy[0],
        )
        gt_angle_deg: float = math.degrees(gt_angle) % 360.0

        # 5c. Choose center hypothesis for polar voting.
        if args.center_source == "ground_truth":
            cx, cy = example.center_xy
        elif args.center_source == "crop_center":
            cx, cy = image_width / 2.0, image_height / 2.0
        elif args.center_source == "bright_centroid":
            bcx, bcy, detected = estimate_bright_centroid_on_crop(image_224)
            cx, cy = (bcx, bcy) if detected else example.center_xy

        # 5d. Run polar voting.
        pred_angle_deg: float | None = _detect_needle_polar(
            image_224,
            center_x=cx,
            center_y=cy,
            dial_radius=dial_radius,
            gauge_spec=spec,
        )

        if pred_angle_deg is not None:
            error: float = _angular_error_deg(gt_angle_deg, pred_angle_deg)
            errors_deg.append(error)
            sample_results.append((sample_name, gt_angle_deg, pred_angle_deg, error))
        else:
            sample_results.append((sample_name, gt_angle_deg, None, None))

    # ------------------------------------------------------------------
    # 6. Compute and print metrics.
    # ------------------------------------------------------------------
    n_total: int = len(sample_results)
    n_valid: int = len(errors_deg)
    n_fail: int = n_total - n_valid
    metrics: dict[str, float] = _compute_metrics(errors_deg, n_total)

    print()
    print("=" * 62)
    print(f"  Polar Voting Validation — center_source = {args.center_source}")
    print("=" * 62)
    print(f"  Samples:     {n_total} total, {n_valid} valid, {n_fail} failed")
    print(f"  MAE:         {metrics['mae']:.3f}°")
    print(f"  RMSE:        {metrics['rmse']:.3f}°")
    print(f"  ≤ 2°:        {metrics['pct_2']:.1f}%")
    print(f"  ≤ 5°:        {metrics['pct_5']:.1f}%")
    print(f"  ≤ 10°:       {metrics['pct_10']:.1f}%")
    print("=" * 62)

    # On failure, dump which samples failed so the user can inspect them.
    if n_fail > 0:
        print()
        print(f"  Failures ({n_fail}):")
        for name, gt, pred, err in sample_results:
            if pred is None:
                print(f"    ✗ {name}  (GT={gt:.1f}°)")

    # ------------------------------------------------------------------
    # 7. Generate per-sample visualizations.
    # ------------------------------------------------------------------
    print()
    print(f"  Generating {n_total} plots...")

    for idx, example in enumerate(test_examples):
        sample_name = Path(example.image_path).name
        gt_result, pred_result, _ = next(
            (gt, pred, err)
            for name, gt, pred, err in sample_results
            if name == sample_name
        )

        # Reload image for the plot.
        image_full = load_rgb_image(Path(example.image_path))
        image_224 = resize_with_pad_rgb(
            image_full,
            example.crop_box_xyxy,
            image_size=image_height,
        )

        # Recompute center for consistency.
        if args.center_source == "ground_truth":
            cx, cy = example.center_xy
        elif args.center_source == "crop_center":
            cx, cy = image_width / 2.0, image_height / 2.0
        elif args.center_source == "bright_centroid":
            bcx, bcy, detected = estimate_bright_centroid_on_crop(image_224)
            cx, cy = (bcx, bcy) if detected else example.center_xy

        luma = rgb_to_luma(image_224)
        votes = _polar_spoke_vote(luma, cx, cy, dial_radius)

        fname: str = f"sample_{idx:04d}_{Path(example.image_path).stem}.png"
        _render_sample_plot(
            image_224,
            center_x=cx,
            center_y=cy,
            gt_angle_deg=gt_result,
            pred_angle_deg=pred_result,
            votes=votes,
            sample_name=sample_name,
            save_path=plots_dir / fname,
        )

    print(f"  Plots saved to {plots_dir}/")

    # ------------------------------------------------------------------
    # 8. Accuracy check and improvement suggestions.
    # ------------------------------------------------------------------
    print()
    target_mae: float = 3.0
    if n_valid > 0 and metrics["mae"] > target_mae:
        print("⚠  WARNING: MAE exceeds 3° target.")
        print()
        print("─── Suggested improvements ───")
        print("1. Tighter annulus (35%–65%) to reduce dial-marking clutter.")
        print("2. HSV saturation weighting to suppress colorful tick marks.")
        print("3. CLAHE preprocessing for more robust edges on low-contrast images.")
        print("4. Shaft-weighting (Gaussian centered on mid-shaft) to reduce hub/rim bias.")
        print("5. Increase bin count from 360→720 for finer angular resolution.")
        print("6. Raise Sobel threshold (currently 8) to filter noise.")
        print("7. Multi-scale Sobel (apertures 3, 5, 7) for scale-robust edges.")
        print("8. Tune confidence gates for this specific dataset.")
        print("9. Multi-hypothesis center search when single center is uncertain.")
        print("10. Dark-contrast local neighborhood weighting (baseline_classical_cv style).")
    elif n_valid > 0:
        print("✓  MAE is within the 3° target. Good baseline performance.")
    else:
        print("✗  No valid detections. Check image quality and center hypotheses.")


if __name__ == "__main__":
    main()
