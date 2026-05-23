#!/usr/bin/env python3
"""Analyze why the center branch underperforms the tip branch on the tiny set."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import (
    heatmap_index_to_crop_pixel,
    load_clean_geometry_examples,
    load_identity_crop,
    make_target_heatmaps,
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.heatmap_utils import argmax_2d, softargmax_2d


@dataclass(frozen=True)
class BranchSample:
    """One tiny-overfit sample with ground-truth and predicted heatmaps."""

    image_path: str
    split: str
    temperature_c: float
    crop_image: np.ndarray
    metadata: dict[str, float]
    target_center_heatmap: np.ndarray
    target_tip_heatmap: np.ndarray
    predicted_center_heatmap: np.ndarray
    predicted_tip_heatmap: np.ndarray
    confidence: float


def _patch_dense_deserialization() -> Callable[[], None]:
    """Patch Keras Dense deserialization so the legacy v1 model can load."""

    original_from_config = keras.layers.Dense.from_config

    def _patched_from_config(cls: type[keras.layers.Dense], config: dict[str, Any]) -> keras.layers.Dense:
        """Strip the legacy quantization field before recreating Dense layers."""

        cleaned_config = dict(config)
        cleaned_config.pop("quantization_config", None)
        return original_from_config(cleaned_config)

    keras.layers.Dense.from_config = classmethod(_patched_from_config)

    def _restore() -> None:
        """Restore the original Dense deserialization hook."""

        keras.layers.Dense.from_config = original_from_config

    return _restore


def _load_legacy_model(model_path: Path) -> keras.Model:
    """Load a saved Keras model produced by the earlier heatmap run."""

    restore = _patch_dense_deserialization()
    try:
        return keras.models.load_model(model_path, compile=False)
    finally:
        restore()


def _load_branch_samples(
    *,
    manifest_path: Path,
    base_path: Path,
    heatmap_size: int,
    sigma_pixels: float,
    limit: int,
) -> list[BranchSample]:
    """Load the same deterministic 8-image set used by the tiny overfit gate."""

    examples = load_clean_geometry_examples(manifest_path)
    selected_examples = select_examples_from_split(examples, split="train", limit=limit)
    if len(selected_examples) < limit:
        raise RuntimeError(f"Only found {len(selected_examples)} clean train rows; need {limit}.")

    samples: list[BranchSample] = []
    for example in selected_examples:
        crop_image, metadata, _ = load_identity_crop(example, base_path)
        center_heatmap, tip_heatmap = make_target_heatmaps(
            center_x_norm=metadata["center_x_norm"],
            center_y_norm=metadata["center_y_norm"],
            tip_x_norm=metadata["tip_x_norm"],
            tip_y_norm=metadata["tip_y_norm"],
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
        )
        samples.append(
            BranchSample(
                image_path=example.image_path,
                split=example.split,
                temperature_c=example.temperature_c,
                crop_image=crop_image,
                metadata=metadata,
                target_center_heatmap=center_heatmap,
                target_tip_heatmap=tip_heatmap,
                predicted_center_heatmap=np.zeros_like(center_heatmap),
                predicted_tip_heatmap=np.zeros_like(tip_heatmap),
                confidence=0.0,
            )
        )

    return samples


def _decode_prediction(
    sample: BranchSample,
    prediction: list[np.ndarray],
    *,
    heatmap_size: int,
) -> BranchSample:
    """Attach predicted heatmaps and confidence to a sample."""

    return BranchSample(
        image_path=sample.image_path,
        split=sample.split,
        temperature_c=sample.temperature_c,
        crop_image=sample.crop_image,
        metadata=sample.metadata,
        target_center_heatmap=sample.target_center_heatmap,
        target_tip_heatmap=sample.target_tip_heatmap,
        predicted_center_heatmap=np.squeeze(np.asarray(prediction[0][0], dtype=np.float32)),
        predicted_tip_heatmap=np.squeeze(np.asarray(prediction[1][0], dtype=np.float32)),
        confidence=float(np.ravel(prediction[2])[0]),
    )


def _point_from_heatmap(
    heatmap: np.ndarray,
    *,
    heatmap_size: int,
    method: str,
) -> tuple[float, float]:
    """Convert a heatmap to crop-space coordinates."""

    if method == "argmax":
        row, col = argmax_2d(heatmap)
    elif method == "softargmax":
        row, col = softargmax_2d(heatmap)
    else:
        raise ValueError(f"Unknown heatmap decode method: {method}")

    return (
        heatmap_index_to_crop_pixel(col, heatmap_size=heatmap_size),
        heatmap_index_to_crop_pixel(row, heatmap_size=heatmap_size),
    )


def _sample_metrics(sample: BranchSample, *, heatmap_size: int) -> dict[str, Any]:
    """Compute per-sample branch diagnostics for the report and overlays."""

    true_center_argmax_x, true_center_argmax_y = _point_from_heatmap(
        sample.target_center_heatmap,
        heatmap_size=heatmap_size,
        method="argmax",
    )
    true_tip_argmax_x, true_tip_argmax_y = _point_from_heatmap(
        sample.target_tip_heatmap,
        heatmap_size=heatmap_size,
        method="argmax",
    )
    true_center_soft_x, true_center_soft_y = _point_from_heatmap(
        sample.target_center_heatmap,
        heatmap_size=heatmap_size,
        method="softargmax",
    )
    true_tip_soft_x, true_tip_soft_y = _point_from_heatmap(
        sample.target_tip_heatmap,
        heatmap_size=heatmap_size,
        method="softargmax",
    )

    pred_center_argmax_x, pred_center_argmax_y = _point_from_heatmap(
        sample.predicted_center_heatmap,
        heatmap_size=heatmap_size,
        method="argmax",
    )
    pred_tip_argmax_x, pred_tip_argmax_y = _point_from_heatmap(
        sample.predicted_tip_heatmap,
        heatmap_size=heatmap_size,
        method="argmax",
    )
    pred_center_soft_x, pred_center_soft_y = _point_from_heatmap(
        sample.predicted_center_heatmap,
        heatmap_size=heatmap_size,
        method="softargmax",
    )
    pred_tip_soft_x, pred_tip_soft_y = _point_from_heatmap(
        sample.predicted_tip_heatmap,
        heatmap_size=heatmap_size,
        method="softargmax",
    )

    center_argmax_error = math.hypot(pred_center_argmax_x - sample.metadata["center_x_224"], pred_center_argmax_y - sample.metadata["center_y_224"])
    tip_argmax_error = math.hypot(pred_tip_argmax_x - sample.metadata["tip_x_224"], pred_tip_argmax_y - sample.metadata["tip_y_224"])
    center_soft_error = math.hypot(pred_center_soft_x - sample.metadata["center_x_224"], pred_center_soft_y - sample.metadata["center_y_224"])
    tip_soft_error = math.hypot(pred_tip_soft_x - sample.metadata["tip_x_224"], pred_tip_soft_y - sample.metadata["tip_y_224"])

    center_target_distance = math.hypot(sample.metadata["center_x_224"] - 112.0, sample.metadata["center_y_224"] - 112.0)
    tip_target_distance = math.hypot(sample.metadata["tip_x_224"] - 112.0, sample.metadata["tip_y_224"] - 112.0)

    predicted_angle = angle_degrees_from_center_to_tip(
        pred_center_soft_x,
        pred_center_soft_y,
        pred_tip_soft_x,
        pred_tip_soft_y,
    )
    true_angle = angle_degrees_from_center_to_tip(
        sample.metadata["center_x_224"],
        sample.metadata["center_y_224"],
        sample.metadata["tip_x_224"],
        sample.metadata["tip_y_224"],
    )
    predicted_temp = celsius_from_inner_dial_angle_degrees(predicted_angle)

    return {
        "image_path": sample.image_path,
        "split": sample.split,
        "temperature_c": sample.temperature_c,
        "confidence": sample.confidence,
        "true_angle_degrees": true_angle,
        "predicted_angle_degrees": predicted_angle,
        "angle_error_degrees": abs(circular_angle_error_degrees(predicted_angle, true_angle)),
        "predicted_temperature_c": predicted_temp,
        "absolute_error_c": abs(predicted_temp - sample.temperature_c),
        "target_center_heatmap_min": float(np.min(sample.target_center_heatmap)),
        "target_center_heatmap_max": float(np.max(sample.target_center_heatmap)),
        "target_center_heatmap_mean": float(np.mean(sample.target_center_heatmap)),
        "target_center_heatmap_std": float(np.std(sample.target_center_heatmap)),
        "target_tip_heatmap_min": float(np.min(sample.target_tip_heatmap)),
        "target_tip_heatmap_max": float(np.max(sample.target_tip_heatmap)),
        "target_tip_heatmap_mean": float(np.mean(sample.target_tip_heatmap)),
        "target_tip_heatmap_std": float(np.std(sample.target_tip_heatmap)),
        "pred_center_peak": float(np.max(sample.predicted_center_heatmap)),
        "pred_tip_peak": float(np.max(sample.predicted_tip_heatmap)),
        "pred_center_heatmap_std": float(np.std(sample.predicted_center_heatmap)),
        "pred_tip_heatmap_std": float(np.std(sample.predicted_tip_heatmap)),
        "center_argmax_error": center_argmax_error,
        "tip_argmax_error": tip_argmax_error,
        "center_softargmax_error": center_soft_error,
        "tip_softargmax_error": tip_soft_error,
        "center_target_distance_from_crop_center": center_target_distance,
        "tip_target_distance_from_crop_center": tip_target_distance,
        "true_center_x_224": sample.metadata["center_x_224"],
        "true_center_y_224": sample.metadata["center_y_224"],
        "true_tip_x_224": sample.metadata["tip_x_224"],
        "true_tip_y_224": sample.metadata["tip_y_224"],
        "pred_center_x_224_argmax": pred_center_argmax_x,
        "pred_center_y_224_argmax": pred_center_argmax_y,
        "pred_tip_x_224_argmax": pred_tip_argmax_x,
        "pred_tip_y_224_argmax": pred_tip_argmax_y,
        "pred_center_x_224_softargmax": pred_center_soft_x,
        "pred_center_y_224_softargmax": pred_center_soft_y,
        "pred_tip_x_224_softargmax": pred_tip_soft_x,
        "pred_tip_y_224_softargmax": pred_tip_soft_y,
        "true_center_x_224_argmax": true_center_argmax_x,
        "true_center_y_224_argmax": true_center_argmax_y,
        "true_tip_x_224_argmax": true_tip_argmax_x,
        "true_tip_y_224_argmax": true_tip_argmax_y,
        "true_center_x_224_softargmax": true_center_soft_x,
        "true_center_y_224_softargmax": true_center_soft_y,
        "true_tip_x_224_softargmax": true_tip_soft_x,
        "true_tip_y_224_softargmax": true_tip_soft_y,
        "target_center_heatmap": sample.target_center_heatmap,
        "target_tip_heatmap": sample.target_tip_heatmap,
        "pred_center_heatmap": sample.predicted_center_heatmap,
        "pred_tip_heatmap": sample.predicted_tip_heatmap,
        "crop_image": sample.crop_image,
    }


def _plot_heatmap_panel(
    ax: plt.Axes,
    heatmap: np.ndarray,
    *,
    expected_x: float,
    expected_y: float,
    output_title: str,
    heatmap_size: int,
) -> None:
    """Render a heatmap with expected, argmax, and softargmax markers."""

    row_argmax, col_argmax = argmax_2d(heatmap)
    row_soft, col_soft = softargmax_2d(heatmap)
    ax.imshow(heatmap, cmap="magma", origin="upper")
    ax.scatter([expected_x], [expected_y], c="white", s=45, marker="o", edgecolors="black")
    ax.scatter([col_argmax], [row_argmax], c="cyan", s=55, marker="x", linewidths=2.0)
    ax.scatter([col_soft], [row_soft], c="yellow", s=55, marker="d", edgecolors="black")
    ax.set_xlim(-0.5, heatmap_size - 0.5)
    ax.set_ylim(heatmap_size - 0.5, -0.5)
    ax.set_xlabel("x / col")
    ax.set_ylabel("y / row")
    ax.set_title(f"{output_title}\nmax={float(np.max(heatmap)):.4f}")


def _save_branch_overlay(
    sample: BranchSample,
    *,
    metrics: dict[str, Any],
    output_path: Path,
    heatmap_size: int,
) -> None:
    """Save a single sample overlay with the true and predicted branches."""

    fig = plt.figure(figsize=(17, 10), dpi=150)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 1.0), height_ratios=(1.0, 1.0))

    ax_crop = fig.add_subplot(grid[:, 0])
    ax_center_target = fig.add_subplot(grid[0, 1])
    ax_center_pred = fig.add_subplot(grid[0, 2])
    ax_tip_target = fig.add_subplot(grid[1, 1])
    ax_tip_pred = fig.add_subplot(grid[1, 2])

    ax_crop.imshow(sample.crop_image)
    ax_crop.scatter(
        [sample.metadata["center_x_224"], metrics["pred_center_x_224_softargmax"]],
        [sample.metadata["center_y_224"], metrics["pred_center_y_224_softargmax"]],
        c=["lime", "cyan"],
        s=70,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        label="center",
    )
    ax_crop.scatter(
        [sample.metadata["tip_x_224"], metrics["pred_tip_x_224_softargmax"]],
        [sample.metadata["tip_y_224"], metrics["pred_tip_y_224_softargmax"]],
        c=["red", "yellow"],
        s=70,
        marker="x",
        linewidths=2.0,
        label="tip",
    )
    ax_crop.plot(
        [sample.metadata["center_x_224"], sample.metadata["tip_x_224"]],
        [sample.metadata["center_y_224"], sample.metadata["tip_y_224"]],
        color="white",
        linewidth=2.0,
        alpha=0.8,
        label="true needle",
    )
    ax_crop.plot(
        [metrics["pred_center_x_224_softargmax"], metrics["pred_tip_x_224_softargmax"]],
        [metrics["pred_center_y_224_softargmax"], metrics["pred_tip_y_224_softargmax"]],
        color="deepskyblue",
        linewidth=2.0,
        alpha=0.8,
        label="pred needle",
    )
    ax_crop.set_title("Crop overlay")
    ax_crop.set_axis_off()
    ax_crop.legend(loc="lower right", fontsize=8, framealpha=0.9)

    _plot_heatmap_panel(
        ax_center_target,
        sample.target_center_heatmap,
        expected_x=sample.metadata["center_x_norm"] * (heatmap_size - 1),
        expected_y=sample.metadata["center_y_norm"] * (heatmap_size - 1),
        output_title="Target center heatmap",
        heatmap_size=heatmap_size,
    )
    _plot_heatmap_panel(
        ax_center_pred,
        sample.predicted_center_heatmap,
        expected_x=sample.metadata["center_x_norm"] * (heatmap_size - 1),
        expected_y=sample.metadata["center_y_norm"] * (heatmap_size - 1),
        output_title="Predicted center heatmap",
        heatmap_size=heatmap_size,
    )
    _plot_heatmap_panel(
        ax_tip_target,
        sample.target_tip_heatmap,
        expected_x=sample.metadata["tip_x_norm"] * (heatmap_size - 1),
        expected_y=sample.metadata["tip_y_norm"] * (heatmap_size - 1),
        output_title="Target tip heatmap",
        heatmap_size=heatmap_size,
    )
    _plot_heatmap_panel(
        ax_tip_pred,
        sample.predicted_tip_heatmap,
        expected_x=sample.metadata["tip_x_norm"] * (heatmap_size - 1),
        expected_y=sample.metadata["tip_y_norm"] * (heatmap_size - 1),
        output_title="Predicted tip heatmap",
        heatmap_size=heatmap_size,
    )

    summary_lines = [
        f"true temp: {sample.temperature_c:.2f} C",
        f"pred temp: {metrics['predicted_temperature_c']:.2f} C",
        f"abs temp err: {metrics['absolute_error_c']:.2f} C",
        f"center err: {metrics['center_softargmax_error']:.2f} px",
        f"tip err: {metrics['tip_softargmax_error']:.2f} px",
        f"center peak: {metrics['pred_center_peak']:.4f}",
        f"tip peak: {metrics['pred_tip_peak']:.4f}",
    ]
    fig.text(0.02, 0.01, "\n".join(summary_lines), family="monospace", fontsize=10, va="bottom")
    fig.suptitle(Path(sample.image_path).name, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_predictions_csv(predictions: list[dict[str, Any]], output_path: Path) -> None:
    """Persist the per-sample decode diagnostics."""

    fieldnames = [
        key
        for key in predictions[0].keys()
        if key not in {"target_center_heatmap", "target_tip_heatmap", "pred_center_heatmap", "pred_tip_heatmap", "crop_image"}
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def _write_report(samples: list[dict[str, Any]], output_path: Path, *, model_path: Path, heatmap_size: int) -> None:
    """Summarize the branch behavior and the likely center ambiguity."""

    center_target_peak_mean = float(np.mean([sample["target_center_heatmap_max"] for sample in samples]))
    tip_target_peak_mean = float(np.mean([sample["target_tip_heatmap_max"] for sample in samples]))
    center_target_mean_mean = float(np.mean([sample["target_center_heatmap_mean"] for sample in samples]))
    tip_target_mean_mean = float(np.mean([sample["target_tip_heatmap_mean"] for sample in samples]))
    center_target_std_mean = float(np.mean([sample["target_center_heatmap_std"] for sample in samples]))
    tip_target_std_mean = float(np.mean([sample["target_tip_heatmap_std"] for sample in samples]))

    pred_center_peak_mean = float(np.mean([sample["pred_center_peak"] for sample in samples]))
    pred_tip_peak_mean = float(np.mean([sample["pred_tip_peak"] for sample in samples]))
    pred_center_peak_std = float(np.std([sample["pred_center_peak"] for sample in samples]))
    pred_tip_peak_std = float(np.std([sample["pred_tip_peak"] for sample in samples]))

    center_argmax_error_mean = float(np.mean([sample["center_argmax_error"] for sample in samples]))
    tip_argmax_error_mean = float(np.mean([sample["tip_argmax_error"] for sample in samples]))
    center_soft_error_mean = float(np.mean([sample["center_softargmax_error"] for sample in samples]))
    tip_soft_error_mean = float(np.mean([sample["tip_softargmax_error"] for sample in samples]))

    center_target_distance_mean = float(np.mean([sample["center_target_distance_from_crop_center"] for sample in samples]))
    tip_target_distance_mean = float(np.mean([sample["tip_target_distance_from_crop_center"] for sample in samples]))
    center_target_distance_min = float(np.min([sample["center_target_distance_from_crop_center"] for sample in samples]))
    center_target_distance_max = float(np.max([sample["center_target_distance_from_crop_center"] for sample in samples]))
    tip_target_distance_min = float(np.min([sample["tip_target_distance_from_crop_center"] for sample in samples]))
    tip_target_distance_max = float(np.max([sample["tip_target_distance_from_crop_center"] for sample in samples]))

    center_soft_distance_mean = float(
        np.mean(
            [
                math.hypot(sample["pred_center_x_224_softargmax"] - 112.0, sample["pred_center_y_224_softargmax"] - 112.0)
                for sample in samples
            ]
        )
    )
    tip_soft_distance_mean = float(
        np.mean(
            [
                math.hypot(sample["pred_tip_x_224_softargmax"] - 112.0, sample["pred_tip_y_224_softargmax"] - 112.0)
                for sample in samples
            ]
        )
    )

    lines = [
        "# Geometry Heatmap Branch Error Analysis v2",
        "",
        "## Run Summary",
        "",
        f"- Model: {model_path}",
        f"- Samples: {len(samples)}",
        f"- Heatmap size: {heatmap_size}x{heatmap_size}",
        "- Same 8 clean train examples used by the tiny-overfit gate",
        "",
        "## Target Heatmap Statistics",
        "",
        "| Branch | Mean Max | Mean | Mean Std Dev |",
        "| --- | ---: | ---: | ---: |",
        f"| Center | {center_target_peak_mean:.6f} | {center_target_mean_mean:.6f} | {center_target_std_mean:.6f} |",
        f"| Tip | {tip_target_peak_mean:.6f} | {tip_target_mean_mean:.6f} | {tip_target_std_mean:.6f} |",
        "",
        "## Predicted Heatmap Statistics",
        "",
        "| Branch | Mean Peak | Peak Std Dev | Mean Softargmax Error (px) | Mean Argmax Error (px) |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Center | {pred_center_peak_mean:.6f} | {pred_center_peak_std:.6f} | {center_soft_error_mean:.3f} | {center_argmax_error_mean:.3f} |",
        f"| Tip | {pred_tip_peak_mean:.6f} | {pred_tip_peak_std:.6f} | {tip_soft_error_mean:.3f} | {tip_argmax_error_mean:.3f} |",
        "",
        "## Center Ambiguity Check",
        "",
        f"- Mean center distance from crop center: {center_target_distance_mean:.3f} px",
        f"- Mean tip distance from crop center: {tip_target_distance_mean:.3f} px",
        f"- Center distance range: {center_target_distance_min:.3f} px to {center_target_distance_max:.3f} px",
        f"- Tip distance range: {tip_target_distance_min:.3f} px to {tip_target_distance_max:.3f} px",
        f"- Predicted center distance from crop center (softargmax): {center_soft_distance_mean:.3f} px",
        f"- Predicted tip distance from crop center (softargmax): {tip_soft_distance_mean:.3f} px",
        "",
        "## Interpretation",
        "",
        "- The center target is much closer to the crop hub than the tip target, so even small center mistakes can disturb the angle a lot more than a similar tip mistake.",
        "- On these 8 samples, the center labels live in the hub zone while the tip labels sit far out on the needle, which makes the center branch the more ambiguous branch visually.",
        "- If the v2 tiny-overfit gate still struggles after stronger center weighting, the cleanest simplification may be to keep the tip heatmap and substitute a prior or fixed center for the first board version.",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Analyze the branch mismatch on the same 8-image tiny-overfit subset."""

    parser = argparse.ArgumentParser(description="Analyze tiny heatmap branch errors")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_tiny_overfit_v1/model.keras"),
        help="Legacy tiny-overfit model from v1.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Path to the clean geometry manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/debug/geometry_heatmap_tiny_overfit_v2/branch_error_analysis"),
        help="Directory for branch analysis overlays.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_branch_error_analysis_v2.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Number of samples to inspect.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap grid size.")
    parser.add_argument("--sigma-pixels", type=float, default=2.5, help="Target heatmap sigma.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / args.model_path if not args.model_path.is_absolute() else args.model_path
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model = _load_legacy_model(model_path)
    samples = _load_branch_samples(
        manifest_path=manifest_path,
        base_path=base_path,
        heatmap_size=args.heatmap_size,
        sigma_pixels=args.sigma_pixels,
        limit=args.limit,
    )

    decoded_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        prediction = model.predict(sample.crop_image[np.newaxis, ...], verbose=0)
        decoded_sample = _decode_prediction(sample, prediction, heatmap_size=args.heatmap_size)
        metrics = _sample_metrics(decoded_sample, heatmap_size=args.heatmap_size)
        metrics["example_index"] = index
        decoded_samples.append(metrics)

        overlay_path = output_dir / f"{index:03d}_{Path(sample.image_path).stem}.png"
        _save_branch_overlay(
            decoded_sample,
            metrics=metrics,
            output_path=overlay_path,
            heatmap_size=args.heatmap_size,
        )

    _write_predictions_csv(decoded_samples, output_dir / "branch_predictions.csv")
    _write_report(
        decoded_samples,
        report_path,
        model_path=model_path,
        heatmap_size=args.heatmap_size,
    )

    print(f"Branch overlays written to {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
