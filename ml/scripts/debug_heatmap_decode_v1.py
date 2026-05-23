#!/usr/bin/env python3
"""Inspect predictions from the failed geometry heatmap v1 model."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

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
    select_examples_from_split,
)
from embedded_gauge_reading_tinyml.heatmap_utils import argmax_2d, softargmax_2d


def _draw_predicted_overlay(
    *,
    crop_image: np.ndarray,
    center_heatmap: np.ndarray,
    tip_heatmap: np.ndarray,
    metadata: dict[str, float],
    center_argmax: tuple[float, float],
    tip_argmax: tuple[float, float],
    center_softargmax: tuple[float, float],
    tip_softargmax: tuple[float, float],
    output_path: Path,
    heatmap_size: int,
    title: str,
) -> None:
    """Save a decode inspection overlay for one sample."""

    fig = plt.figure(figsize=(15, 9), dpi=140)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.2, 1.0), height_ratios=(1.0, 1.0))

    ax_crop = fig.add_subplot(grid[:, 0])
    ax_center = fig.add_subplot(grid[0, 1])
    ax_tip = fig.add_subplot(grid[1, 1])

    ax_crop.imshow(crop_image)
    ax_crop.plot(
        [metadata["center_x_224"], metadata["tip_x_224"]],
        [metadata["center_y_224"], metadata["tip_y_224"]],
        color="white",
        linewidth=2.5,
        label="true line",
    )
    ax_crop.plot(
        [center_argmax[0], tip_argmax[0]],
        [center_argmax[1], tip_argmax[1]],
        color="cyan",
        linewidth=2.0,
        label="argmax line",
    )
    ax_crop.plot(
        [center_softargmax[0], tip_softargmax[0]],
        [center_softargmax[1], tip_softargmax[1]],
        color="yellow",
        linewidth=2.0,
        linestyle="--",
        label="softargmax line",
    )
    ax_crop.scatter(
        [metadata["center_x_224"], metadata["tip_x_224"]],
        [metadata["center_y_224"], metadata["tip_y_224"]],
        c=["lime", "red"],
        s=60,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        label="true points",
    )
    ax_crop.scatter(
        [center_argmax[0], tip_argmax[0]],
        [center_argmax[1], tip_argmax[1]],
        c=["cyan", "cyan"],
        s=70,
        marker="x",
        linewidths=2.0,
        label="argmax",
    )
    ax_crop.scatter(
        [center_softargmax[0], tip_softargmax[0]],
        [center_softargmax[1], tip_softargmax[1]],
        c=["yellow", "yellow"],
        s=60,
        marker="d",
        edgecolors="black",
        label="softargmax",
    )
    ax_crop.set_title("Crop view")
    ax_crop.set_axis_off()
    ax_crop.legend(loc="lower right", fontsize=8, framealpha=0.85)

    def _draw_heatmap_panel(
        ax: plt.Axes,
        heatmap: np.ndarray,
        expected_x: float,
        expected_y: float,
        argmax_xy: tuple[float, float],
        soft_xy: tuple[float, float],
        label: str,
    ) -> None:
        """Render a single predicted heatmap panel."""

        ax.imshow(heatmap, cmap="magma", origin="upper")
        ax.scatter([expected_x], [expected_y], c="white", s=45, marker="o", edgecolors="black")
        ax.scatter([argmax_xy[0]], [argmax_xy[1]], c="cyan", s=55, marker="x", linewidths=2.0)
        ax.scatter([soft_xy[0]], [soft_xy[1]], c="yellow", s=55, marker="d", edgecolors="black")
        ax.set_xlim(-0.5, heatmap_size - 0.5)
        ax.set_ylim(heatmap_size - 0.5, -0.5)
        ax.set_title(f"{label}\nmax={float(np.max(heatmap)):.4f}")
        ax.set_xlabel("x / col")
        ax.set_ylabel("y / row")

    _draw_heatmap_panel(
        ax_center,
        center_heatmap,
        metadata["center_x_norm"] * (heatmap_size - 1),
        metadata["center_y_norm"] * (heatmap_size - 1),
        center_argmax,
        center_softargmax,
        "Predicted center heatmap",
    )
    _draw_heatmap_panel(
        ax_tip,
        tip_heatmap,
        metadata["tip_x_norm"] * (heatmap_size - 1),
        metadata["tip_y_norm"] * (heatmap_size - 1),
        tip_argmax,
        tip_softargmax,
        "Predicted tip heatmap",
    )

    summary_lines = [
        f"{title}",
        f"center peak: {float(np.max(center_heatmap)):.4f}",
        f"tip peak: {float(np.max(tip_heatmap)):.4f}",
        f"center argmax: ({center_argmax[0]:.2f}, {center_argmax[1]:.2f})",
        f"center softargmax: ({center_softargmax[0]:.2f}, {center_softargmax[1]:.2f})",
        f"tip argmax: ({tip_argmax[0]:.2f}, {tip_argmax[1]:.2f})",
        f"tip softargmax: ({tip_softargmax[0]:.2f}, {tip_softargmax[1]:.2f})",
    ]
    fig.text(0.02, 0.02, "\n".join(summary_lines), family="monospace", fontsize=10, va="bottom")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _sample_prediction(
    model: keras.Model,
    *,
    example_index: int,
    example_path: str,
    split: str,
    temperature_c: float,
    crop_image: np.ndarray,
    metadata: dict[str, float],
    center_heatmap: np.ndarray,
    tip_heatmap: np.ndarray,
    center_argmax: tuple[float, float],
    tip_argmax: tuple[float, float],
    center_softargmax: tuple[float, float],
    tip_softargmax: tuple[float, float],
    confidence: float,
    heatmap_size: int,
) -> dict[str, Any]:
    """Build a metrics row for one decoded prediction."""

    center_argmax_error = math.hypot(center_argmax[0] - metadata["center_x_224"], center_argmax[1] - metadata["center_y_224"])
    tip_argmax_error = math.hypot(tip_argmax[0] - metadata["tip_x_224"], tip_argmax[1] - metadata["tip_y_224"])
    center_soft_error = math.hypot(
        center_softargmax[0] - metadata["center_x_224"],
        center_softargmax[1] - metadata["center_y_224"],
    )
    tip_soft_error = math.hypot(
        tip_softargmax[0] - metadata["tip_x_224"],
        tip_softargmax[1] - metadata["tip_y_224"],
    )

    center_distance_from_image_center = math.hypot(center_softargmax[0] - 112.0, center_softargmax[1] - 112.0)
    tip_distance_from_image_center = math.hypot(tip_softargmax[0] - 112.0, tip_softargmax[1] - 112.0)

    predicted_angle = angle_degrees_from_center_to_tip(
        center_softargmax[0],
        center_softargmax[1],
        tip_softargmax[0],
        tip_softargmax[1],
    )
    true_angle = angle_degrees_from_center_to_tip(
        metadata["center_x_224"],
        metadata["center_y_224"],
        metadata["tip_x_224"],
        metadata["tip_y_224"],
    )
    predicted_temp = celsius_from_inner_dial_angle_degrees(predicted_angle)

    return {
        "example_index": example_index,
        "image_path": example_path,
        "split": split,
        "temperature_c": temperature_c,
        "predicted_temperature_c": predicted_temp,
        "absolute_error_c": abs(predicted_temp - temperature_c),
        "true_angle_degrees": true_angle,
        "predicted_angle_degrees": predicted_angle,
        "angle_error_degrees": abs(circular_angle_error_degrees(predicted_angle, true_angle)),
        "confidence": confidence,
        "center_heatmap_max": float(np.max(center_heatmap)),
        "tip_heatmap_max": float(np.max(tip_heatmap)),
        "center_heatmap_std": float(np.std(center_heatmap)),
        "tip_heatmap_std": float(np.std(tip_heatmap)),
        "center_argmax_error": center_argmax_error,
        "tip_argmax_error": tip_argmax_error,
        "center_softargmax_error": center_soft_error,
        "tip_softargmax_error": tip_soft_error,
        "argmax_softargmax_gap_center": math.hypot(center_argmax[0] - center_softargmax[0], center_argmax[1] - center_softargmax[1]),
        "argmax_softargmax_gap_tip": math.hypot(tip_argmax[0] - tip_softargmax[0], tip_argmax[1] - tip_softargmax[1]),
        "center_distance_from_image_center": center_distance_from_image_center,
        "tip_distance_from_image_center": tip_distance_from_image_center,
        "predicted_center_x_224_argmax": center_argmax[0],
        "predicted_center_y_224_argmax": center_argmax[1],
        "predicted_tip_x_224_argmax": tip_argmax[0],
        "predicted_tip_y_224_argmax": tip_argmax[1],
        "predicted_center_x_224_softargmax": center_softargmax[0],
        "predicted_center_y_224_softargmax": center_softargmax[1],
        "predicted_tip_x_224_softargmax": tip_softargmax[0],
        "predicted_tip_y_224_softargmax": tip_softargmax[1],
        "center_heatmap": center_heatmap,
        "tip_heatmap": tip_heatmap,
        "crop_image": crop_image,
        "model_name": model.name,
        "heatmap_size": heatmap_size,
    }


def _write_report(samples: list[dict[str, Any]], output_path: Path, *, model_path: Path) -> None:
    """Write a markdown report summarizing the decode inspection."""

    center_peak_mean = float(np.mean([sample["center_heatmap_max"] for sample in samples]))
    tip_peak_mean = float(np.mean([sample["tip_heatmap_max"] for sample in samples]))
    center_peak_std = float(np.std([sample["center_heatmap_max"] for sample in samples]))
    tip_peak_std = float(np.std([sample["tip_heatmap_max"] for sample in samples]))
    center_heatmap_std_mean = float(np.mean([sample["center_heatmap_std"] for sample in samples]))
    tip_heatmap_std_mean = float(np.mean([sample["tip_heatmap_std"] for sample in samples]))

    center_argmax_error_mean = float(np.mean([sample["center_argmax_error"] for sample in samples]))
    tip_argmax_error_mean = float(np.mean([sample["tip_argmax_error"] for sample in samples]))
    center_soft_error_mean = float(np.mean([sample["center_softargmax_error"] for sample in samples]))
    tip_soft_error_mean = float(np.mean([sample["tip_softargmax_error"] for sample in samples]))

    center_gap_mean = float(np.mean([sample["argmax_softargmax_gap_center"] for sample in samples]))
    tip_gap_mean = float(np.mean([sample["argmax_softargmax_gap_tip"] for sample in samples]))
    center_distance_from_center_mean = float(np.mean([sample["center_distance_from_image_center"] for sample in samples]))
    tip_distance_from_center_mean = float(np.mean([sample["tip_distance_from_image_center"] for sample in samples]))

    collapse_to_center = center_distance_from_center_mean < 25.0 and tip_distance_from_center_mean < 25.0
    nearly_flat = center_peak_mean < 0.20 and tip_peak_mean < 0.20 and center_heatmap_std_mean < 0.05 and tip_heatmap_std_mean < 0.05

    worst_5 = sorted(samples, key=lambda row: row["absolute_error_c"], reverse=True)[:5]

    lines = [
        "# Geometry Heatmap Decode Debug v1",
        "",
        "## Run Summary",
        "",
        f"- Model: {model_path}",
        f"- Samples inspected: {len(samples)}",
        "- Split: test",
        "- Decode methods: argmax and softargmax",
        "",
        "## Heatmap Peak Statistics",
        "",
        "| Map | Mean Max | Std Dev of Max | Mean Heatmap Std Dev |",
        "| --- | ---: | ---: | ---: |",
        f"| Center | {center_peak_mean:.6f} | {center_peak_std:.6f} | {center_heatmap_std_mean:.6f} |",
        f"| Tip | {tip_peak_mean:.6f} | {tip_peak_std:.6f} | {tip_heatmap_std_mean:.6f} |",
        "",
        "## Decode Error",
        "",
        f"- Center argmax error: {center_argmax_error_mean:.3f} px",
        f"- Tip argmax error: {tip_argmax_error_mean:.3f} px",
        f"- Center softargmax error: {center_soft_error_mean:.3f} px",
        f"- Tip softargmax error: {tip_soft_error_mean:.3f} px",
        f"- Mean argmax-softargmax gap, center: {center_gap_mean:.3f} px",
        f"- Mean argmax-softargmax gap, tip: {tip_gap_mean:.3f} px",
        "",
        "## Collapse / Flatness Checks",
        "",
        f"- Mean center distance from crop center (softargmax): {center_distance_from_center_mean:.3f} px",
        f"- Mean tip distance from crop center (softargmax): {tip_distance_from_center_mean:.3f} px",
        f"- Predictions collapsing to center: {'yes' if collapse_to_center else 'no'}",
        f"- Heatmaps nearly flat: {'yes' if nearly_flat else 'no'}",
        "",
        "## Worst 5 Predictions",
        "",
        "| Rank | Image | Abs Error (C) | Center Argmax Error (px) | Tip Argmax Error (px) | Peak Center | Peak Tip |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for rank, row in enumerate(worst_5, start=1):
        lines.append(
            f"| {rank} | {Path(row['image_path']).name} | {row['absolute_error_c']:.3f} | "
            f"{row['center_argmax_error']:.3f} | {row['tip_argmax_error']:.3f} | "
            f"{row['center_heatmap_max']:.4f} | {row['tip_heatmap_max']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the heatmaps are diffuse, argmax and softargmax often diverge or stay near the crop center while peak values remain low.",
            "- A healthy heatmap should show a sharp peak, with argmax and softargmax landing close to the same point and close to the true label.",
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Run the decode inspection on the failed geometry heatmap v1 model."""

    parser = argparse.ArgumentParser(description="Inspect decoded geometry heatmap predictions")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ml/artifacts/training/geometry_heatmap_v1/model.keras"),
        help="Path to the failed heatmap v1 model.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Path to the clean manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/debug/geometry_heatmap_debug_v1/predicted_heatmaps"),
        help="Directory for prediction overlays.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_decode_debug_v1.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--limit", type=int, default=30, help="Number of test rows to inspect.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Predicted heatmap size.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / args.model_path if not args.model_path.is_absolute() else args.model_path
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    examples = load_clean_geometry_examples(manifest_path)
    test_examples = select_examples_from_split(examples, split="test", limit=args.limit)
    if len(test_examples) < args.limit:
        print(f"Only found {len(test_examples)} clean test rows; requested {args.limit}.")

    model = keras.models.load_model(model_path, compile=False)

    samples: list[dict[str, Any]] = []
    for index, example in enumerate(test_examples):
        crop_image, metadata, _ = load_identity_crop(example, base_path)
        prediction = model.predict(crop_image[np.newaxis, ...], verbose=0)
        center_heatmap = np.asarray(prediction[0][0, ..., 0], dtype=np.float32)
        tip_heatmap = np.asarray(prediction[1][0, ..., 0], dtype=np.float32)
        confidence = float(prediction[2][0, 0])

        center_argmax_row, center_argmax_col = argmax_2d(center_heatmap)
        tip_argmax_row, tip_argmax_col = argmax_2d(tip_heatmap)
        center_soft_row, center_soft_col = softargmax_2d(center_heatmap)
        tip_soft_row, tip_soft_col = softargmax_2d(tip_heatmap)

        center_argmax = (
            heatmap_index_to_crop_pixel(center_argmax_col, heatmap_size=args.heatmap_size),
            heatmap_index_to_crop_pixel(center_argmax_row, heatmap_size=args.heatmap_size),
        )
        tip_argmax = (
            heatmap_index_to_crop_pixel(tip_argmax_col, heatmap_size=args.heatmap_size),
            heatmap_index_to_crop_pixel(tip_argmax_row, heatmap_size=args.heatmap_size),
        )
        center_softargmax = (
            heatmap_index_to_crop_pixel(center_soft_col, heatmap_size=args.heatmap_size),
            heatmap_index_to_crop_pixel(center_soft_row, heatmap_size=args.heatmap_size),
        )
        tip_softargmax = (
            heatmap_index_to_crop_pixel(tip_soft_col, heatmap_size=args.heatmap_size),
            heatmap_index_to_crop_pixel(tip_soft_row, heatmap_size=args.heatmap_size),
        )

        overlay_path = output_dir / f"{index:03d}_{example.split}_{Path(example.image_path).stem}.png"
        _draw_predicted_overlay(
            crop_image=(crop_image * 255.0).astype(np.uint8),
            center_heatmap=center_heatmap,
            tip_heatmap=tip_heatmap,
            metadata=metadata,
            center_argmax=center_argmax,
            tip_argmax=tip_argmax,
            center_softargmax=center_softargmax,
            tip_softargmax=tip_softargmax,
            output_path=overlay_path,
            heatmap_size=args.heatmap_size,
            title=f"Predicted overlay: {Path(example.image_path).name}",
        )

        samples.append(
            _sample_prediction(
                model,
                example_index=index,
                example_path=example.image_path,
                split=example.split,
                temperature_c=example.temperature_c,
                crop_image=crop_image,
                metadata=metadata,
                center_heatmap=center_heatmap,
                tip_heatmap=tip_heatmap,
                center_argmax=center_argmax,
                tip_argmax=tip_argmax,
                center_softargmax=center_softargmax,
                tip_softargmax=tip_softargmax,
                confidence=confidence,
                heatmap_size=args.heatmap_size,
            )
        )

    _write_report(samples, report_path, model_path=model_path)

    print(f"Wrote {len(samples)} prediction overlays to {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

