#!/usr/bin/env python3
"""Inspect geometry heatmap targets and verify coordinate conversion."""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import (
    load_clean_geometry_examples,
    load_identity_crop,
    make_target_heatmaps,
    select_balanced_examples,
)
from embedded_gauge_reading_tinyml.heatmap_utils import argmax_2d, softargmax_2d


def _save_overlay(
    *,
    crop_image: np.ndarray,
    center_heatmap: np.ndarray,
    tip_heatmap: np.ndarray,
    center_x_224: float,
    center_y_224: float,
    tip_x_224: float,
    tip_y_224: float,
    center_x_norm: float,
    center_y_norm: float,
    tip_x_norm: float,
    tip_y_norm: float,
    output_path: Path,
    heatmap_size: int,
    title: str,
) -> None:
    """Save a figure with the crop image and both target heatmaps."""

    center_row, center_col = argmax_2d(center_heatmap)
    tip_row, tip_col = argmax_2d(tip_heatmap)
    center_soft_row, center_soft_col = softargmax_2d(center_heatmap)
    tip_soft_row, tip_soft_col = softargmax_2d(tip_heatmap)

    center_expected_col = center_x_norm * (heatmap_size - 1)
    center_expected_row = center_y_norm * (heatmap_size - 1)
    tip_expected_col = tip_x_norm * (heatmap_size - 1)
    tip_expected_row = tip_y_norm * (heatmap_size - 1)

    fig = plt.figure(figsize=(14, 9), dpi=140)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.2, 1.0), height_ratios=(1.0, 1.0))

    ax_crop = fig.add_subplot(grid[:, 0])
    ax_center = fig.add_subplot(grid[0, 1])
    ax_tip = fig.add_subplot(grid[1, 1])

    ax_crop.imshow(crop_image)
    ax_crop.scatter(
        [center_x_224, tip_x_224],
        [center_y_224, tip_y_224],
        c=["lime", "red"],
        s=55,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
    )
    ax_crop.plot(
        [center_x_224, tip_x_224],
        [center_y_224, tip_y_224],
        color="deepskyblue",
        linewidth=2.0,
    )
    ax_crop.set_title(
        "Identity crop with true points\n"
        f"center=({center_x_224:.1f}, {center_y_224:.1f}) "
        f"tip=({tip_x_224:.1f}, {tip_y_224:.1f})"
    )
    ax_crop.set_axis_off()

    def _draw_heatmap(
        ax: plt.Axes,
        heatmap: np.ndarray,
        expected_col: float,
        expected_row: float,
        label: str,
    ) -> None:
        """Render one heatmap with expected, argmax, and softargmax markers."""

        row, col = argmax_2d(heatmap)
        soft_row, soft_col = softargmax_2d(heatmap)
        ax.imshow(heatmap, cmap="magma", origin="upper")
        ax.scatter([expected_col], [expected_row], c="white", s=45, marker="o", edgecolors="black")
        ax.scatter([col], [row], c="cyan", s=55, marker="x", linewidths=2.0)
        ax.scatter([soft_col], [soft_row], c="yellow", s=55, marker="d", edgecolors="black")
        ax.set_xlim(-0.5, heatmap_size - 0.5)
        ax.set_ylim(heatmap_size - 0.5, -0.5)
        ax.set_title(f"{label}\nmax={float(np.max(heatmap)):.4f}")
        ax.set_xlabel("x / col")
        ax.set_ylabel("y / row")

    _draw_heatmap(ax_center, center_heatmap, center_expected_col, center_expected_row, "Center heatmap")
    _draw_heatmap(ax_tip, tip_heatmap, tip_expected_col, tip_expected_row, "Tip heatmap")

    summary = [
        f"title: {title}",
        f"center peak: {float(np.max(center_heatmap)):.4f}",
        f"center argmax: ({center_col:.2f}, {center_row:.2f})",
        f"center softargmax: ({center_soft_col:.2f}, {center_soft_row:.2f})",
        f"tip peak: {float(np.max(tip_heatmap)):.4f}",
        f"tip argmax: ({tip_col:.2f}, {tip_row:.2f})",
        f"tip softargmax: ({tip_soft_col:.2f}, {tip_soft_row:.2f})",
    ]
    fig.text(0.02, 0.02, "\n".join(summary), family="monospace", fontsize=10, va="bottom")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_sample_metrics(
    *,
    example_index: int,
    example_path: str,
    split: str,
    temperature_c: float,
    crop_image: np.ndarray,
    metadata: dict[str, float],
    center_heatmap: np.ndarray,
    tip_heatmap: np.ndarray,
    heatmap_size: int,
) -> dict[str, Any]:
    """Compute per-sample stats for the report."""

    center_row, center_col = argmax_2d(center_heatmap)
    tip_row, tip_col = argmax_2d(tip_heatmap)
    center_soft_row, center_soft_col = softargmax_2d(center_heatmap)
    tip_soft_row, tip_soft_col = softargmax_2d(tip_heatmap)

    center_expected_row = metadata["center_y_norm"] * (heatmap_size - 1)
    center_expected_col = metadata["center_x_norm"] * (heatmap_size - 1)
    tip_expected_row = metadata["tip_y_norm"] * (heatmap_size - 1)
    tip_expected_col = metadata["tip_x_norm"] * (heatmap_size - 1)

    center_swapped_row = metadata["center_x_norm"] * (heatmap_size - 1)
    center_swapped_col = metadata["center_y_norm"] * (heatmap_size - 1)
    tip_swapped_row = metadata["tip_x_norm"] * (heatmap_size - 1)
    tip_swapped_col = metadata["tip_y_norm"] * (heatmap_size - 1)

    center_argmax_error = math.hypot(center_col - center_expected_col, center_row - center_expected_row)
    tip_argmax_error = math.hypot(tip_col - tip_expected_col, tip_row - tip_expected_row)
    center_soft_error = math.hypot(center_soft_col - center_expected_col, center_soft_row - center_expected_row)
    tip_soft_error = math.hypot(tip_soft_col - tip_expected_col, tip_soft_row - tip_expected_row)
    center_swapped_error = math.hypot(center_col - center_swapped_col, center_row - center_swapped_row)
    tip_swapped_error = math.hypot(tip_col - tip_swapped_col, tip_row - tip_swapped_row)

    return {
        "example_index": example_index,
        "image_path": example_path,
        "split": split,
        "temperature_c": temperature_c,
        "center_heatmap_min": float(np.min(center_heatmap)),
        "center_heatmap_max": float(np.max(center_heatmap)),
        "center_heatmap_mean": float(np.mean(center_heatmap)),
        "center_heatmap_std": float(np.std(center_heatmap)),
        "tip_heatmap_min": float(np.min(tip_heatmap)),
        "tip_heatmap_max": float(np.max(tip_heatmap)),
        "tip_heatmap_mean": float(np.mean(tip_heatmap)),
        "tip_heatmap_std": float(np.std(tip_heatmap)),
        "center_argmax_error": center_argmax_error,
        "tip_argmax_error": tip_argmax_error,
        "center_softargmax_error": center_soft_error,
        "tip_softargmax_error": tip_soft_error,
        "center_swapped_error": center_swapped_error,
        "tip_swapped_error": tip_swapped_error,
        "center_224_x_error": abs(metadata["center_x_224"] - metadata["center_x_norm"] * 224.0),
        "center_224_y_error": abs(metadata["center_y_224"] - metadata["center_y_norm"] * 224.0),
        "tip_224_x_error": abs(metadata["tip_x_224"] - metadata["tip_x_norm"] * 224.0),
        "tip_224_y_error": abs(metadata["tip_y_224"] - metadata["tip_y_norm"] * 224.0),
        "center_x_224": metadata["center_x_224"],
        "center_y_224": metadata["center_y_224"],
        "tip_x_224": metadata["tip_x_224"],
        "tip_y_224": metadata["tip_y_224"],
        "center_x_norm": metadata["center_x_norm"],
        "center_y_norm": metadata["center_y_norm"],
        "tip_x_norm": metadata["tip_x_norm"],
        "tip_y_norm": metadata["tip_y_norm"],
        "crop_image": crop_image,
        "center_heatmap": center_heatmap,
        "tip_heatmap": tip_heatmap,
    }


def _write_report(samples: list[dict[str, Any]], output_path: Path, *, heatmap_size: int, sigma_pixels: float) -> None:
    """Write a markdown report summarizing target heatmap behavior."""

    center_heatmap_min = min(sample["center_heatmap_min"] for sample in samples)
    center_heatmap_max = max(sample["center_heatmap_max"] for sample in samples)
    center_heatmap_mean = float(np.mean([sample["center_heatmap_mean"] for sample in samples]))
    center_heatmap_std = float(np.mean([sample["center_heatmap_std"] for sample in samples]))
    tip_heatmap_min = min(sample["tip_heatmap_min"] for sample in samples)
    tip_heatmap_max = max(sample["tip_heatmap_max"] for sample in samples)
    tip_heatmap_mean = float(np.mean([sample["tip_heatmap_mean"] for sample in samples]))
    tip_heatmap_std = float(np.mean([sample["tip_heatmap_std"] for sample in samples]))

    center_argmax_error_mean = float(np.mean([sample["center_argmax_error"] for sample in samples]))
    tip_argmax_error_mean = float(np.mean([sample["tip_argmax_error"] for sample in samples]))
    center_soft_error_mean = float(np.mean([sample["center_softargmax_error"] for sample in samples]))
    tip_soft_error_mean = float(np.mean([sample["tip_softargmax_error"] for sample in samples]))
    center_swapped_error_mean = float(np.mean([sample["center_swapped_error"] for sample in samples]))
    tip_swapped_error_mean = float(np.mean([sample["tip_swapped_error"] for sample in samples]))

    center_224_error_mean = float(
        np.mean([sample["center_224_x_error"] + sample["center_224_y_error"] for sample in samples])
    )
    tip_224_error_mean = float(np.mean([sample["tip_224_x_error"] + sample["tip_224_y_error"] for sample in samples]))

    center_ordering_ok = center_argmax_error_mean < center_swapped_error_mean
    tip_ordering_ok = tip_argmax_error_mean < tip_swapped_error_mean
    conversion_ok = center_argmax_error_mean <= 2.0 and tip_argmax_error_mean <= 2.0

    lines = [
        "# Geometry Heatmap Target Debug v1",
        "",
        "## Run Summary",
        "",
        f"- Samples inspected: {len(samples)}",
        f"- Heatmap size: {heatmap_size}x{heatmap_size}",
        f"- Sigma: {sigma_pixels:.2f} pixels",
        f"- Selection: balanced clean rows across train/val/test",
        "",
        "## Heatmap Statistics",
        "",
        "| Map | Min | Max | Mean | Mean Std Dev |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Center | {center_heatmap_min:.6f} | {center_heatmap_max:.6f} | {center_heatmap_mean:.6f} | {center_heatmap_std:.6f} |",
        f"| Tip | {tip_heatmap_min:.6f} | {tip_heatmap_max:.6f} | {tip_heatmap_mean:.6f} | {tip_heatmap_std:.6f} |",
        "",
        "## Argmax Alignment",
        "",
        f"- Center mean argmax error: {center_argmax_error_mean:.3f} heatmap px",
        f"- Tip mean argmax error: {tip_argmax_error_mean:.3f} heatmap px",
        f"- Center mean softargmax error: {center_soft_error_mean:.3f} heatmap px",
        f"- Tip mean softargmax error: {tip_soft_error_mean:.3f} heatmap px",
        f"- Center swapped-order mean error: {center_swapped_error_mean:.3f} heatmap px",
        f"- Tip swapped-order mean error: {tip_swapped_error_mean:.3f} heatmap px",
        "",
        "## Coordinate Conversion Checks",
        "",
        f"- Center normalized-to-224 conversion mean abs error: {center_224_error_mean:.6f} px",
        f"- Tip normalized-to-224 conversion mean abs error: {tip_224_error_mean:.6f} px",
        f"- x/y ordering looks correct for center: {'yes' if center_ordering_ok else 'no'}",
        f"- x/y ordering looks correct for tip: {'yes' if tip_ordering_ok else 'no'}",
        f"- Heatmap argmax is within 1-2 px of the label: {'yes' if conversion_ok else 'no'}",
        "",
        "## Interpretation",
        "",
        "- The target generator is behaving if argmax stays near the expected point and the swapped-order error is much worse.",
        "- The crop-space conversion should be nearly zero because the crop metadata and the target generation use the same normalized coordinates.",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Entry point for target heatmap inspection."""

    parser = argparse.ArgumentParser(description="Inspect ground-truth geometry heatmaps")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("ml/data/geometry_reader_manifest_v2_clean.csv"),
        help="Path to the clean geometry manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/debug/geometry_heatmap_debug_v1/target_heatmaps"),
        help="Directory for target heatmap overlays.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ml/reports/geometry_heatmap_target_debug_v1.md"),
        help="Markdown report path.",
    )
    parser.add_argument("--limit", type=int, default=30, help="Number of clean rows to inspect.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Heatmap grid size.")
    parser.add_argument("--sigma-pixels", type=float, default=2.5, help="Gaussian sigma in heatmap pixels.")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    manifest_path = base_path / args.manifest_path if not args.manifest_path.is_absolute() else args.manifest_path
    output_dir = base_path / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    report_path = base_path / args.report_path if not args.report_path.is_absolute() else args.report_path

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_clean_geometry_examples(manifest_path)
    sampled_examples = select_balanced_examples(examples, total_count=args.limit)

    if len(sampled_examples) < args.limit:
        print(
            f"Only found {len(sampled_examples)} clean rows for inspection; "
            f"requested {args.limit}."
        )

    samples: list[dict[str, Any]] = []
    for index, example in enumerate(sampled_examples):
        crop_image, metadata, _ = load_identity_crop(example, base_path)
        center_heatmap, tip_heatmap = make_target_heatmaps(
            center_x_norm=metadata["center_x_norm"],
            center_y_norm=metadata["center_y_norm"],
            tip_x_norm=metadata["tip_x_norm"],
            tip_y_norm=metadata["tip_y_norm"],
            heatmap_size=args.heatmap_size,
            sigma_pixels=args.sigma_pixels,
        )

        overlay_path = output_dir / f"{index:03d}_{example.split}_{Path(example.image_path).stem}.png"
        _save_overlay(
            crop_image=(crop_image * 255.0).astype(np.uint8),
            center_heatmap=center_heatmap,
            tip_heatmap=tip_heatmap,
            center_x_224=metadata["center_x_224"],
            center_y_224=metadata["center_y_224"],
            tip_x_224=metadata["tip_x_224"],
            tip_y_224=metadata["tip_y_224"],
            center_x_norm=metadata["center_x_norm"],
            center_y_norm=metadata["center_y_norm"],
            tip_x_norm=metadata["tip_x_norm"],
            tip_y_norm=metadata["tip_y_norm"],
            output_path=overlay_path,
            heatmap_size=args.heatmap_size,
            title=f"Target overlay: {Path(example.image_path).name}",
        )

        sample_metrics = _build_sample_metrics(
            example_index=index,
            example_path=example.image_path,
            split=example.split,
            temperature_c=example.temperature_c,
            crop_image=crop_image,
            metadata=metadata,
            center_heatmap=center_heatmap,
            tip_heatmap=tip_heatmap,
            heatmap_size=args.heatmap_size,
        )
        samples.append(sample_metrics)

    _write_report(samples, report_path, heatmap_size=args.heatmap_size, sigma_pixels=args.sigma_pixels)

    print(f"Wrote {len(samples)} target overlays to {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

