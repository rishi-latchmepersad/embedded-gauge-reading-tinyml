"""Shared utilities for the geometry heatmap v2 training and evaluation flow."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
    circular_angle_error_degrees,
)
from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    JitterParams,
    SourceGeometryExample,
    create_jittered_crop,
    load_geometry_manifest,
)
from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import (
    heatmap_index_to_crop_pixel,
)
from embedded_gauge_reading_tinyml.geometry_temperature_calibration import (
    CalibrationCandidate,
    predict_temperature_from_candidate,
)
from embedded_gauge_reading_tinyml.heatmap_utils import argmax_2d, softargmax_2d


@dataclass(frozen=True)
class HeatmapSample:
    """One crop, its metadata, and its supervision heatmaps."""

    example: SourceGeometryExample
    crop_image: np.ndarray
    metadata: dict[str, Any]
    center_heatmap: np.ndarray
    tip_heatmap: np.ndarray


def load_clean_geometry_examples(manifest_path: Path) -> list[SourceGeometryExample]:
    """Load the manifest and keep only clean rows."""

    return [example for example in load_geometry_manifest(manifest_path) if example.quality_flag == "clean"]


def select_examples_from_split(
    examples: Sequence[SourceGeometryExample],
    *,
    split: str,
) -> list[SourceGeometryExample]:
    """Return deterministic examples for one manifest split."""

    return sorted([example for example in examples if example.split == split], key=lambda example: example.image_path)


def sample_jitter_params(
    rng: np.random.Generator,
    *,
    shift_min_px: int,
    shift_max_px: int,
    scale_min: float,
    scale_max: float,
    aspect_min: float,
    aspect_max: float,
) -> JitterParams:
    """Sample a mild geometric jitter configuration."""

    max_shift = int(rng.integers(shift_min_px, shift_max_px + 1))
    return JitterParams(
        shift_x=int(rng.integers(-max_shift, max_shift + 1)),
        shift_y=int(rng.integers(-max_shift, max_shift + 1)),
        scale=float(rng.uniform(scale_min, scale_max)),
        aspect=float(rng.uniform(aspect_min, aspect_max)),
    )


def _identity_jitter() -> JitterParams:
    """Return the deterministic identity crop jitter."""

    return JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)


def load_heatmap_sample(
    example: SourceGeometryExample,
    base_path: Path,
    *,
    input_size: int = 224,
    heatmap_size: int = 56,
    sigma_pixels: float = 5.0,
    jitter: JitterParams | None = None,
    max_attempts: int = 25,
    inner_celsius_mask: bool = False,
) -> HeatmapSample:
    """Load one cropped image and its target heatmaps.

    Args:
        example: Source geometry example with crop box and keypoints.
        base_path: Root directory for source images.
        input_size: Square pixel size after resize (default 224).
        heatmap_size: Square size of the target heatmaps (default 56).
        sigma_pixels: Gaussian sigma for target heatmaps.
        jitter: Optional random crop jitter.
        max_attempts: Max crop jitter retries.
        inner_celsius_mask: If True, apply the shared inner-Celsius-only
            mask after resize to exclude outer distractors.
    """

    jitter = _identity_jitter() if jitter is None else jitter
    crop = create_jittered_crop(example, jitter)
    attempts = 1
    if not crop.accepted and max_attempts > 1:
        crop = create_jittered_crop(example, _identity_jitter())
        attempts += 1

    image_path = base_path / crop.source_image_path
    with Image.open(image_path) as image:
        crop_box = (crop.crop_x1, crop.crop_y1, crop.crop_x2, crop.crop_y2)
        # Crop before converting so large source frames do not allocate a full
        # RGB buffer when we only need the dial region.
        crop_image = image.crop(crop_box).convert("RGB").resize(
            (input_size, input_size),
            Image.Resampling.LANCZOS,
        )
        crop_array = np.asarray(crop_image, dtype=np.float32) / 255.0

    if inner_celsius_mask:
        from embedded_gauge_reading_tinyml.inner_celsius_mask import apply_inner_celsius_mask
        crop_array = apply_inner_celsius_mask(crop_array)

    from embedded_gauge_reading_tinyml.heatmap_utils import HeatmapConfig, generate_center_tip_heatmaps

    heatmap_config = HeatmapConfig(
        heatmap_height=heatmap_size,
        heatmap_width=heatmap_size,
        sigma_pixels=sigma_pixels,
    )
    center_heatmap, tip_heatmap = generate_center_tip_heatmaps(
        crop.center_x_normalized,
        crop.center_y_normalized,
        crop.tip_x_normalized,
        crop.tip_y_normalized,
        config=heatmap_config,
    )

    metadata: dict[str, Any] = {
        "image_path": crop.source_image_path,
        "split": crop.split,
        "temperature_c": float(crop.temperature_c),
        "source_manifest": example.source_manifest,
        "source_width": int(example.source_width),
        "source_height": int(example.source_height),
        "quality_flag": example.quality_flag,
        "dial_radius_source": float(example.dial_radius_source),
        "crop_x1": int(crop.crop_x1),
        "crop_y1": int(crop.crop_y1),
        "crop_x2": int(crop.crop_x2),
        "crop_y2": int(crop.crop_y2),
        "crop_width": int(crop.crop_x2 - crop.crop_x1),
        "crop_height": int(crop.crop_y2 - crop.crop_y1),
        "jitter_shift_x": int(crop.jitter_shift_x),
        "jitter_shift_y": int(crop.jitter_shift_y),
        "jitter_scale": float(crop.jitter_scale),
        "jitter_aspect": float(crop.jitter_aspect),
        "center_x_norm": float(crop.center_x_normalized),
        "center_y_norm": float(crop.center_y_normalized),
        "tip_x_norm": float(crop.tip_x_normalized),
        "tip_y_norm": float(crop.tip_y_normalized),
        "center_x_224": float(crop.center_x_224),
        "center_y_224": float(crop.center_y_224),
        "tip_x_224": float(crop.tip_x_224),
        "tip_y_224": float(crop.tip_y_224),
        "angle_degrees": float(crop.angle_degrees) if crop.angle_degrees is not None else math.nan,
        "deterministic_temperature_c": float(crop.deterministic_temperature_c) if crop.deterministic_temperature_c is not None else math.nan,
        "absolute_temperature_difference_c": float(crop.absolute_temperature_difference_c) if crop.absolute_temperature_difference_c is not None else math.nan,
        "image_width": int(input_size),
        "image_height": int(input_size),
        "heatmap_size": int(heatmap_size),
        "sigma_pixels": float(sigma_pixels),
        "source_image_width": int(example.source_width),
        "source_image_height": int(example.source_height),
        "loaded_with_identity_crop": bool(
            crop.jitter_shift_x == 0 and crop.jitter_shift_y == 0 and crop.jitter_scale == 1.0 and crop.jitter_aspect == 1.0
        ),
        "jitter_attempts": int(attempts),
        "inner_celsius_mask": bool(inner_celsius_mask),
    }

    return HeatmapSample(
        example=example,
        crop_image=crop_array,
        metadata=metadata,
        center_heatmap=center_heatmap.astype(np.float32),
        tip_heatmap=tip_heatmap.astype(np.float32),
    )


def load_heatmap_samples(
    examples: Sequence[SourceGeometryExample],
    base_path: Path,
    *,
    input_size: int = 224,
    heatmap_size: int = 56,
    sigma_pixels: float = 5.0,
    jitter: JitterParams | None = None,
) -> list[HeatmapSample]:
    """Load a list of crop samples."""

    return [
        load_heatmap_sample(
            example,
            base_path,
            input_size=input_size,
            heatmap_size=heatmap_size,
            sigma_pixels=sigma_pixels,
            jitter=jitter,
        )
        for example in examples
    ]


def _heatmap_index_to_crop_index(index_value: float, *, heatmap_size: int, crop_size: int = 224) -> float:
    """Map a crop pixel coordinate back into heatmap index space."""

    return float(index_value) * float(heatmap_size - 1) / float(crop_size)


def _softargmax_or_argmax_coords(heatmap: np.ndarray, *, method: str) -> tuple[float, float]:
    """Decode a single heatmap into x/y pixel coordinates at 224x224 scale."""

    if method == "softargmax":
        row, col = softargmax_2d(heatmap)
    elif method == "argmax":
        row, col = argmax_2d(heatmap)
    else:
        raise ValueError(f"Unknown decode method: {method}")
    return float(heatmap_index_to_crop_pixel(col, heatmap_size=heatmap.shape[1])), float(
        heatmap_index_to_crop_pixel(row, heatmap_size=heatmap.shape[0])
    )


def decode_prediction_row(
    sample: HeatmapSample,
    predicted_center_heatmap: np.ndarray,
    predicted_tip_heatmap: np.ndarray,
    confidence: float,
    *,
    calibration_candidate: CalibrationCandidate | None = None,
) -> dict[str, Any]:
    """Decode a model output bundle into a rich metrics row."""

    center_heatmap = np.squeeze(np.asarray(predicted_center_heatmap, dtype=np.float32))
    tip_heatmap = np.squeeze(np.asarray(predicted_tip_heatmap, dtype=np.float32))

    pred_center_x_argmax, pred_center_y_argmax = _softargmax_or_argmax_coords(center_heatmap, method="argmax")
    pred_tip_x_argmax, pred_tip_y_argmax = _softargmax_or_argmax_coords(tip_heatmap, method="argmax")
    pred_center_x_soft, pred_center_y_soft = _softargmax_or_argmax_coords(center_heatmap, method="softargmax")
    pred_tip_x_soft, pred_tip_y_soft = _softargmax_or_argmax_coords(tip_heatmap, method="softargmax")

    true_center_x = float(sample.metadata["center_x_224"])
    true_center_y = float(sample.metadata["center_y_224"])
    true_tip_x = float(sample.metadata["tip_x_224"])
    true_tip_y = float(sample.metadata["tip_y_224"])
    true_angle = angle_degrees_from_center_to_tip(true_center_x, true_center_y, true_tip_x, true_tip_y)
    predicted_angle = angle_degrees_from_center_to_tip(pred_center_x_soft, pred_center_y_soft, pred_tip_x_soft, pred_tip_y_soft)
    predicted_angle_argmax = angle_degrees_from_center_to_tip(
        pred_center_x_argmax,
        pred_center_y_argmax,
        pred_tip_x_argmax,
        pred_tip_y_argmax,
    )

    predicted_temperature_current = celsius_from_inner_dial_angle_degrees(predicted_angle)
    predicted_temperature_current_argmax = celsius_from_inner_dial_angle_degrees(predicted_angle_argmax)
    predicted_temperature_calibrated = (
        predicted_temperature_current
        if calibration_candidate is None
        else predict_temperature_from_candidate(predicted_angle, calibration_candidate)
    )
    predicted_temperature_calibrated_argmax = (
        predicted_temperature_current_argmax
        if calibration_candidate is None
        else predict_temperature_from_candidate(predicted_angle_argmax, calibration_candidate)
    )

    row: dict[str, Any] = {
        **sample.metadata,
        "image_path": sample.metadata["image_path"],
        "split": sample.metadata["split"],
        "true_temperature_c": float(sample.metadata["temperature_c"]),
        "true_angle_degrees": float(true_angle),
        "predicted_angle_degrees": float(predicted_angle),
        "predicted_angle_degrees_argmax": float(predicted_angle_argmax),
        "predicted_temperature_c_current_mapping": float(predicted_temperature_current),
        "predicted_temperature_c_current_mapping_argmax": float(predicted_temperature_current_argmax),
        "predicted_temperature_c_calibrated": float(predicted_temperature_calibrated),
        "predicted_temperature_c_calibrated_argmax": float(predicted_temperature_calibrated_argmax),
        "absolute_error_c_current_mapping": float(abs(predicted_temperature_current - float(sample.metadata["temperature_c"]))),
        "absolute_error_c_current_mapping_argmax": float(
            abs(predicted_temperature_current_argmax - float(sample.metadata["temperature_c"]))
        ),
        "absolute_error_c_calibrated": float(abs(predicted_temperature_calibrated - float(sample.metadata["temperature_c"]))),
        "absolute_error_c_calibrated_argmax": float(
            abs(predicted_temperature_calibrated_argmax - float(sample.metadata["temperature_c"]))
        ),
        "predicted_center_x_224": float(pred_center_x_soft),
        "predicted_center_y_224": float(pred_center_y_soft),
        "predicted_tip_x_224": float(pred_tip_x_soft),
        "predicted_tip_y_224": float(pred_tip_y_soft),
        "predicted_center_x_224_argmax": float(pred_center_x_argmax),
        "predicted_center_y_224_argmax": float(pred_center_y_argmax),
        "predicted_tip_x_224_argmax": float(pred_tip_x_argmax),
        "predicted_tip_y_224_argmax": float(pred_tip_y_argmax),
        "center_px_mae_224": float(
            math.hypot(pred_center_x_soft - true_center_x, pred_center_y_soft - true_center_y)
        ),
        "tip_px_mae_224": float(math.hypot(pred_tip_x_soft - true_tip_x, pred_tip_y_soft - true_tip_y)),
        "center_px_mae_224_argmax": float(
            math.hypot(pred_center_x_argmax - true_center_x, pred_center_y_argmax - true_center_y)
        ),
        "tip_px_mae_224_argmax": float(math.hypot(pred_tip_x_argmax - true_tip_x, pred_tip_y_argmax - true_tip_y)),
        "angle_mae_degrees": float(abs(circular_angle_error_degrees(predicted_angle, true_angle))),
        "angle_mae_degrees_argmax": float(abs(circular_angle_error_degrees(predicted_angle_argmax, true_angle))),
        "center_heatmap_peak_value": float(np.max(center_heatmap)),
        "tip_heatmap_peak_value": float(np.max(tip_heatmap)),
        "center_heatmap_mean_value": float(np.mean(center_heatmap)),
        "tip_heatmap_mean_value": float(np.mean(tip_heatmap)),
        "confidence": float(confidence),
        "pred_center_heatmap_array": center_heatmap,
        "pred_tip_heatmap_array": tip_heatmap,
    }
    return row


def write_prediction_overlay(
    sample: HeatmapSample,
    prediction_row: dict[str, Any],
    output_path: Path,
    *,
    heatmap_size: int = 56,
    show_argmax: bool = True,
) -> None:
    """Render a crop overlay with heatmap insets."""

    fig = plt.figure(figsize=(17, 10), dpi=150)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 1.0), height_ratios=(1.0, 1.0))

    ax_crop = fig.add_subplot(grid[:, 0])
    ax_center_pred = fig.add_subplot(grid[0, 1])
    ax_tip_pred = fig.add_subplot(grid[1, 1])
    ax_center_target = fig.add_subplot(grid[0, 2])
    ax_tip_target = fig.add_subplot(grid[1, 2])

    crop = sample.crop_image
    ax_crop.imshow(crop)
    ax_crop.scatter(
        [float(sample.metadata["center_x_224"]), float(prediction_row["predicted_center_x_224"])],
        [float(sample.metadata["center_y_224"]), float(prediction_row["predicted_center_y_224"])],
        c=["lime", "cyan"],
        s=70,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        label="center",
    )
    ax_crop.scatter(
        [float(sample.metadata["tip_x_224"]), float(prediction_row["predicted_tip_x_224"])],
        [float(sample.metadata["tip_y_224"]), float(prediction_row["predicted_tip_y_224"])],
        c=["red", "yellow"],
        s=70,
        marker="x",
        linewidths=2.0,
        label="tip",
    )
    ax_crop.plot(
        [float(sample.metadata["center_x_224"]), float(sample.metadata["tip_x_224"])],
        [float(sample.metadata["center_y_224"]), float(sample.metadata["tip_y_224"])],
        color="white",
        linewidth=2.0,
        alpha=0.85,
        label="true needle",
    )
    ax_crop.plot(
        [float(prediction_row["predicted_center_x_224"]), float(prediction_row["predicted_tip_x_224"])],
        [float(prediction_row["predicted_center_y_224"]), float(prediction_row["predicted_tip_y_224"])],
        color="deepskyblue",
        linewidth=2.0,
        alpha=0.85,
        label="pred needle",
    )
    ax_crop.set_title("Crop overlay")
    ax_crop.set_axis_off()
    ax_crop.legend(loc="lower right", fontsize=8, framealpha=0.9)

    def _plot_heatmap(ax: plt.Axes, heatmap: np.ndarray, *, title: str, true_x: float, true_y: float, pred_x: float, pred_y: float) -> None:
        """Plot a predicted heatmap with true/predicted markers."""

        row_argmax, col_argmax = argmax_2d(heatmap)
        row_soft, col_soft = softargmax_2d(heatmap)
        ax.imshow(heatmap, cmap="magma", origin="upper")
        ax.scatter(
            [_heatmap_index_to_crop_index(true_x, heatmap_size=heatmap_size)],
            [_heatmap_index_to_crop_index(true_y, heatmap_size=heatmap_size)],
            c="white",
            s=45,
            marker="o",
            edgecolors="black",
            linewidths=1.0,
        )
        ax.scatter(
            [_heatmap_index_to_crop_index(pred_x, heatmap_size=heatmap_size)],
            [_heatmap_index_to_crop_index(pred_y, heatmap_size=heatmap_size)],
            c="cyan",
            s=55,
            marker="x",
            linewidths=2.0,
        )
        if show_argmax:
            ax.scatter([col_argmax], [row_argmax], c="yellow", s=40, marker="d", edgecolors="black", linewidths=0.8)
            ax.scatter([col_soft], [row_soft], c="lime", s=35, marker="s", edgecolors="black", linewidths=0.8)
        ax.set_xlim(-0.5, heatmap_size - 0.5)
        ax.set_ylim(heatmap_size - 0.5, -0.5)
        ax.set_title(f"{title}\nmax={float(np.max(heatmap)):.4f}")
        ax.set_xlabel("x / col")
        ax.set_ylabel("y / row")

    _plot_heatmap(
        ax_center_pred,
        np.asarray(prediction_row["pred_center_heatmap_array"], dtype=np.float32),
        title="Predicted center heatmap",
        true_x=float(sample.metadata["center_x_224"]),
        true_y=float(sample.metadata["center_y_224"]),
        pred_x=float(prediction_row["predicted_center_x_224"]),
        pred_y=float(prediction_row["predicted_center_y_224"]),
    )
    _plot_heatmap(
        ax_tip_pred,
        np.asarray(prediction_row["pred_tip_heatmap_array"], dtype=np.float32),
        title="Predicted tip heatmap",
        true_x=float(sample.metadata["tip_x_224"]),
        true_y=float(sample.metadata["tip_y_224"]),
        pred_x=float(prediction_row["predicted_tip_x_224"]),
        pred_y=float(prediction_row["predicted_tip_y_224"]),
    )
    _plot_heatmap(
        ax_center_target,
        sample.center_heatmap,
        title="Target center heatmap",
        true_x=float(sample.metadata["center_x_224"]),
        true_y=float(sample.metadata["center_y_224"]),
        pred_x=float(sample.metadata["center_x_224"]),
        pred_y=float(sample.metadata["center_y_224"]),
    )
    _plot_heatmap(
        ax_tip_target,
        sample.tip_heatmap,
        title="Target tip heatmap",
        true_x=float(sample.metadata["tip_x_224"]),
        true_y=float(sample.metadata["tip_y_224"]),
        pred_x=float(sample.metadata["tip_x_224"]),
        pred_y=float(sample.metadata["tip_y_224"]),
    )

    summary = [
        f"file: {Path(str(sample.metadata['image_path'])).name}",
        f"split: {sample.metadata['split']}",
        f"true temp: {float(sample.metadata['temperature_c']):.2f} C",
        f"pred temp current: {float(prediction_row['predicted_temperature_c_current_mapping']):.2f} C",
        f"pred temp calibrated: {float(prediction_row['predicted_temperature_c_calibrated']):.2f} C",
        f"abs err calibrated: {float(prediction_row['absolute_error_c_calibrated']):.2f} C",
        f"confidence: {float(prediction_row['confidence']):.4f}",
        f"center err: {float(prediction_row['center_px_mae_224']):.2f} px",
        f"tip err: {float(prediction_row['tip_px_mae_224']):.2f} px",
    ]
    fig.suptitle(Path(str(sample.metadata["image_path"])).name, fontsize=15)
    fig.text(0.02, 0.01, "\n".join(summary), family="monospace", fontsize=10, va="bottom")
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_selected_calibration_candidate(calibration_json_path: Path) -> tuple[CalibrationCandidate, dict[str, Any]]:
    """Load the selected train-fitted calibration candidate from JSON."""

    with open(calibration_json_path, "r", encoding="utf-8") as handle:
        calibration_json = json.load(handle)

    selected_candidate_name = str(calibration_json["selected_candidate_name"])
    raw_candidate = calibration_json["candidates"][selected_candidate_name]
    candidate = CalibrationCandidate(
        name=str(raw_candidate["name"]),
        kind=str(raw_candidate["kind"]),
        params={str(key): float(value) for key, value in raw_candidate["params"].items()},
    )
    return candidate, calibration_json
