"""Board-style preprocessing and visualization helpers for heatmap replay.

The goal of this module is to make the image path reproducible outside the
firmware while keeping the preprocessing contract explicit:
- crop the same loose ROI used during geometry training,
- resize with either the Python-training bilinear path or the board-like
  nearest-neighbor path,
- optionally convert to luma before resizing, and
- return the model-ready tensor together with the crop metadata needed for
  evaluation and overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Final, Literal, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from embedded_gauge_reading_tinyml.board_crop_compare import (
    load_rgb_image,
    load_yuv422_capture_as_rgb,
    rgb_to_luma,
)
from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)
from embedded_gauge_reading_tinyml.geometry_crop_dataset import SourceGeometryExample
from embedded_gauge_reading_tinyml.geometry_heatmap_debug_utils import heatmap_index_to_crop_pixel
from embedded_gauge_reading_tinyml.geometry_heatmap_v2_utils import HeatmapSample
from embedded_gauge_reading_tinyml.heatmap_utils import argmax_2d, softargmax_2d


RGBImage = NDArray[np.uint8]
FloatImage = NDArray[np.float32]
BoardReplayMode = Literal[
    "python_training_rgb_bilinear",
    "board_like_rgb_nearest",
    "board_like_luma_nearest_if_supported",
]

SUPPORTED_BOARD_REPLAY_MODES: Final[tuple[BoardReplayMode, ...]] = (
    "python_training_rgb_bilinear",
    "board_like_rgb_nearest",
    "board_like_luma_nearest_if_supported",
)


@dataclass(frozen=True)
class BoardReplayInput:
    """The preprocessed model input plus the metadata needed for replay."""

    sample: HeatmapSample
    source_kind: str
    preprocessing_mode: BoardReplayMode
    resize_method: str
    metadata: dict[str, Any]


def load_board_replay_image(
    image_path: Path,
    *,
    image_width: int = 224,
    image_height: int = 224,
) -> tuple[RGBImage, str]:
    """Load a replay source image as RGB and return its source kind."""

    suffix = image_path.suffix.lower()
    if suffix == ".yuv422":
        return (
            load_yuv422_capture_as_rgb(
                image_path,
                image_width=image_width,
                image_height=image_height,
            ),
            "yuv422",
        )
    return load_rgb_image(image_path), "rgb"


def _clip_crop_box(
    crop_box_xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Clip a floating-point crop box to integer source-image bounds."""

    x_min, y_min, x_max, y_max = crop_box_xyxy
    x_min_i = max(0, int(math.floor(x_min)))
    y_min_i = max(0, int(math.floor(y_min)))
    x_max_i = min(image_width, int(math.ceil(x_max)))
    y_max_i = min(image_height, int(math.ceil(y_max)))
    if x_max_i <= x_min_i:
        x_max_i = min(image_width, x_min_i + 1)
    if y_max_i <= y_min_i:
        y_max_i = min(image_height, y_min_i + 1)
    return (x_min_i, y_min_i, x_max_i, y_max_i)


def _resize_with_pad(
    image: np.ndarray,
    *,
    crop_box_xyxy: tuple[float, float, float, float],
    image_size: int,
    resample: int,
    mode: Literal["RGB", "L"],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop, resize, and zero-pad an image using the requested resampling rule."""

    image_height, image_width = image.shape[:2]
    x_min_i, y_min_i, x_max_i, y_max_i = _clip_crop_box(
        crop_box_xyxy,
        image_width=image_width,
        image_height=image_height,
    )
    crop = image[y_min_i:y_max_i, x_min_i:x_max_i]
    crop_height, crop_width = crop.shape[:2]
    scale = min(float(image_size) / float(crop_width), float(image_size) / float(crop_height))
    resized_width = max(1, int(round(float(crop_width) * scale)))
    resized_height = max(1, int(round(float(crop_height) * scale)))

    pil_image = Image.fromarray(crop, mode=mode)
    resized = pil_image.resize((resized_width, resized_height), resample=resample)
    resized_array = np.asarray(resized, dtype=np.uint8)

    if mode == "RGB":
        canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        canvas[
            (image_size - resized_height) // 2 : (image_size - resized_height) // 2 + resized_height,
            (image_size - resized_width) // 2 : (image_size - resized_width) // 2 + resized_width,
        ] = resized_array
    else:
        canvas = np.zeros((image_size, image_size), dtype=np.uint8)
        canvas[
            (image_size - resized_height) // 2 : (image_size - resized_height) // 2 + resized_height,
            (image_size - resized_width) // 2 : (image_size - resized_width) // 2 + resized_width,
        ] = resized_array

    pad_y = (image_size - resized_height) // 2
    pad_x = (image_size - resized_width) // 2
    pad_bottom = image_size - resized_height - pad_y
    pad_right = image_size - resized_width - pad_x
    resize_metadata: dict[str, Any] = {
        "crop_x1": int(x_min_i),
        "crop_y1": int(y_min_i),
        "crop_x2": int(x_max_i),
        "crop_y2": int(y_max_i),
        "crop_width": int(crop_width),
        "crop_height": int(crop_height),
        "scale": float(scale),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
        "pad_bottom": int(pad_bottom),
        "pad_right": int(pad_right),
    }
    return canvas, resize_metadata


def _identity_crop_metadata(example: SourceGeometryExample) -> dict[str, float]:
    """Build the identity loose-crop geometry and its transformed labels."""

    crop_x1 = int(example.loose_crop_x1)
    crop_y1 = int(example.loose_crop_y1)
    crop_x2 = int(example.loose_crop_x2)
    crop_y2 = int(example.loose_crop_y2)
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"Invalid loose crop for {example.image_path}: {crop_width}x{crop_height}")

    center_x_norm = (float(example.center_x_source) - float(crop_x1)) / float(crop_width)
    center_y_norm = (float(example.center_y_source) - float(crop_y1)) / float(crop_height)
    tip_x_norm = (float(example.tip_x_source) - float(crop_x1)) / float(crop_width)
    tip_y_norm = (float(example.tip_y_source) - float(crop_y1)) / float(crop_height)
    for name, value in {
        "center_x_norm": center_x_norm,
        "center_y_norm": center_y_norm,
        "tip_x_norm": tip_x_norm,
        "tip_y_norm": tip_y_norm,
    }.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{example.image_path} has {name} outside the crop: {value}")

    center_x_224 = center_x_norm * 224.0
    center_y_224 = center_y_norm * 224.0
    tip_x_224 = tip_x_norm * 224.0
    tip_y_224 = tip_y_norm * 224.0
    angle_degrees = angle_degrees_from_center_to_tip(center_x_224, center_y_224, tip_x_224, tip_y_224)
    deterministic_temperature_c = celsius_from_inner_dial_angle_degrees(angle_degrees)

    return {
        "image_path": example.image_path,
        "split": example.split,
        "temperature_c": float(example.temperature_c),
        "source_manifest": example.source_manifest,
        "source_width": int(example.source_width),
        "source_height": int(example.source_height),
        "quality_flag": example.quality_flag,
        "dial_radius_source": float(example.dial_radius_source),
        "crop_x1": int(crop_x1),
        "crop_y1": int(crop_y1),
        "crop_x2": int(crop_x2),
        "crop_y2": int(crop_y2),
        "crop_width": int(crop_width),
        "crop_height": int(crop_height),
        "jitter_shift_x": 0,
        "jitter_shift_y": 0,
        "jitter_scale": 1.0,
        "jitter_aspect": 1.0,
        "center_x_norm": float(center_x_norm),
        "center_y_norm": float(center_y_norm),
        "tip_x_norm": float(tip_x_norm),
        "tip_y_norm": float(tip_y_norm),
        "center_x_224": float(center_x_224),
        "center_y_224": float(center_y_224),
        "tip_x_224": float(tip_x_224),
        "tip_y_224": float(tip_y_224),
        "angle_degrees": float(angle_degrees),
        "deterministic_temperature_c": float(deterministic_temperature_c),
        "absolute_temperature_difference_c": float(abs(deterministic_temperature_c - float(example.temperature_c))),
    }


def preprocess_board_replay_image(
    image: RGBImage,
    *,
    crop_box_xyxy: tuple[float, float, float, float],
    mode: BoardReplayMode,
    input_size: int = 224,
) -> tuple[FloatImage, dict[str, Any]]:
    """Crop, resize, and normalize one image according to the selected replay mode."""

    if mode == "python_training_rgb_bilinear":
        cropped, resize_metadata = _resize_with_pad(
            image,
            crop_box_xyxy=crop_box_xyxy,
            image_size=input_size,
            resample=Image.Resampling.BILINEAR,
            mode="RGB",
        )
        metadata = {
            **resize_metadata,
            "preprocessing_mode": mode,
            "resize_method": "rgb_bilinear",
            "channel_strategy": "rgb",
            "normalization": "uint8_to_float32_0_1",
        }
        return cropped.astype(np.float32) / 255.0, metadata

    if mode == "board_like_rgb_nearest":
        cropped, resize_metadata = _resize_with_pad(
            image,
            crop_box_xyxy=crop_box_xyxy,
            image_size=input_size,
            resample=Image.Resampling.NEAREST,
            mode="RGB",
        )
        metadata = {
            **resize_metadata,
            "preprocessing_mode": mode,
            "resize_method": "rgb_nearest",
            "channel_strategy": "rgb",
            "normalization": "uint8_to_float32_0_1",
        }
        return cropped.astype(np.float32) / 255.0, metadata

    if mode == "board_like_luma_nearest_if_supported":
        luma = rgb_to_luma(image)
        cropped_luma, resize_metadata = _resize_with_pad(
            luma,
            crop_box_xyxy=crop_box_xyxy,
            image_size=input_size,
            resample=Image.Resampling.NEAREST,
            mode="L",
        )
        rgb = np.repeat(cropped_luma[:, :, None], 3, axis=2)
        metadata = {
            **resize_metadata,
            "preprocessing_mode": mode,
            "resize_method": "luma_nearest",
            "channel_strategy": "luma_replicated_to_rgb",
            "normalization": "uint8_to_float32_0_1",
            "luma_supported": True,
        }
        return rgb.astype(np.float32) / 255.0, metadata

    raise ValueError(f"Unsupported board replay mode: {mode}")


def build_board_replay_sample(
    example: SourceGeometryExample,
    base_path: Path,
    *,
    mode: BoardReplayMode,
    input_size: int = 224,
    heatmap_size: int = 56,
    sigma_pixels: float = 5.0,
) -> HeatmapSample:
    """Build one model-ready board replay sample and its supervision heatmaps."""

    image_path = base_path / Path(example.image_path)
    source_image, source_kind = load_board_replay_image(
        image_path,
        image_width=int(example.source_width),
        image_height=int(example.source_height),
    )
    metadata = _identity_crop_metadata(example)
    input_tensor, preprocess_metadata = preprocess_board_replay_image(
        source_image,
        crop_box_xyxy=(
            float(metadata["crop_x1"]),
            float(metadata["crop_y1"]),
            float(metadata["crop_x2"]),
            float(metadata["crop_y2"]),
        ),
        mode=mode,
        input_size=input_size,
    )

    from embedded_gauge_reading_tinyml.heatmap_utils import HeatmapConfig, generate_center_tip_heatmaps

    heatmap_config = HeatmapConfig(
        heatmap_height=heatmap_size,
        heatmap_width=heatmap_size,
        sigma_pixels=sigma_pixels,
    )
    center_heatmap, tip_heatmap = generate_center_tip_heatmaps(
        float(metadata["center_x_norm"]),
        float(metadata["center_y_norm"]),
        float(metadata["tip_x_norm"]),
        float(metadata["tip_y_norm"]),
        config=heatmap_config,
    )

    return HeatmapSample(
        example=example,
        crop_image=input_tensor.astype(np.float32),
        metadata={
            **metadata,
            **preprocess_metadata,
            "source_kind": source_kind,
            "input_size": int(input_size),
            "heatmap_size": int(heatmap_size),
            "sigma_pixels": float(sigma_pixels),
        },
        center_heatmap=center_heatmap.astype(np.float32),
        tip_heatmap=tip_heatmap.astype(np.float32),
    )


def _decode_heatmap_points(
    heatmap: np.ndarray,
    *,
    decode_method: Literal["softargmax", "argmax"] = "softargmax",
) -> tuple[float, float, float, float]:
    """Decode one heatmap into both softargmax and argmax pixel coordinates."""

    squeezed = np.squeeze(np.asarray(heatmap, dtype=np.float32))
    if squeezed.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {squeezed.shape!r}")
    if decode_method == "softargmax":
        row, col = softargmax_2d(squeezed)
    elif decode_method == "argmax":
        row, col = argmax_2d(squeezed)
    else:
        raise ValueError(f"Unknown decode method: {decode_method}")
    return (
        float(heatmap_index_to_crop_pixel(float(col), heatmap_size=squeezed.shape[1])),
        float(heatmap_index_to_crop_pixel(float(row), heatmap_size=squeezed.shape[0])),
        float(col),
        float(row),
    )


def write_board_replay_overlay(
    sample: HeatmapSample,
    prediction_row: Mapping[str, Any],
    predicted_center_heatmap: np.ndarray,
    predicted_tip_heatmap: np.ndarray,
    output_path: Path,
) -> None:
    """Render one board replay prediction overlay with a compact diagnostics panel."""

    fig = plt.figure(figsize=(17, 10), dpi=150)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.4, 1.0, 0.95), height_ratios=(1.0, 1.0))

    ax_crop = fig.add_subplot(grid[:, 0])
    ax_center = fig.add_subplot(grid[0, 1])
    ax_tip = fig.add_subplot(grid[1, 1])
    ax_text = fig.add_subplot(grid[:, 2])

    crop = np.asarray(sample.crop_image, dtype=np.float32)
    ax_crop.imshow(np.clip(crop, 0.0, 1.0))
    ax_crop.scatter(
        [float(sample.metadata["center_x_224"]), float(prediction_row["predicted_center_x_224"])],
        [float(sample.metadata["center_y_224"]), float(prediction_row["predicted_center_y_224"])],
        c=["lime", "cyan"],
        s=70,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
    )
    ax_crop.scatter(
        [float(sample.metadata["tip_x_224"]), float(prediction_row["predicted_tip_x_224"])],
        [float(sample.metadata["tip_y_224"]), float(prediction_row["predicted_tip_y_224"])],
        c=["red", "yellow"],
        s=75,
        marker="x",
        linewidths=2.0,
    )
    ax_crop.plot(
        [float(sample.metadata["center_x_224"]), float(sample.metadata["tip_x_224"])],
        [float(sample.metadata["center_y_224"]), float(sample.metadata["tip_y_224"])],
        color="white",
        linewidth=2.0,
        alpha=0.9,
    )
    ax_crop.plot(
        [float(prediction_row["predicted_center_x_224"]), float(prediction_row["predicted_tip_x_224"])],
        [float(prediction_row["predicted_center_y_224"]), float(prediction_row["predicted_tip_y_224"])],
        color="deepskyblue",
        linewidth=2.0,
        alpha=0.9,
    )
    ax_crop.set_title("Preprocessed crop fed to model")
    ax_crop.set_axis_off()

    def _plot_heatmap(
        ax: plt.Axes,
        heatmap: np.ndarray,
        *,
        title: str,
        true_x: float,
        true_y: float,
        pred_x: float,
        pred_y: float,
    ) -> None:
        """Draw a predicted heatmap with decoded overlays and peak readout."""

        squeezed = np.squeeze(np.asarray(heatmap, dtype=np.float32))
        row_soft, col_soft = softargmax_2d(squeezed)
        row_argmax, col_argmax = argmax_2d(squeezed)
        ax.imshow(squeezed, cmap="magma", origin="upper")
        ax.scatter([true_x * (squeezed.shape[1] - 1) / 223.0], [true_y * (squeezed.shape[0] - 1) / 223.0], c="white", s=45, marker="o", edgecolors="black", linewidths=0.8)
        ax.scatter([pred_x * (squeezed.shape[1] - 1) / 223.0], [pred_y * (squeezed.shape[0] - 1) / 223.0], c="cyan", s=55, marker="x", linewidths=2.0)
        ax.scatter([col_argmax], [row_argmax], c="yellow", s=40, marker="d", edgecolors="black", linewidths=0.8)
        ax.scatter([col_soft], [row_soft], c="lime", s=35, marker="s", edgecolors="black", linewidths=0.8)
        ax.set_xlim(-0.5, squeezed.shape[1] - 0.5)
        ax.set_ylim(squeezed.shape[0] - 0.5, -0.5)
        ax.set_title(f"{title}\npeak={float(np.max(squeezed)):.4f}")
        ax.set_xlabel("x / col")
        ax.set_ylabel("y / row")

    _plot_heatmap(
        ax_center,
        predicted_center_heatmap,
        title="Predicted center heatmap",
        true_x=float(sample.metadata["center_x_224"]),
        true_y=float(sample.metadata["center_y_224"]),
        pred_x=float(prediction_row["predicted_center_x_224"]),
        pred_y=float(prediction_row["predicted_center_y_224"]),
    )
    _plot_heatmap(
        ax_tip,
        predicted_tip_heatmap,
        title="Predicted tip heatmap",
        true_x=float(sample.metadata["tip_x_224"]),
        true_y=float(sample.metadata["tip_y_224"]),
        pred_x=float(prediction_row["predicted_tip_x_224"]),
        pred_y=float(prediction_row["predicted_tip_y_224"]),
    )

    status = str(prediction_row["guardrail_status"])
    reasons = str(prediction_row["rejection_reasons"])
    if not reasons:
        reasons = "none"
    old_status = str(prediction_row.get("old_guardrail_status", ""))
    old_reasons = str(prediction_row.get("old_rejection_reasons", ""))
    if not old_reasons:
        old_reasons = "none"
    text_lines = [
        f"file: {Path(str(sample.metadata['image_path'])).name}",
        f"split: {sample.metadata['split']}",
        f"mode: {prediction_row['preprocessing_mode']}",
        f"old guardrail: {old_status or 'n/a'}",
        f"old reasons: {old_reasons}",
        f"new guardrail: {status}",
        f"new reasons: {reasons}",
        "",
        f"true temp: {float(sample.metadata['temperature_c']):.2f} C",
        f"pred temp current: {float(prediction_row['predicted_temperature_c_current_mapping']):.2f} C",
        f"pred temp calibrated: {float(prediction_row['predicted_temperature_c_calibrated']):.2f} C",
        f"guarded temp: {float(prediction_row['guarded_temperature_c']) if math.isfinite(float(prediction_row['guarded_temperature_c'])) else float('nan'):.2f} C",
        f"abs err guarded: {float(prediction_row['absolute_error_c_guarded']) if math.isfinite(float(prediction_row['absolute_error_c_guarded'])) else float('nan'):.2f} C",
        "",
        f"confidence: {float(prediction_row['confidence']):.4f}",
        f"center err: {float(prediction_row['center_px_mae_224']):.2f} px",
        f"tip err: {float(prediction_row['tip_px_mae_224']):.2f} px",
        f"center peak: {float(prediction_row['center_heatmap_peak_value']):.4f}",
        f"tip peak: {float(prediction_row['tip_heatmap_peak_value']):.4f}",
        f"center entropy: {float(prediction_row['center_heatmap_entropy']):.4f}",
        f"tip entropy: {float(prediction_row['tip_heatmap_entropy']):.4f}",
        f"center spread: {float(prediction_row['center_heatmap_spread_px']):.2f} px",
        f"tip spread: {float(prediction_row['tip_heatmap_spread_px']):.2f} px",
        f"distance ratio: {float(prediction_row['center_tip_distance_ratio']):.3f}",
    ]
    ax_text.set_axis_off()
    ax_text.text(0.0, 1.0, "\n".join(text_lines), family="monospace", fontsize=9.5, va="top")

    fig.suptitle(
        f"{Path(str(sample.metadata['image_path'])).name} | {status} | {prediction_row['preprocessing_mode']}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
