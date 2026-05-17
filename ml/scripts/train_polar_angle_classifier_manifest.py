#!/usr/bin/env python3
"""Train a polar-profile angle classifier on a manifest of gauge images.

This is the polar-voting branch we actually want:
1. crop each gauge to the rectified ROI,
2. project it into polar space,
3. collapse the polar image into a 1D angular profile,
4. classify the needle position with soft sweep-aligned labels, and
5. convert the predicted sweep position back into temperature.

The model predicts sweep-position bins rather than raw temperature. That keeps
the geometry explicit and matches the spirit of the classical polar-voting
baseline much better than a direct scalar regressor or a temperature-bin
distribution head.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import (  # noqa: E402
    load_rgb_image,
    load_yuv422_capture_as_rgb,
    resize_with_pad_rgb,
)
from embedded_gauge_reading_tinyml.gauge.processing import (  # noqa: E402
    fraction_to_angle_rad,
    load_gauge_specs,
    value_to_fraction,
)
from embedded_gauge_reading_tinyml.polar_projection import (  # noqa: E402
    polar_project_image,
)

VALUE_MIN: float = -30.0
VALUE_MAX: float = 50.0
IMAGE_SIZE: int = 224
NUM_ANGLE_BINS: int = 90
SOFT_TARGET_SIGMA_BINS: float = 2.0
STRATIFY_BIN_SIZE: float = 10.0
DEFAULT_TARGET_MODE: Literal["circular", "sweep"] = "sweep"
DEFAULT_STRUCTURE_MODE: Literal["vote", "ordinal", "coarse_to_fine", "two_stage"] = "vote"
DEFAULT_LOSS_MODE: Literal["standard", "balanced_softmax"] = "standard"
DEFAULT_AUX_HEAD_MODE: Literal["none", "broad"] = "none"
DEFAULT_AUX_SIGMA_MULTIPLIER: float = 3.0
DEFAULT_COARSE_BINS: int = 16
DEFAULT_FINE_BINS: int = 14
DEFAULT_FRACTION_LOSS_WEIGHT: float = 0.0
DEFAULT_FRACTION_LOSS_DELTA: float = 0.04
DEFAULT_SWEEP_KERNEL: Literal["gaussian", "reflect"] = "gaussian"


@dataclass(frozen=True)
class ManifestRecord:
    """One row from a merged image/value manifest."""

    image_path: Path
    value: float
    sample_weight: float


def _normalize_path(path_str: str, repo_root: Path) -> str:
    """Normalize a manifest path to a repo-relative POSIX path."""
    normalized = path_str.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        try:
            path = path.relative_to(repo_root)
        except ValueError:
            pass
    return path.as_posix()


def _resolve_path(normalized_path: str, repo_root: Path) -> Path:
    """Resolve a normalized manifest path back to an absolute path."""
    return repo_root / normalized_path


def _load_manifest(manifest_path: Path, repo_root: Path) -> pd.DataFrame:
    """Load a CSV manifest and standardize its path/value columns."""
    df = pd.read_csv(manifest_path)
    if "image_path" not in df.columns:
        raise ValueError(f"Manifest is missing image_path column: {manifest_path}")
    if "value" not in df.columns:
        raise ValueError(f"Manifest is missing value column: {manifest_path}")

    df = df.copy()
    df["image_path"] = df["image_path"].apply(
        lambda p: _normalize_path(str(p), repo_root)
    )
    df["image_path_resolved"] = df["image_path"].apply(
        lambda p: str(_resolve_path(p, repo_root))
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "sample_weight" in df.columns:
        df["sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce")
    df = df.dropna(subset=["value"])
    if "sample_weight" in df.columns:
        df = df.dropna(subset=["sample_weight"])
    df = df[df["image_path_resolved"].apply(lambda p: Path(p).exists())]
    return df.reset_index(drop=True)


def _load_crop_boxes(
    crop_boxes_path: Path, repo_root: Path
) -> dict[str, tuple[float, float, float, float]]:
    """Load rectifier crop boxes keyed by absolute image path."""
    boxes_df = pd.read_csv(crop_boxes_path)
    required_columns = {"image_path", "x0", "y0", "x1", "y1"}
    missing = required_columns.difference(boxes_df.columns)
    if missing:
        raise ValueError(
            f"Crop-box CSV is missing columns {sorted(missing)}: {crop_boxes_path}"
        )

    boxes: dict[str, tuple[float, float, float, float]] = {}
    for _, row in boxes_df.iterrows():
        normalized = _normalize_path(str(row["image_path"]), repo_root)
        resolved = str(_resolve_path(normalized, repo_root))
        boxes[resolved] = (
            float(row["x0"]),
            float(row["y0"]),
            float(row["x1"]),
            float(row["y1"]),
        )
    return boxes


def _load_rgb_or_yuv422(image_path: Path) -> np.ndarray:
    """Load a labeled image or board capture as RGB."""
    suffix = image_path.suffix.lower()
    if suffix == ".yuv422":
        return load_yuv422_capture_as_rgb(image_path)
    return load_rgb_image(image_path)


def _crop_and_polar_profile(
    image_path: Path,
    *,
    crop_box: tuple[float, float, float, float] | None,
    polar_size: int,
) -> np.ndarray:
    """Crop an image and collapse its polar projection into a 1D profile."""
    rgb = _load_rgb_or_yuv422(image_path)
    if crop_box is not None:
        cropped = resize_with_pad_rgb(rgb, crop_box, image_size=polar_size)
    else:
        cropped = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize(
                (polar_size, polar_size), resample=Image.BILINEAR
            ),
            dtype=np.uint8,
        )

    polar = polar_project_image(
        cropped,
        center_xy=(float(polar_size) * 0.5, float(polar_size) * 0.5),
        max_radius=float(polar_size) * 0.5,
        polar_size=polar_size,
    )
    profile = np.sum(polar, axis=(0, 2))
    max_value = float(np.max(profile))
    if max_value > 0.0:
        profile = profile / max_value
    return profile.astype(np.float32)


def _build_polar_vote_prior_channel(
    luma: np.ndarray,
    grad_theta: np.ndarray,
) -> np.ndarray:
    """Build a baseline-style angular prior map from polar luma/edge evidence.

    The goal is not to replace learning, but to provide one additional channel
    that highlights angle columns with dark, radially-continuous evidence.
    """
    if luma.ndim != 2 or grad_theta.ndim != 2:
        raise ValueError("luma and grad_theta must be 2D arrays.")
    if luma.shape != grad_theta.shape:
        raise ValueError("luma and grad_theta must share the same shape.")

    height, width = luma.shape
    row_positions = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
    radial_mask = ((row_positions >= 0.18) & (row_positions <= 0.95)).astype(np.float32)
    radial_mask = radial_mask[:, None]

    darkness = np.clip((0.62 - luma) / 0.62, 0.0, 1.0).astype(np.float32)
    edge = np.clip(grad_theta, 0.0, 1.0).astype(np.float32)
    evidence = (0.65 * darkness + 0.35 * edge) * radial_mask

    denom = float(np.sum(radial_mask))
    if denom <= 1e-6:
        return np.zeros((height, width, 1), dtype=np.float32)
    mean_evidence = np.sum(evidence, axis=0) / np.float32(denom)

    # Baseline-like continuity cue: columns where dark support persists across
    # radius are more likely to be the needle than isolated texture.
    darkness_threshold = float(np.quantile(luma, 0.38))
    binary_dark = ((luma <= darkness_threshold).astype(np.float32) * radial_mask).astype(np.float32)
    continuity = np.zeros(width, dtype=np.float32)
    for column in range(width):
        support = binary_dark[:, column]
        run = 0
        best = 0
        for value in support:
            if value > 0.5:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        continuity[column] = np.float32(best / max(1, height))

    score = (0.70 * mean_evidence + 0.30 * continuity).astype(np.float32)

    # Smooth along angle bins so neighboring columns cooperate on weak frames.
    score_smoothed = (
        0.50 * score
        + 0.25 * np.roll(score, 1)
        + 0.25 * np.roll(score, -1)
    ).astype(np.float32)
    score_smoothed -= np.min(score_smoothed)
    peak = float(np.max(score_smoothed))
    if peak > 1e-6:
        score_smoothed = score_smoothed / np.float32(peak)
    score_map = np.repeat(score_smoothed[None, :], height, axis=0)
    return score_map[..., None].astype(np.float32)


def _score_polar_alignment(polar_rgb: np.ndarray) -> float:
    """Score how well a polar image aligns with a dark radial needle trace."""
    luma = (
        0.299 * polar_rgb[..., 0]
        + 0.587 * polar_rgb[..., 1]
        + 0.114 * polar_rgb[..., 2]
    ).astype(np.float32)
    height = luma.shape[0]
    row_positions = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
    radial_mask = ((row_positions >= 0.18) & (row_positions <= 0.95)).astype(np.float32)
    radial_mask = radial_mask[:, None]

    darkness_threshold = float(np.quantile(luma, 0.38))
    binary_dark = ((luma <= darkness_threshold).astype(np.float32) * radial_mask).astype(np.float32)

    continuity_sum = 0.0
    for column in range(binary_dark.shape[1]):
        support = binary_dark[:, column]
        run = 0
        best = 0
        for value in support:
            if value > 0.5:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        continuity_sum += float(best)
    continuity = continuity_sum / float(max(1, binary_dark.shape[1] * height))
    return continuity


def _crop_and_polar_image(
    image_path: Path,
    *,
    crop_box: tuple[float, float, float, float] | None,
    polar_size: int,
    input_mode: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = "rgb",
    center_search_px: int = 0,
) -> np.ndarray:
    """Crop an image and keep the full polar projection as a 2D CNN input."""
    rgb = _load_rgb_or_yuv422(image_path)
    if crop_box is not None:
        cropped = resize_with_pad_rgb(rgb, crop_box, image_size=polar_size)
    else:
        cropped = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize(
                (polar_size, polar_size), resample=Image.BILINEAR
            ),
            dtype=np.uint8,
        )

    default_center_x = float(polar_size) * 0.5
    default_center_y = float(polar_size) * 0.5

    if center_search_px <= 0:
        polar = polar_project_image(
            cropped,
            center_xy=(default_center_x, default_center_y),
            max_radius=float(polar_size) * 0.5,
            polar_size=polar_size,
        )
    else:
        offsets = [0, -center_search_px, center_search_px]
        best_score = -1.0
        best_polar: np.ndarray | None = None
        for dy in offsets:
            for dx in offsets:
                candidate = polar_project_image(
                    cropped,
                    center_xy=(default_center_x + float(dx), default_center_y + float(dy)),
                    max_radius=float(polar_size) * 0.5,
                    polar_size=polar_size,
                )
                candidate_rgb = (candidate.astype(np.float32) / 255.0).astype(np.float32)
                score = _score_polar_alignment(candidate_rgb)
                if score > best_score:
                    best_score = score
                    best_polar = candidate
        if best_polar is None:
            raise RuntimeError("Center search failed to produce a polar projection.")
        polar = best_polar

    polar_rgb = (polar.astype(np.float32) / 255.0).astype(np.float32)
    if input_mode == "rgb":
        return polar_rgb

    # Edge-aware three-channel input for low-light robustness:
    #  - ch0: normalized luma
    #  - ch1: normalized |dI/dtheta| (angular gradient)
    #  - ch2: normalized |dI/dr| (radial gradient)
    luma = (
        0.299 * polar_rgb[..., 0]
        + 0.587 * polar_rgb[..., 1]
        + 0.114 * polar_rgb[..., 2]
    ).astype(np.float32)
    luma = np.clip(luma, 0.0, 1.0)

    grad_r, grad_theta = np.gradient(luma)
    grad_theta = np.abs(grad_theta).astype(np.float32)
    grad_r = np.abs(grad_r).astype(np.float32)

    def _normalize_channel(channel: np.ndarray) -> np.ndarray:
        hi = float(np.percentile(channel, 99.0))
        if hi <= 1e-6:
            return np.zeros_like(channel, dtype=np.float32)
        return np.clip(channel / hi, 0.0, 1.0).astype(np.float32)

    edge3 = np.stack(
        [
            _normalize_channel(luma),
            _normalize_channel(grad_theta),
            _normalize_channel(grad_r),
        ],
        axis=-1,
    )
    if input_mode == "edge3":
        return edge3.astype(np.float32)

    if input_mode == "rgb_edge6":
        return np.concatenate([polar_rgb, edge3], axis=-1).astype(np.float32)

    vote_prior = _build_polar_vote_prior_channel(luma, edge3[..., 1])
    return np.concatenate([polar_rgb, edge3, vote_prior], axis=-1).astype(np.float32)


def _angle_to_soft_target(
    angle_rad: float,
    *,
    num_bins: int,
    sigma_bins: float,
) -> np.ndarray:
    """Convert an angle into a circular Gaussian distribution over bins."""
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2.")
    if sigma_bins <= 0.0:
        raise ValueError("sigma_bins must be > 0.")

    bin_position = (float(angle_rad) % (2.0 * math.pi)) / (2.0 * math.pi)
    center_bin = bin_position * float(num_bins)
    bin_indices = np.arange(num_bins, dtype=np.float32)
    distances = np.minimum(
        np.abs(bin_indices - np.float32(center_bin)),
        np.float32(num_bins) - np.abs(bin_indices - np.float32(center_bin)),
    )
    target = np.exp(-0.5 * (distances / np.float32(sigma_bins)) ** 2)
    total = float(np.sum(target))
    if total > 0.0:
        target /= np.float32(total)
    return target.astype(np.float32)


def _fraction_to_linear_soft_target(
    fraction: float,
    *,
    num_bins: int,
    sigma_bins: float,
    sweep_kernel: Literal["gaussian", "reflect"] = DEFAULT_SWEEP_KERNEL,
) -> np.ndarray:
    """Convert a sweep fraction into a soft target over linear bins."""
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2.")
    if sigma_bins <= 0.0:
        raise ValueError("sigma_bins must be > 0.")

    fraction_clamped = min(max(float(fraction), 0.0), 1.0)
    center_bin = fraction_clamped * float(num_bins - 1)
    bin_indices = np.arange(num_bins, dtype=np.float32)
    center = np.float32(center_bin)
    if sweep_kernel == "reflect":
        # Reflective boundary correction:
        # mirror the center around both ends so edge labels keep symmetric mass
        # instead of being biased toward the interior bins.
        left_center = -center
        right_center = np.float32(2.0 * float(num_bins - 1)) - center
        target = (
            np.exp(-0.5 * ((bin_indices - center) / np.float32(sigma_bins)) ** 2)
            + np.exp(-0.5 * ((bin_indices - left_center) / np.float32(sigma_bins)) ** 2)
            + np.exp(-0.5 * ((bin_indices - right_center) / np.float32(sigma_bins)) ** 2)
        )
    else:
        distances = np.abs(bin_indices - center)
        target = np.exp(-0.5 * (distances / np.float32(sigma_bins)) ** 2)
    total = float(np.sum(target))
    if total > 0.0:
        target /= np.float32(total)
    return target.astype(np.float32)


def _fraction_to_ordinal_target(
    fraction: float,
    *,
    num_thresholds: int,
    sigma_bins: float,
    sweep_kernel: Literal["gaussian", "reflect"] = DEFAULT_SWEEP_KERNEL,
) -> np.ndarray:
    """Convert a sweep fraction into a cumulative ordinal target vector."""
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be >= 2.")

    base_target = _fraction_to_linear_soft_target(
        fraction,
        num_bins=num_thresholds,
        sigma_bins=sigma_bins,
        sweep_kernel=sweep_kernel,
    )
    ordinal_target = np.cumsum(base_target[::-1], dtype=np.float32)[::-1]
    return ordinal_target.astype(np.float32)


def _fraction_to_coarse_fine_targets(
    fraction: float,
    *,
    coarse_bins: int,
    fine_bins: int,
    sigma_bins: float,
    sweep_kernel: Literal["gaussian", "reflect"] = DEFAULT_SWEEP_KERNEL,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one fraction into coarse and fine categorical targets.

    The coarse target covers the broad sweep sector. The fine target captures
    the within-sector residual, which gives the second head a much easier local
    decision to learn.
    """
    if coarse_bins < 2:
        raise ValueError("coarse_bins must be >= 2.")
    if fine_bins < 2:
        raise ValueError("fine_bins must be >= 2.")

    combined_bins = coarse_bins * fine_bins
    base_target = _fraction_to_linear_soft_target(
        fraction,
        num_bins=combined_bins,
        sigma_bins=sigma_bins,
        sweep_kernel=sweep_kernel,
    )
    coarse_target = base_target.reshape(coarse_bins, fine_bins).sum(axis=1)
    coarse_sum = float(np.sum(coarse_target))
    if coarse_sum > 0.0:
        coarse_target = coarse_target / np.float32(coarse_sum)

    scaled_position = min(max(float(fraction), 0.0), 1.0) * float(combined_bins - 1)
    coarse_index = int(math.floor(scaled_position / float(fine_bins)))
    coarse_index = max(0, min(coarse_index, coarse_bins - 1))
    fine_start = coarse_index * fine_bins
    fine_target = base_target[fine_start : fine_start + fine_bins]
    fine_sum = float(np.sum(fine_target))
    if fine_sum > 0.0:
        fine_target = fine_target / np.float32(fine_sum)
    return coarse_target.astype(np.float32), fine_target.astype(np.float32)


def _ordinal_logits_to_temperature(
    logits: np.ndarray,
    *,
    value_min: float,
    value_max: float,
) -> np.ndarray:
    """Convert cumulative ordinal logits into Celsius values."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits[None, ...]
    probs = 1.0 / (1.0 + np.exp(-logits))
    fraction = np.mean(probs, axis=-1)
    span = np.float32(value_max - value_min)
    return (np.float32(value_min) + fraction * span).astype(np.float32)


def _coarse_fine_logits_to_temperature(
    coarse_logits: np.ndarray,
    fine_logits: np.ndarray,
    *,
    value_min: float,
    value_max: float,
) -> np.ndarray:
    """Convert coarse/fine logits into Celsius values."""
    coarse_logits = np.asarray(coarse_logits, dtype=np.float32)
    fine_logits = np.asarray(fine_logits, dtype=np.float32)
    if coarse_logits.ndim == 1:
        coarse_logits = coarse_logits[None, ...]
    if fine_logits.ndim == 1:
        fine_logits = fine_logits[None, ...]

    coarse_shifted = np.exp(coarse_logits - np.max(coarse_logits, axis=-1, keepdims=True))
    coarse_probs = coarse_shifted / np.sum(coarse_shifted, axis=-1, keepdims=True)
    fine_shifted = np.exp(fine_logits - np.max(fine_logits, axis=-1, keepdims=True))
    fine_probs = fine_shifted / np.sum(fine_shifted, axis=-1, keepdims=True)

    coarse_positions = np.arange(coarse_probs.shape[-1], dtype=np.float32)
    fine_positions = np.arange(fine_probs.shape[-1], dtype=np.float32)
    coarse_expectation = np.sum(coarse_probs * coarse_positions[None, :], axis=-1)
    fine_expectation = np.sum(fine_probs * fine_positions[None, :], axis=-1)

    total_bins = float(coarse_probs.shape[-1] * fine_probs.shape[-1] - 1)
    if total_bins <= 0.0:
        raise ValueError("coarse/fine bin product must be > 1.")
    fraction = (coarse_expectation * float(fine_probs.shape[-1]) + fine_expectation) / np.float32(total_bins)
    span = np.float32(value_max - value_min)
    return (np.float32(value_min) + fraction * span).astype(np.float32)


def _build_vote_backbone(
    *,
    polar_size: int,
    input_channels: int,
    base_filters: int,
    head_units: int,
    dropout: float,
) -> tuple[keras.Input, tf.Tensor]:
    """Build the shared polar vote feature extractor."""
    inputs = keras.Input(
        shape=(polar_size, polar_size, input_channels),
        name="polar_image",
    )

    x = keras.layers.Conv2D(
        base_filters,
        kernel_size=3,
        strides=(2, 1),
        padding="same",
        use_bias=False,
        name="vote_conv2d_1",
    )(inputs)
    x = keras.layers.BatchNormalization(name="vote_bn2d_1")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_1")(x)

    x = keras.layers.SeparableConv2D(
        base_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_sepconv2d_2",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_2")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_2")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 1), name="vote_pool2d_1")(x)

    x = keras.layers.SeparableConv2D(
        base_filters * 2,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_sepconv2d_3",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_3")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_3")(x)
    x = keras.layers.SeparableConv2D(
        base_filters * 2,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_sepconv2d_4",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_4")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_4")(x)

    radial_mean = keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name="vote_radial_mean",
    )(x)
    radial_max = keras.layers.Lambda(
        lambda t: tf.reduce_max(t, axis=1),
        name="vote_radial_max",
    )(x)
    x = keras.layers.Concatenate(axis=-1, name="vote_radial_fuse")(
        [radial_mean, radial_max]
    )

    x = keras.layers.Conv1D(
        head_units,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_conv1d_1",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn1d_1")(x)
    x = keras.layers.Activation("swish", name="vote_act1d_1")(x)
    x = keras.layers.Dropout(dropout, name="vote_dropout")(x)
    return inputs, x


def _build_model(
    *,
    polar_size: int,
    num_bins: int,
    base_filters: int = 32,
    head_units: int = 128,
    dropout: float = 0.2,
) -> keras.Model:
    """Build a compact 1D CNN that votes over angle bins."""
    inputs = keras.Input(shape=(polar_size,), name="polar_profile")
    x = keras.layers.Reshape((polar_size, 1), name="profile_reshape")(inputs)

    x = keras.layers.Conv1D(
        base_filters,
        kernel_size=5,
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Activation("swish", name="act_1")(x)
    x = keras.layers.Conv1D(
        base_filters,
        kernel_size=5,
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = keras.layers.BatchNormalization(name="bn_2")(x)
    x = keras.layers.Activation("swish", name="act_2")(x)
    x = keras.layers.MaxPooling1D(pool_size=2, name="pool_1")(x)

    x = keras.layers.Conv1D(
        base_filters * 2,
        kernel_size=5,
        padding="same",
        use_bias=False,
        name="conv_3",
    )(x)
    x = keras.layers.BatchNormalization(name="bn_3")(x)
    x = keras.layers.Activation("swish", name="act_3")(x)
    x = keras.layers.Conv1D(
        base_filters * 2,
        kernel_size=5,
        padding="same",
        use_bias=False,
        name="conv_4",
    )(x)
    x = keras.layers.BatchNormalization(name="bn_4")(x)
    x = keras.layers.Activation("swish", name="act_4")(x)
    x = keras.layers.MaxPooling1D(pool_size=2, name="pool_2")(x)

    x = keras.layers.Conv1D(
        base_filters * 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="conv_5",
    )(x)
    x = keras.layers.BatchNormalization(name="bn_5")(x)
    x = keras.layers.Activation("swish", name="act_5")(x)
    x = keras.layers.Conv1D(
        base_filters * 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="conv_6",
    )(x)
    x = keras.layers.BatchNormalization(name="bn_6")(x)
    x = keras.layers.Activation("swish", name="act_6")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(head_units, activation="swish", name="head_dense")(x)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    angle_logits = keras.layers.Dense(num_bins, name="angle_logits")(x)

    return keras.Model(inputs=inputs, outputs=angle_logits, name="polar_angle_profile")


def _build_model_2d(
    *,
    polar_size: int,
    input_channels: int,
    num_bins: int,
    base_filters: int = 32,
    head_units: int = 128,
    dropout: float = 0.2,
) -> keras.Model:
    """Build a lightweight 2D polar CNN for angle voting."""
    inputs = keras.Input(
        shape=(polar_size, polar_size, input_channels),
        name="polar_image",
    )
    x = keras.layers.Conv2D(
        base_filters,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="conv2d_1",
    )(inputs)
    x = keras.layers.BatchNormalization(name="bn2d_1")(x)
    x = keras.layers.Activation("swish", name="act2d_1")(x)

    x = keras.layers.SeparableConv2D(
        base_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="sepconv2d_2",
    )(x)
    x = keras.layers.BatchNormalization(name="bn2d_2")(x)
    x = keras.layers.Activation("swish", name="act2d_2")(x)
    x = keras.layers.MaxPooling2D(pool_size=2, name="pool2d_1")(x)

    x = keras.layers.SeparableConv2D(
        base_filters * 2,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="sepconv2d_3",
    )(x)
    x = keras.layers.BatchNormalization(name="bn2d_3")(x)
    x = keras.layers.Activation("swish", name="act2d_3")(x)
    x = keras.layers.SeparableConv2D(
        base_filters * 2,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="sepconv2d_4",
    )(x)
    x = keras.layers.BatchNormalization(name="bn2d_4")(x)
    x = keras.layers.Activation("swish", name="act2d_4")(x)
    x = keras.layers.MaxPooling2D(pool_size=2, name="pool2d_2")(x)

    x = keras.layers.SeparableConv2D(
        base_filters * 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="sepconv2d_5",
    )(x)
    x = keras.layers.BatchNormalization(name="bn2d_5")(x)
    x = keras.layers.Activation("swish", name="act2d_5")(x)
    x = keras.layers.SeparableConv2D(
        base_filters * 4,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="sepconv2d_6",
    )(x)
    x = keras.layers.BatchNormalization(name="bn2d_6")(x)
    x = keras.layers.Activation("swish", name="act2d_6")(x)

    x = keras.layers.GlobalAveragePooling2D(name="gap2d")(x)
    x = keras.layers.Dense(head_units, activation="swish", name="head_dense")(x)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    angle_logits = keras.layers.Dense(num_bins, name="angle_logits")(x)
    return keras.Model(inputs=inputs, outputs=angle_logits, name="polar_angle_image")


def _build_model_vote(
    *,
    polar_size: int,
    input_channels: int,
    num_bins: int,
    base_filters: int = 24,
    head_units: int = 96,
    dropout: float = 0.2,
) -> keras.Model:
    """Build a polar CNN that keeps angular position explicit until voting.

    This is intentionally closer to the classical polar accumulator than a
    global-pooling classifier. We only downsample radially, then collapse the
    radius axis with a learned vote pool and predict a circular angular density.
    """
    inputs, x = _build_vote_backbone(
        polar_size=polar_size,
        input_channels=input_channels,
        base_filters=base_filters,
        head_units=head_units,
        dropout=dropout,
    )
    angle_logits = keras.layers.Conv1D(
        1,
        kernel_size=1,
        padding="same",
        name="angle_logits",
    )(x)
    angle_logits = keras.layers.Flatten(name="angle_logits_flat")(angle_logits)
    return keras.Model(inputs=inputs, outputs=angle_logits, name="polar_angle_vote")


def _build_model_ordinal(
    *,
    polar_size: int,
    input_channels: int,
    num_bins: int,
    base_filters: int = 24,
    head_units: int = 96,
    dropout: float = 0.2,
) -> keras.Model:
    """Build an ordinal-threshold polar reader on top of the vote backbone."""
    inputs, x = _build_vote_backbone(
        polar_size=polar_size,
        input_channels=input_channels,
        base_filters=base_filters,
        head_units=head_units,
        dropout=dropout,
    )
    ordinal_logits = keras.layers.Conv1D(
        1,
        kernel_size=1,
        padding="same",
        name="ordinal_logits",
    )(x)
    ordinal_logits = keras.layers.Flatten(name="ordinal_logits_flat")(ordinal_logits)
    return keras.Model(
        inputs=inputs,
        outputs=ordinal_logits,
        name="polar_angle_ordinal",
    )


def _build_model_coarse_to_fine(
    *,
    polar_size: int,
    input_channels: int,
    coarse_bins: int,
    fine_bins: int,
    base_filters: int = 24,
    head_units: int = 96,
    dropout: float = 0.2,
) -> keras.Model:
    """Build a two-head coarse-to-fine polar reader.

    The coarse head learns the broad sector and the fine head learns the local
    residual within that sector. This keeps the supervision ordered without
    forcing the model to solve the full sweep in a single softmax.
    """
    inputs, x = _build_vote_backbone(
        polar_size=polar_size,
        input_channels=input_channels,
        base_filters=base_filters,
        head_units=head_units,
        dropout=dropout,
    )
    x = keras.layers.Flatten(name="structured_flatten")(x)
    x = keras.layers.Dense(head_units, activation="swish", name="structured_dense")(x)
    x = keras.layers.Dropout(dropout, name="structured_dropout")(x)
    coarse_logits = keras.layers.Dense(coarse_bins, name="coarse_logits")(x)

    fine_x = keras.layers.Dense(head_units, activation="swish", name="fine_dense")(x)
    fine_x = keras.layers.Dropout(dropout, name="fine_dropout")(fine_x)
    fine_logits = keras.layers.Dense(fine_bins, name="fine_logits")(fine_x)
    return keras.Model(
        inputs=inputs,
        outputs={
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
        },
        name="polar_angle_coarse_to_fine",
    )


def _build_model_two_stage(
    *,
    polar_size: int,
    input_channels: int,
    coarse_bins: int,
    fine_bins: int,
    base_filters: int = 24,
    head_units: int = 96,
    dropout: float = 0.2,
) -> keras.Model:
    """Build a two-stage polar reader with coarse probabilities feeding stage 2."""
    inputs, x = _build_vote_backbone(
        polar_size=polar_size,
        input_channels=input_channels,
        base_filters=base_filters,
        head_units=head_units,
        dropout=dropout,
    )
    x = keras.layers.Flatten(name="structured_flatten")(x)
    shared = keras.layers.Dense(head_units, activation="swish", name="stage1_dense")(x)
    shared = keras.layers.Dropout(dropout, name="stage1_dropout")(shared)
    coarse_logits = keras.layers.Dense(coarse_bins, name="coarse_logits")(shared)
    coarse_probs = keras.layers.Activation("softmax", name="coarse_probs")(coarse_logits)
    fine_input = keras.layers.Concatenate(name="stage2_concat")([shared, coarse_probs])
    fine_hidden = keras.layers.Dense(head_units, activation="swish", name="stage2_dense")(fine_input)
    fine_hidden = keras.layers.Dropout(dropout, name="stage2_dropout")(fine_hidden)
    fine_logits = keras.layers.Dense(fine_bins, name="fine_logits")(fine_hidden)
    return keras.Model(
        inputs=inputs,
        outputs={
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
        },
        name="polar_angle_two_stage",
    )


def _build_model_vote_broad(
    *,
    polar_size: int,
    input_channels: int,
    num_bins: int,
    base_filters: int = 24,
    head_units: int = 96,
    dropout: float = 0.2,
) -> keras.Model:
    """Build a polar vote model with sharp and broad vote heads.

    The sharp head is the deployable readout, while the broad auxiliary head
    gives the shared backbone a less brittle global target during training.
    """
    inputs = keras.Input(
        shape=(polar_size, polar_size, input_channels),
        name="polar_image",
    )

    x = keras.layers.Conv2D(
        base_filters,
        kernel_size=3,
        strides=(2, 1),
        padding="same",
        use_bias=False,
        name="vote_conv2d_1",
    )(inputs)
    x = keras.layers.BatchNormalization(name="vote_bn2d_1")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_1")(x)

    x = keras.layers.SeparableConv2D(
        base_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_sepconv2d_2",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_2")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_2")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 1), name="vote_pool2d_1")(x)

    x = keras.layers.SeparableConv2D(
        base_filters * 2,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_sepconv2d_3",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_3")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_3")(x)
    x = keras.layers.SeparableConv2D(
        base_filters * 2,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_sepconv2d_4",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_4")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_4")(x)

    radial_mean = keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name="vote_radial_mean",
    )(x)
    radial_max = keras.layers.Lambda(
        lambda t: tf.reduce_max(t, axis=1),
        name="vote_radial_max",
    )(x)
    x = keras.layers.Concatenate(axis=-1, name="vote_radial_fuse")(
        [radial_mean, radial_max]
    )

    x = keras.layers.Conv1D(
        head_units,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="vote_conv1d_1",
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn1d_1")(x)
    x = keras.layers.Activation("swish", name="vote_act1d_1")(x)
    x = keras.layers.Dropout(dropout, name="vote_dropout")(x)

    fine_logits = keras.layers.Conv1D(
        1,
        kernel_size=1,
        padding="same",
        name="fine_angle_logits",
    )(x)
    fine_logits = keras.layers.Flatten(name="fine_angle_logits_flat")(fine_logits)

    broad_logits = keras.layers.Conv1D(
        1,
        kernel_size=1,
        padding="same",
        name="broad_angle_logits",
    )(x)
    broad_logits = keras.layers.Flatten(name="broad_angle_logits_flat")(broad_logits)

    return keras.Model(
        inputs=inputs,
        outputs={
            "fine_angle_logits": fine_logits,
            "broad_angle_logits": broad_logits,
        },
        name="polar_angle_vote_broad",
    )


def _logits_to_temperature(
    logits: np.ndarray,
    *,
    value_min: float,
    value_max: float,
    min_angle_rad: float,
    sweep_rad: float,
) -> np.ndarray:
    """Convert logits over angle bins into Celsius values."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits[None, ...]
    num_bins = int(logits.shape[-1])
    angles = np.linspace(0.0, 2.0 * math.pi, num_bins, endpoint=False, dtype=np.float32)
    shifted = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = shifted / np.sum(shifted, axis=-1, keepdims=True)
    sin_sum = np.sum(probs * np.sin(angles)[None, :], axis=-1)
    cos_sum = np.sum(probs * np.cos(angles)[None, :], axis=-1)
    mean_angle = np.mod(np.arctan2(sin_sum, cos_sum), 2.0 * math.pi)
    sweep_shifted = np.mod(mean_angle - np.float32(min_angle_rad), 2.0 * math.pi)
    fraction = np.clip(sweep_shifted / np.float32(sweep_rad), 0.0, 1.0)
    span = np.float32(value_max - value_min)
    return (np.float32(value_min) + fraction * span).astype(np.float32)


def _logits_to_temperature_sweep(
    logits: np.ndarray,
    *,
    value_min: float,
    value_max: float,
) -> np.ndarray:
    """Convert linear sweep logits into Celsius values."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits[None, ...]
    num_bins = int(logits.shape[-1])
    if num_bins < 2:
        raise ValueError("logits must have at least two bins.")

    shifted = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = shifted / np.sum(shifted, axis=-1, keepdims=True)
    bin_positions = np.linspace(0.0, 1.0, num_bins, dtype=np.float32)
    fraction = np.sum(probs * bin_positions[None, :], axis=-1)
    span = np.float32(value_max - value_min)
    return (np.float32(value_min) + fraction * span).astype(np.float32)


def _structured_logits_to_temperature(
    logits: np.ndarray | dict[str, np.ndarray],
    *,
    structure_mode: Literal["vote", "ordinal", "coarse_to_fine", "two_stage"],
    value_min: float,
    value_max: float,
) -> np.ndarray:
    """Convert structured logits into Celsius values."""
    if structure_mode == "vote":
        if isinstance(logits, dict):
            logits = logits["fine_angle_logits"]
        return _logits_to_temperature_sweep(
            logits,
            value_min=value_min,
            value_max=value_max,
        )
    if structure_mode == "ordinal":
        assert not isinstance(logits, dict)
        return _ordinal_logits_to_temperature(
            logits,
            value_min=value_min,
            value_max=value_max,
        )

    if not isinstance(logits, dict):
        raise TypeError("Structured coarse/fine predictions must be a dict.")
    coarse_logits = logits["coarse_logits"]
    fine_logits = logits["fine_logits"]
    return _coarse_fine_logits_to_temperature(
        coarse_logits,
        fine_logits,
        value_min=value_min,
        value_max=value_max,
    )


def _make_balanced_softmax_loss(
    class_counts: np.ndarray,
    *,
    label_smoothing: float,
) -> keras.losses.Loss:
    """Build a balanced-softmax cross-entropy loss for long-tailed bins."""
    counts = np.asarray(class_counts, dtype=np.float32)
    counts = np.maximum(counts, np.float32(1e-6))
    log_counts = tf.constant(np.log(counts), dtype=tf.float32)
    ce = keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=label_smoothing,
        reduction=keras.losses.Reduction.NONE,
    )

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        adjusted_logits = y_pred + log_counts
        return ce(y_true, adjusted_logits)

    return _loss


def _make_fraction_regularized_loss(
    *,
    base_loss: keras.losses.Loss,
    num_bins: int,
    fraction_loss_weight: float,
    fraction_loss_delta: float,
) -> keras.losses.Loss:
    """Add an ordered fraction-consistency term on top of a base class loss.

    The class target keeps the polar voting geometry, while the auxiliary Huber
    term directly penalizes large sweep-position misses that dominate hard-case
    MAE in edge/corner failures.
    """
    if fraction_loss_weight <= 0.0:
        return base_loss
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2 for fraction regularization.")

    bin_positions = tf.constant(
        np.linspace(0.0, 1.0, num_bins, dtype=np.float32),
        dtype=tf.float32,
    )
    huber = keras.losses.Huber(
        delta=fraction_loss_delta,
        reduction=keras.losses.Reduction.NONE,
    )

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        primary = base_loss(y_true, y_pred)
        probs = tf.nn.softmax(y_pred, axis=-1)
        pred_fraction = tf.reduce_sum(probs * bin_positions[None, :], axis=-1)
        true_fraction = tf.reduce_sum(y_true * bin_positions[None, :], axis=-1)
        fraction_term = huber(true_fraction, pred_fraction)
        return primary + tf.cast(fraction_loss_weight, tf.float32) * fraction_term

    return _loss


def _make_head_losses(
    *,
    loss_mode: Literal["standard", "balanced_softmax"],
    label_smoothing: float,
    fine_class_counts: np.ndarray,
    aux_class_counts: np.ndarray | None = None,
    fine_num_bins: int,
    aux_num_bins: int | None = None,
    fraction_loss_weight: float = DEFAULT_FRACTION_LOSS_WEIGHT,
    fraction_loss_delta: float = DEFAULT_FRACTION_LOSS_DELTA,
) -> tuple[keras.losses.Loss, keras.losses.Loss | None]:
    """Build the fine and optional auxiliary losses for the vote heads."""
    if loss_mode == "balanced_softmax":
        fine_loss = _make_balanced_softmax_loss(
            fine_class_counts,
            label_smoothing=label_smoothing,
        )
        aux_loss = (
            _make_balanced_softmax_loss(
                aux_class_counts if aux_class_counts is not None else fine_class_counts,
                label_smoothing=label_smoothing,
            )
            if aux_class_counts is not None
            else None
        )
    else:
        fine_loss = keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=label_smoothing,
            reduction=keras.losses.Reduction.NONE,
        )
        aux_loss = (
            keras.losses.CategoricalCrossentropy(
                from_logits=True,
                label_smoothing=label_smoothing,
                reduction=keras.losses.Reduction.NONE,
            )
            if aux_class_counts is not None
            else None
        )

    fine_loss = _make_fraction_regularized_loss(
        base_loss=fine_loss,
        num_bins=fine_num_bins,
        fraction_loss_weight=fraction_loss_weight,
        fraction_loss_delta=fraction_loss_delta,
    )
    if aux_loss is not None:
        aux_loss = _make_fraction_regularized_loss(
            base_loss=aux_loss,
            num_bins=aux_num_bins if aux_num_bins is not None else fine_num_bins,
            fraction_loss_weight=fraction_loss_weight * 0.5,
            fraction_loss_delta=fraction_loss_delta,
        )
    return fine_loss, aux_loss


def _compute_metrics(
    values: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Summarize absolute-error metrics for one evaluation split."""
    errors = np.abs(np.asarray(predictions, dtype=np.float32) - np.asarray(values, dtype=np.float32))
    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(np.max(errors)),
        "median_error": float(np.median(errors)),
        "pct_under_5c": float(np.mean(errors < 5.0)),
        "pct_under_3c": float(np.mean(errors < 3.0)),
        "pct_under_1c": float(np.mean(errors < 1.0)),
    }


class TemperatureMaeCallback(keras.callbacks.Callback):
    """Compute validation temperature MAE and push it into epoch logs."""

    def __init__(
        self,
        *,
        val_inputs: np.ndarray,
        val_values: np.ndarray,
        batch_size: int,
        structure_mode: Literal["vote", "ordinal", "coarse_to_fine", "two_stage"],
        value_min: float,
        value_max: float,
    ) -> None:
        super().__init__()
        self._val_inputs = val_inputs
        self._val_values = np.asarray(val_values, dtype=np.float32)
        self._batch_size = int(batch_size)
        self._structure_mode = structure_mode
        self._value_min = float(value_min)
        self._value_max = float(value_max)

    def on_epoch_end(
        self, epoch: int, logs: dict[str, float] | None = None
    ) -> None:
        if logs is None:
            logs = {}
        logits = self.model.predict(
            self._val_inputs,
            batch_size=self._batch_size,
            verbose=0,
        )
        predictions = _structured_logits_to_temperature(
            logits,
            structure_mode=self._structure_mode,
            value_min=self._value_min,
            value_max=self._value_max,
        )
        mae = float(
            np.mean(np.abs(predictions.astype(np.float32) - self._val_values))
        )
        logs["val_temp_mae"] = mae
        print(f"[ANGLE] epoch={epoch + 1} val_temp_mae={mae:.4f}")


def _load_eval_rows(manifest_path: Path, repo_root: Path) -> pd.DataFrame:
    """Load a manifest for post-training evaluation."""
    df = _load_manifest(manifest_path, repo_root)
    if len(df) == 0:
        raise ValueError(f"Manifest is empty after filtering existing images: {manifest_path}")
    return df


def _precompute_arrays(
    df: pd.DataFrame,
    crop_boxes: dict[str, tuple[float, float, float, float]] | None,
    *,
    gauge_id: str,
    polar_size: int,
    num_bins: int,
    sigma_bins: float,
    structure_mode: Literal["vote", "ordinal", "coarse_to_fine", "two_stage"] = "vote",
    target_mode: Literal["circular", "sweep"] = DEFAULT_TARGET_MODE,
    representation: Literal["profile", "image", "vote"] = "profile",
    input_mode: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = "rgb",
    coarse_bins: int | None = None,
    fine_bins: int | None = None,
    aux_num_bins: int | None = None,
    aux_sigma_bins: float | None = None,
    sweep_kernel: Literal["gaussian", "reflect"] = DEFAULT_SWEEP_KERNEL,
    center_search_px: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Precompute polar inputs, primary targets, optional auxiliary targets, values, and weights."""
    spec = load_gauge_specs()[gauge_id]
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    aux_targets: list[np.ndarray] = []
    values: list[np.float32] = []
    weights: list[np.float32] = []

    for index, row in df.iterrows():
        image_path = Path(row["image_path_resolved"])
        crop_box = crop_boxes.get(str(image_path)) if crop_boxes is not None else None
        if representation in {"image", "vote"}:
            representation_input = _crop_and_polar_image(
                image_path,
                crop_box=crop_box,
                polar_size=polar_size,
                input_mode=input_mode,
                center_search_px=center_search_px,
            )
        else:
            representation_input = _crop_and_polar_profile(
                image_path,
                crop_box=crop_box,
                polar_size=polar_size,
            )
        value = float(row["value"])
        fraction = value_to_fraction(value, spec)
        if structure_mode == "ordinal":
            target = _fraction_to_ordinal_target(
                fraction,
                num_thresholds=num_bins,
                sigma_bins=sigma_bins,
                sweep_kernel=sweep_kernel,
            )
        elif structure_mode in {"coarse_to_fine", "two_stage"}:
            if coarse_bins is None or fine_bins is None:
                raise ValueError("coarse_bins and fine_bins are required for structured heads.")
            coarse_target, aux_target = _fraction_to_coarse_fine_targets(
                fraction,
                coarse_bins=coarse_bins,
                fine_bins=fine_bins,
                sigma_bins=sigma_bins,
                sweep_kernel=sweep_kernel,
            )
            target = coarse_target
        else:
            if target_mode == "sweep":
                target = _fraction_to_linear_soft_target(
                    fraction,
                    num_bins=num_bins,
                    sigma_bins=sigma_bins,
                    sweep_kernel=sweep_kernel,
                )
                if aux_num_bins is not None:
                    aux_target = _fraction_to_linear_soft_target(
                        fraction,
                        num_bins=aux_num_bins,
                        sigma_bins=aux_sigma_bins if aux_sigma_bins is not None else sigma_bins,
                        sweep_kernel=sweep_kernel,
                    )
            else:
                angle_rad = fraction_to_angle_rad(fraction, spec)
                target = _angle_to_soft_target(
                    angle_rad,
                    num_bins=num_bins,
                    sigma_bins=sigma_bins,
                )
                if aux_num_bins is not None:
                    aux_target = _angle_to_soft_target(
                        angle_rad,
                        num_bins=aux_num_bins,
                        sigma_bins=aux_sigma_bins if aux_sigma_bins is not None else sigma_bins,
                    )
        inputs.append(representation_input)
        targets.append(target)
        if structure_mode in {"coarse_to_fine", "two_stage"}:
            aux_targets.append(aux_target)
        elif aux_num_bins is not None:
            aux_targets.append(aux_target)
        values.append(np.float32(value))
        sample_weight = float(row["sample_weight"]) if "sample_weight" in df.columns else 1.0
        weights.append(np.float32(sample_weight))

    aux_array: np.ndarray | None
    if structure_mode in {"coarse_to_fine", "two_stage"} or aux_num_bins is not None:
        aux_array = np.asarray(aux_targets, dtype=np.float32)
    else:
        aux_array = None
    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        aux_array,
        np.asarray(values, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
    )


def _make_tf_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    aux_targets: np.ndarray | None,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    num_bins: int,
    structure_mode: Literal["vote", "ordinal", "coarse_to_fine", "two_stage"] = "vote",
    target_mode: Literal["circular", "sweep"] = DEFAULT_TARGET_MODE,
    representation: Literal["profile", "image", "vote"] = "profile",
    input_mode: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = "rgb",
    max_shift_bins: int = 4,
) -> tf.data.Dataset:
    """Build a TensorFlow dataset, optionally adding small circular shifts."""
    if structure_mode in {"coarse_to_fine", "two_stage"}:
        if aux_targets is None:
            raise ValueError("Structured coarse/fine heads require auxiliary targets.")
        ds = tf.data.Dataset.from_tensor_slices(
            (
                inputs,
                {
                    "coarse_logits": targets,
                    "fine_logits": aux_targets,
                },
                weights,
            )
        )
    elif aux_targets is None:
        ds = tf.data.Dataset.from_tensor_slices((inputs, targets, weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices(
            (
                inputs,
                {
                    "fine_angle_logits": targets,
                    "broad_angle_logits": aux_targets,
                },
                weights,
            )
        )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(inputs), reshuffle_each_iteration=True)

    if augment and structure_mode == "vote":
        # Small angular shifts are useful, but sweep-aligned labels must not
        # wrap around because the gauge only occupies a linear arc.
        def _augment(
            representation_input: tf.Tensor,
            target: tf.Tensor | dict[str, tf.Tensor],
            *rest: tf.Tensor,
        ) -> tuple[tf.Tensor, ...]:
            shift = tf.random.uniform([], minval=-max_shift_bins, maxval=max_shift_bins + 1, dtype=tf.int32)

            # Apply board-like photometric perturbations before angular shift.
            if representation in {"image", "vote"}:
                if input_mode == "rgb":
                    representation_input = tf.image.random_brightness(
                        representation_input, max_delta=0.12
                    )
                    representation_input = tf.image.random_contrast(
                        representation_input, lower=0.80, upper=1.25
                    )
                    gamma = tf.random.uniform([], minval=0.85, maxval=1.20)
                    representation_input = tf.pow(
                        tf.clip_by_value(representation_input, 0.0, 1.0),
                        gamma,
                    )
                    representation_input = tf.clip_by_value(representation_input, 0.0, 1.0)
                elif input_mode == "edge3":
                    gain = tf.random.uniform([1, 1, 3], minval=0.90, maxval=1.10)
                    representation_input = tf.clip_by_value(
                        representation_input * gain, 0.0, 1.0
                    )
                elif input_mode == "rgb_edge6":
                    rgb = representation_input[..., :3]
                    edge = representation_input[..., 3:]
                    rgb = tf.image.random_brightness(rgb, max_delta=0.12)
                    rgb = tf.image.random_contrast(rgb, lower=0.82, upper=1.20)
                    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
                    edge_gain = tf.random.uniform([1, 1, 3], minval=0.92, maxval=1.08)
                    edge = tf.clip_by_value(edge * edge_gain, 0.0, 1.0)
                    representation_input = tf.concat([rgb, edge], axis=-1)
                else:
                    rgb = representation_input[..., :3]
                    edge = representation_input[..., 3:6]
                    vote_prior = representation_input[..., 6:7]
                    rgb = tf.image.random_brightness(rgb, max_delta=0.12)
                    rgb = tf.image.random_contrast(rgb, lower=0.82, upper=1.20)
                    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
                    edge_gain = tf.random.uniform([1, 1, 3], minval=0.92, maxval=1.08)
                    edge = tf.clip_by_value(edge * edge_gain, 0.0, 1.0)
                    vote_gain = tf.random.uniform([1, 1, 1], minval=0.95, maxval=1.05)
                    vote_prior = tf.clip_by_value(vote_prior * vote_gain, 0.0, 1.0)
                    representation_input = tf.concat([rgb, edge, vote_prior], axis=-1)
                representation_input = representation_input + tf.random.normal(
                    tf.shape(representation_input), stddev=0.015
                )
                representation_input = tf.clip_by_value(representation_input, 0.0, 1.0)
            if aux_targets is not None:
                target_main = target["fine_angle_logits"]
                target_aux = target["broad_angle_logits"]
            else:
                target_main = target

            def _shift_1d_with_zeros(x: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
                length = tf.shape(x)[0]

                def _shift_right() -> tf.Tensor:
                    padding = tf.stack([[s, 0]])
                    padded = tf.pad(x[: tf.maximum(length - s, 0)], padding)
                    return padded[:length]

                def _shift_left() -> tf.Tensor:
                    left = tf.abs(s)
                    padding = tf.stack([[0, left]])
                    padded = tf.pad(x[left:], padding)
                    return padded[:length]

                return tf.cond(s > 0, _shift_right, _shift_left)

            def _shift_2d_with_zeros(x: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
                width = tf.shape(x)[1]

                def _shift_right() -> tf.Tensor:
                    padding = tf.stack([[0, 0], [s, 0], [0, 0]])
                    padded = tf.pad(x[:, : tf.maximum(width - s, 0), :], padding)
                    return padded[:, :width, :]

                def _shift_left() -> tf.Tensor:
                    left = tf.abs(s)
                    padding = tf.stack([[0, 0], [0, left], [0, 0]])
                    padded = tf.pad(x[:, left:, :], padding)
                    return padded[:, :width, :]

                return tf.cond(s > 0, _shift_right, _shift_left)

            if target_mode == "sweep":
                if representation in {"image", "vote"}:
                    representation_input = _shift_2d_with_zeros(representation_input, shift)
                else:
                    representation_input = _shift_1d_with_zeros(representation_input, shift)
                target_main = _shift_1d_with_zeros(target_main, shift)
                target_sum = tf.reduce_sum(target_main)
                target_main = tf.where(target_sum > 0.0, target_main / target_sum, target_main)
                if aux_targets is not None:
                    aux_target = _shift_1d_with_zeros(target_aux, shift)
                    aux_target_sum = tf.reduce_sum(aux_target)
                    aux_target = tf.where(
                        aux_target_sum > 0.0,
                        aux_target / aux_target_sum,
                        aux_target,
                    )
            else:
                if representation in {"image", "vote"}:
                    # Roll along the angular axis only so the target bin shift stays valid.
                    representation_input = tf.roll(representation_input, shift=shift, axis=1)
                else:
                    representation_input = tf.roll(representation_input, shift=shift, axis=0)
                target_main = tf.roll(target_main, shift=shift, axis=0)
                if aux_targets is not None:
                    aux_target = tf.roll(target_aux, shift=shift, axis=0)

            sample_weight = rest[-1] if aux_targets is not None else rest[0]
            if aux_targets is not None:
                return (
                    representation_input,
                    {
                        "fine_angle_logits": target_main,
                        "broad_angle_logits": aux_target,
                    },
                    sample_weight,
                )
            return representation_input, target_main, sample_weight

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _load_image_path_manifest(manifest_path: Path, repo_root: Path) -> pd.DataFrame:
    """Load and filter one manifest into an evaluation dataframe."""
    df = _load_manifest(manifest_path, repo_root)
    if len(df) == 0:
        raise ValueError(f"Manifest has no loadable rows: {manifest_path}")
    return df


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the polar angle classifier."""
    parser = argparse.ArgumentParser(
        description="Train a polar-profile angle classifier on a CSV manifest."
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--crop-boxes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gauge-id", type=str, default="littlegood_home_temp_gauge_c")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bins", type=int, default=NUM_ANGLE_BINS)
    parser.add_argument("--sigma-bins", type=float, default=SOFT_TARGET_SIGMA_BINS)
    parser.add_argument("--polar-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--head-units", type=int, default=128)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint weights (.weights.h5) to warm-start from. "
            "Useful for continuing a prior polar-vote run with the same model shape."
        ),
    )
    parser.add_argument(
        "--max-shift-bins",
        type=int,
        default=4,
        help=(
            "Maximum angular-bin shift for training augmentation in vote/image "
            "modes. Set 0 to disable angular shifts while keeping photometric aug."
        ),
    )
    parser.add_argument(
        "--center-search-px",
        type=int,
        default=0,
        help=(
            "If > 0, evaluate a 3x3 center-offset grid (0, +/-value pixels) "
            "and keep the polar projection with highest radial continuity score."
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for the angle distribution loss.",
    )
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--representation",
        type=str,
        choices=["profile", "image", "vote"],
        default="profile",
        help="Use a 1D polar profile or the full 2D polar image.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"],
        default="rgb",
        help="Pixel mode for image/vote representations.",
    )
    parser.add_argument(
        "--structure-mode",
        type=str,
        choices=["vote", "ordinal", "coarse_to_fine", "two_stage"],
        default=DEFAULT_STRUCTURE_MODE,
        help="Choose the supervision shape for the polar vote head.",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["circular", "sweep"],
        default=DEFAULT_TARGET_MODE,
        help="Use circular angle bins or linear sweep-aligned bins.",
    )
    parser.add_argument(
        "--sweep-kernel",
        type=str,
        choices=["gaussian", "reflect"],
        default=DEFAULT_SWEEP_KERNEL,
        help=(
            "Soft-label kernel for sweep targets. "
            "'reflect' reduces boundary bias near min/max values."
        ),
    )
    parser.add_argument(
        "--loss-mode",
        type=str,
        choices=["standard", "balanced_softmax"],
        default=DEFAULT_LOSS_MODE,
        help="Use standard cross-entropy or balanced-softmax correction.",
    )
    parser.add_argument(
        "--fraction-loss-weight",
        type=float,
        default=DEFAULT_FRACTION_LOSS_WEIGHT,
        help=(
            "Weight for the auxiliary expected-fraction Huber term added on top "
            "of the classification loss (sweep target mode only)."
        ),
    )
    parser.add_argument(
        "--fraction-loss-delta",
        type=float,
        default=DEFAULT_FRACTION_LOSS_DELTA,
        help="Huber delta for the expected-fraction auxiliary term.",
    )
    parser.add_argument(
        "--aux-head-mode",
        type=str,
        choices=["none", "broad"],
        default=DEFAULT_AUX_HEAD_MODE,
        help="Optionally train a broad auxiliary vote head alongside the sharp head.",
    )
    parser.add_argument(
        "--aux-sigma-multiplier",
        type=float,
        default=DEFAULT_AUX_SIGMA_MULTIPLIER,
        help="How much broader the auxiliary vote target should be than the sharp target.",
    )
    parser.add_argument(
        "--coarse-bins",
        type=int,
        default=DEFAULT_COARSE_BINS,
        help="Number of broad bins for the coarse-to-fine structured heads.",
    )
    parser.add_argument(
        "--fine-bins",
        type=int,
        default=DEFAULT_FINE_BINS,
        help="Number of residual bins inside each coarse sector.",
    )
    parser.add_argument(
        "--extra-eval-manifest",
        action="append",
        default=[],
        help="Additional manifests to evaluate after training.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the angle classifier and report temperature MAE on holdout data."""
    args = _parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_boxes = _load_crop_boxes(args.crop_boxes, REPO_ROOT)
    df = _load_manifest(args.manifest_path, REPO_ROOT)
    spec = load_gauge_specs()[args.gauge_id]
    if len(df) == 0:
        raise ValueError(f"No loadable rows in manifest: {args.manifest_path}")

    # Stratify the split on coarse temperature bins so we keep the full range.
    value_bins = pd.cut(
        df["value"],
        bins=np.arange(
            math.floor(float(df["value"].min()) / STRATIFY_BIN_SIZE) * STRATIFY_BIN_SIZE,
            math.ceil(float(df["value"].max()) / STRATIFY_BIN_SIZE) * STRATIFY_BIN_SIZE
            + STRATIFY_BIN_SIZE,
            STRATIFY_BIN_SIZE,
        ),
        include_lowest=True,
    )
    if value_bins.value_counts().min() < 2:
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=args.seed)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=args.seed)
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=args.seed,
            stratify=value_bins,
        )
        temp_bins = pd.cut(
            temp_df["value"],
            bins=np.arange(
                math.floor(float(df["value"].min()) / STRATIFY_BIN_SIZE) * STRATIFY_BIN_SIZE,
                math.ceil(float(df["value"].max()) / STRATIFY_BIN_SIZE) * STRATIFY_BIN_SIZE
                + STRATIFY_BIN_SIZE,
                STRATIFY_BIN_SIZE,
            ),
            include_lowest=True,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=args.seed,
            stratify=temp_bins if temp_bins.value_counts().min() >= 2 else None,
        )

    print(f"[ANGLE] Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")
    print(
        f"[ANGLE] Representation={args.representation} "
        f"InputMode={args.input_mode} SweepKernel={args.sweep_kernel}"
    )
    if args.input_mode == "rgb_edge6":
        input_channels = 6
    elif args.input_mode == "rgb_edge6_vote7":
        input_channels = 7
    else:
        input_channels = 3

    if args.fraction_loss_weight < 0.0:
        raise ValueError("--fraction-loss-weight must be >= 0.")
    if args.fraction_loss_delta <= 0.0:
        raise ValueError("--fraction-loss-delta must be > 0.")
    if args.fraction_loss_weight > 0.0 and args.target_mode != "sweep":
        raise ValueError("Fraction consistency loss currently supports --target-mode sweep only.")
    if args.fraction_loss_weight > 0.0 and args.structure_mode != "vote":
        raise ValueError("Fraction consistency loss currently supports --structure-mode vote only.")

    if args.structure_mode != "vote" and args.representation != "vote":
        raise ValueError("Structured vote heads require --representation vote.")
    if args.structure_mode in {"coarse_to_fine", "two_stage"} and args.coarse_bins * args.fine_bins != args.bins:
        raise ValueError("For coarse/fine modes, coarse_bins * fine_bins must equal --bins.")

    aux_num_bins = args.bins if args.aux_head_mode == "broad" and args.structure_mode == "vote" else None
    aux_sigma_bins = (
        args.sigma_bins * args.aux_sigma_multiplier
        if args.aux_head_mode == "broad" and args.structure_mode == "vote"
        else None
    )
    use_structure_targets = args.structure_mode in {"ordinal", "coarse_to_fine", "two_stage"}
    coarse_bins = args.coarse_bins if use_structure_targets else None
    fine_bins = args.fine_bins if use_structure_targets else None

    train_profiles, train_targets, train_aux_targets, train_values, train_weights = _precompute_arrays(
        train_df,
        crop_boxes,
        gauge_id=args.gauge_id,
        polar_size=args.polar_size,
        num_bins=args.bins,
        sigma_bins=args.sigma_bins,
        structure_mode=args.structure_mode,
        target_mode=args.target_mode,
        representation=args.representation,
        input_mode=args.input_mode,
        coarse_bins=coarse_bins,
        fine_bins=fine_bins,
        aux_num_bins=aux_num_bins,
        aux_sigma_bins=aux_sigma_bins,
        sweep_kernel=args.sweep_kernel,
        center_search_px=args.center_search_px,
    )
    val_inputs, val_targets, val_aux_targets, val_values, val_weights = _precompute_arrays(
        val_df,
        crop_boxes,
        gauge_id=args.gauge_id,
        polar_size=args.polar_size,
        num_bins=args.bins,
        sigma_bins=args.sigma_bins,
        structure_mode=args.structure_mode,
        target_mode=args.target_mode,
        representation=args.representation,
        input_mode=args.input_mode,
        coarse_bins=coarse_bins,
        fine_bins=fine_bins,
        aux_num_bins=aux_num_bins,
        aux_sigma_bins=aux_sigma_bins,
        sweep_kernel=args.sweep_kernel,
        center_search_px=args.center_search_px,
    )
    test_inputs, test_targets, test_aux_targets, test_values, test_weights = _precompute_arrays(
        test_df,
        crop_boxes,
        gauge_id=args.gauge_id,
        polar_size=args.polar_size,
        num_bins=args.bins,
        sigma_bins=args.sigma_bins,
        structure_mode=args.structure_mode,
        target_mode=args.target_mode,
        representation=args.representation,
        input_mode=args.input_mode,
        coarse_bins=coarse_bins,
        fine_bins=fine_bins,
        aux_num_bins=aux_num_bins,
        aux_sigma_bins=aux_sigma_bins,
        sweep_kernel=args.sweep_kernel,
        center_search_px=args.center_search_px,
    )

    # Balanced softmax uses the effective bin frequencies from the training
    # targets so the rare end bins do not get washed out by the middle.
    train_class_counts = np.sum(train_targets * train_weights[:, None], axis=0)

    train_ds = _make_tf_dataset(
        train_profiles,
        train_targets,
        train_aux_targets,
        train_values,
        train_weights,
        batch_size=args.batch_size,
        shuffle=True,
        augment=args.structure_mode == "vote",
        num_bins=args.bins,
        structure_mode=args.structure_mode,
        target_mode=args.target_mode,
        representation=args.representation,
        input_mode=args.input_mode,
        max_shift_bins=args.max_shift_bins,
    )
    val_ds = _make_tf_dataset(
        val_inputs,
        val_targets,
        val_aux_targets,
        val_values,
        val_weights,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        num_bins=args.bins,
        structure_mode=args.structure_mode,
        target_mode=args.target_mode,
        representation=args.representation,
        input_mode=args.input_mode,
    )
    test_ds = _make_tf_dataset(
        test_inputs,
        test_targets,
        test_aux_targets,
        test_values,
        test_weights,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        num_bins=args.bins,
        structure_mode=args.structure_mode,
        target_mode=args.target_mode,
        representation=args.representation,
        input_mode=args.input_mode,
        max_shift_bins=args.max_shift_bins,
    )

    if args.structure_mode == "ordinal":
        model = _build_model_ordinal(
            polar_size=args.polar_size,
            input_channels=input_channels,
            num_bins=args.bins,
            base_filters=args.base_filters,
            head_units=args.head_units,
            dropout=args.dropout,
        )
    elif args.structure_mode == "coarse_to_fine":
        model = _build_model_coarse_to_fine(
            polar_size=args.polar_size,
            input_channels=input_channels,
            coarse_bins=args.coarse_bins,
            fine_bins=args.fine_bins,
            base_filters=args.base_filters,
            head_units=args.head_units,
            dropout=args.dropout,
        )
    elif args.structure_mode == "two_stage":
        model = _build_model_two_stage(
            polar_size=args.polar_size,
            input_channels=input_channels,
            coarse_bins=args.coarse_bins,
            fine_bins=args.fine_bins,
            base_filters=args.base_filters,
            head_units=args.head_units,
            dropout=args.dropout,
        )
    elif args.representation == "image":
        model = _build_model_2d(
            polar_size=args.polar_size,
            input_channels=input_channels,
            num_bins=args.bins,
            base_filters=args.base_filters,
            head_units=args.head_units,
            dropout=args.dropout,
        )
    elif args.representation == "vote":
        if args.aux_head_mode == "broad":
            model = _build_model_vote_broad(
                polar_size=args.polar_size,
                input_channels=input_channels,
                num_bins=args.bins,
                base_filters=args.base_filters,
                head_units=args.head_units,
                dropout=args.dropout,
            )
        else:
            model = _build_model_vote(
                polar_size=args.polar_size,
                input_channels=input_channels,
                num_bins=args.bins,
                base_filters=args.base_filters,
                head_units=args.head_units,
                dropout=args.dropout,
            )
    else:
        model = _build_model(
            polar_size=args.polar_size,
            num_bins=args.bins,
            base_filters=args.base_filters,
            head_units=args.head_units,
            dropout=args.dropout,
        )
    if args.init_weights is not None:
        if not args.init_weights.exists():
            raise FileNotFoundError(f"Warm-start weights not found: {args.init_weights}")
        print(f"[ANGLE] Loading warm-start weights from {args.init_weights}...")
        model.load_weights(args.init_weights)
        print("[ANGLE] Warm-start weights loaded.")
    if args.structure_mode in {"coarse_to_fine", "two_stage"}:
        if args.structure_mode == "two_stage":
            print("[ANGLE] Using two-stage coarse/fine heads.")
        else:
            print("[ANGLE] Using coarse-to-fine structured heads.")
        loss_fn = {
            "coarse_logits": keras.losses.CategoricalCrossentropy(from_logits=True),
            "fine_logits": keras.losses.CategoricalCrossentropy(from_logits=True),
        }
        loss_weights = {
            "coarse_logits": 1.0,
            "fine_logits": 1.0,
        }
    elif args.structure_mode == "ordinal":
        print("[ANGLE] Using ordinal threshold loss.")
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        loss_weights = None
    elif args.aux_head_mode == "broad":
        fine_class_counts = np.sum(train_targets * train_weights[:, None], axis=0)
        aux_class_counts = np.sum(train_aux_targets * train_weights[:, None], axis=0)
        fine_loss, aux_loss = _make_head_losses(
            loss_mode=args.loss_mode,
            label_smoothing=args.label_smoothing,
            fine_class_counts=fine_class_counts,
            aux_class_counts=aux_class_counts,
            fine_num_bins=args.bins,
            aux_num_bins=args.bins,
            fraction_loss_weight=args.fraction_loss_weight,
            fraction_loss_delta=args.fraction_loss_delta,
        )
        if args.loss_mode == "balanced_softmax":
            print("[ANGLE] Using balanced softmax loss.")
        else:
            print("[ANGLE] Using standard loss with broad auxiliary head.")
        if args.fraction_loss_weight > 0.0:
            print(
                "[ANGLE] Fraction consistency regularizer enabled "
                f"(weight={args.fraction_loss_weight:.3f}, delta={args.fraction_loss_delta:.4f})."
            )
        loss_fn = {
            "fine_angle_logits": fine_loss,
            "broad_angle_logits": aux_loss if aux_loss is not None else fine_loss,
        }
        loss_weights = {
            "fine_angle_logits": 1.0,
            "broad_angle_logits": 0.35,
        }
    else:
        # This is a single-output classifier, so keep compile() aligned with
        # that shape. The dataset yields plain angle targets plus sample weights.
        if args.loss_mode == "balanced_softmax":
            loss_fn = _make_balanced_softmax_loss(
                train_class_counts,
                label_smoothing=args.label_smoothing,
            )
            print("[ANGLE] Using balanced softmax loss.")
        else:
            loss_fn = keras.losses.CategoricalCrossentropy(
                from_logits=True,
                label_smoothing=args.label_smoothing,
                reduction=keras.losses.Reduction.NONE,
            )
        loss_fn = _make_fraction_regularized_loss(
            base_loss=loss_fn,
            num_bins=args.bins,
            fraction_loss_weight=args.fraction_loss_weight,
            fraction_loss_delta=args.fraction_loss_delta,
        )
        if args.fraction_loss_weight > 0.0:
            print(
                "[ANGLE] Fraction consistency regularizer enabled "
                f"(weight={args.fraction_loss_weight:.3f}, delta={args.fraction_loss_delta:.4f})."
            )
        loss_weights = None
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0,
        ),
        loss=loss_fn,
        loss_weights=loss_weights,
        metrics=(
            {"coarse_logits": [keras.metrics.CategoricalAccuracy(name="coarse_acc")], "fine_logits": [keras.metrics.CategoricalAccuracy(name="fine_acc")]}
            if args.structure_mode in {"coarse_to_fine", "two_stage"}
            else (
                {"fine_angle_logits": [keras.metrics.CategoricalAccuracy(name="acc")]}
                if args.aux_head_mode == "broad"
                else ([keras.metrics.BinaryAccuracy(name="bit_acc")] if args.structure_mode == "ordinal" else [keras.metrics.CategoricalAccuracy(name="acc")])
            )
        ),
    )

    temp_metric_callback = TemperatureMaeCallback(
        val_inputs=val_inputs,
        val_values=val_values,
        batch_size=args.batch_size,
        structure_mode=args.structure_mode,
        value_min=spec.min_value,
        value_max=spec.max_value,
    )

    callbacks = [
        temp_metric_callback,
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_dir / "best_weights.weights.h5"),
            monitor="val_temp_mae",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_temp_mae",
            mode="min",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_temp_mae",
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("[ANGLE] Training polar angle classifier...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("[ANGLE] Evaluating held-out test split...")
    test_logits_raw = model.predict(test_inputs, batch_size=args.batch_size, verbose=0)
    test_predictions = _structured_logits_to_temperature(
        test_logits_raw,
        structure_mode=args.structure_mode,
        value_min=spec.min_value,
        value_max=spec.max_value,
    )
    test_metrics = _compute_metrics(test_values, test_predictions)
    for key, value in test_metrics.items():
        if key.startswith("pct_"):
            print(f"[ANGLE] test_{key}={value * 100.0:.2f}%")
        else:
            print(f"[ANGLE] test_{key}={value:.4f}")

    test_out = pd.DataFrame(
        {
            "image_path": test_df["image_path"].values,
            "value": test_values,
            "prediction": test_predictions,
        }
    )
    test_out["abs_error"] = np.abs(test_out["prediction"] - test_out["value"])
    test_out.to_csv(args.output_dir / "test_predictions.csv", index=False)

    model.save(args.output_dir / "model.keras")
    print(f"[ANGLE] Saved model to {args.output_dir / 'model.keras'}")

    history_json = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history_json, handle, indent=2)

    metrics_out = {f"test_{key}": float(value) for key, value in test_metrics.items()}
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_out, handle, indent=2)

    # Optional extra evaluations for hard cases or board captures.
    for extra_manifest in args.extra_eval_manifest:
        extra_path = Path(extra_manifest)
        if not extra_path.is_absolute():
            # Wrapper calls pass ml-relative paths, so prefer the project root
            # first and fall back to the repo root if a caller uses that layout.
            project_relative = PROJECT_ROOT / extra_path
            repo_relative = REPO_ROOT / extra_path
            if project_relative.exists():
                extra_path = project_relative
            else:
                extra_path = repo_relative
        extra_df = _load_image_path_manifest(extra_path, REPO_ROOT)
        extra_inputs, _, _, extra_values, _ = _precompute_arrays(
            extra_df,
            crop_boxes,
            gauge_id=args.gauge_id,
            polar_size=args.polar_size,
            num_bins=args.bins,
            sigma_bins=args.sigma_bins,
            structure_mode=args.structure_mode,
            representation=args.representation,
            input_mode=args.input_mode,
            coarse_bins=coarse_bins,
            fine_bins=fine_bins,
            sweep_kernel=args.sweep_kernel,
            center_search_px=args.center_search_px,
        )
        extra_logits_raw = model.predict(
            extra_inputs,
            batch_size=args.batch_size,
            verbose=0,
        )
        extra_predictions = _structured_logits_to_temperature(
            extra_logits_raw,
            structure_mode=args.structure_mode,
            value_min=spec.min_value,
            value_max=spec.max_value,
        )
        extra_metrics = _compute_metrics(extra_values, extra_predictions)
        stem = extra_path.stem.replace(".", "_")
        print(f"[ANGLE] Extra eval: {extra_path}")
        for key, value in extra_metrics.items():
            if key.startswith("pct_"):
                print(f"[ANGLE] extra_{stem}_{key}={value * 100.0:.2f}%")
            else:
                print(f"[ANGLE] extra_{stem}_{key}={value:.4f}")
        extra_out = pd.DataFrame(
            {
                "image_path": extra_df["image_path"].values,
                "value": extra_values,
                "prediction": extra_predictions,
            }
        )
        extra_out["abs_error"] = np.abs(extra_out["prediction"] - extra_out["value"])
        extra_out.to_csv(args.output_dir / f"{stem}_predictions.csv", index=False)


if __name__ == "__main__":
    main()
