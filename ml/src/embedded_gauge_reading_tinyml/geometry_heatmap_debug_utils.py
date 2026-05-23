"""Shared helpers for geometry heatmap diagnostics and overfit checks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from embedded_gauge_reading_tinyml.geometry_crop_dataset import (
    JitterParams,
    SourceGeometryExample,
    create_jittered_crop,
    load_geometry_manifest,
)
from embedded_gauge_reading_tinyml.heatmap_utils import (
    HeatmapConfig,
    generate_center_tip_heatmaps,
)


def load_clean_geometry_examples(manifest_path: Path) -> list[SourceGeometryExample]:
    """Load the manifest and keep only clean rows."""

    return [
        example
        for example in load_geometry_manifest(manifest_path)
        if example.quality_flag == "clean"
    ]


def sort_examples_by_path(examples: Iterable[SourceGeometryExample]) -> list[SourceGeometryExample]:
    """Return examples sorted deterministically by split and image path."""

    return sorted(examples, key=lambda example: (example.split, example.image_path))


def select_examples_from_split(
    examples: Iterable[SourceGeometryExample],
    *,
    split: str,
    limit: int,
) -> list[SourceGeometryExample]:
    """Select the first `limit` examples from one split in deterministic order."""

    split_examples = sorted(
        [example for example in examples if example.split == split],
        key=lambda example: example.image_path,
    )
    return split_examples[:limit]


def select_balanced_examples(
    examples: Iterable[SourceGeometryExample],
    *,
    total_count: int,
) -> list[SourceGeometryExample]:
    """Select a balanced sample across train/val/test splits."""

    split_order = ("train", "val", "test")
    split_examples = {
        split: sorted(
            [example for example in examples if example.split == split],
            key=lambda example: example.image_path,
        )
        for split in split_order
    }
    per_split = max(1, total_count // len(split_order))
    selected: list[SourceGeometryExample] = []

    for split in split_order:
        selected.extend(split_examples[split][:per_split])

    if len(selected) < total_count:
        for split in split_order:
            for example in split_examples[split][per_split:]:
                if len(selected) >= total_count:
                    break
                selected.append(example)

    return selected[:total_count]


def load_identity_crop(
    example: SourceGeometryExample,
    base_path: Path,
    *,
    input_size: int = 224,
) -> tuple[np.ndarray, dict[str, float], Image.Image]:
    """Load the identity crop for an example and return image plus metadata."""

    jitter = JitterParams(shift_x=0, shift_y=0, scale=1.0, aspect=1.0)
    crop = create_jittered_crop(example, jitter)
    if not crop.accepted:
        raise ValueError(f"Identity crop rejected: {crop.rejection_reason}")

    image_path = base_path / crop.source_image_path
    image = Image.open(image_path).convert("RGB")
    crop_box = (crop.crop_x1, crop.crop_y1, crop.crop_x2, crop.crop_y2)
    crop_image = image.crop(crop_box).resize((input_size, input_size), Image.LANCZOS)
    crop_array = np.asarray(crop_image, dtype=np.float32) / 255.0

    metadata = {
        "center_x_224": crop.center_x_224,
        "center_y_224": crop.center_y_224,
        "tip_x_224": crop.tip_x_224,
        "tip_y_224": crop.tip_y_224,
        "center_x_norm": crop.center_x_normalized,
        "center_y_norm": crop.center_y_normalized,
        "tip_x_norm": crop.tip_x_normalized,
        "tip_y_norm": crop.tip_y_normalized,
        "temperature_c": crop.temperature_c,
        "crop_x1": float(crop.crop_x1),
        "crop_y1": float(crop.crop_y1),
        "crop_x2": float(crop.crop_x2),
        "crop_y2": float(crop.crop_y2),
    }

    return crop_array, metadata, image


def make_target_heatmaps(
    *,
    center_x_norm: float,
    center_y_norm: float,
    tip_x_norm: float,
    tip_y_norm: float,
    heatmap_size: int = 56,
    sigma_pixels: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate center and tip supervision heatmaps."""

    config = HeatmapConfig(
        heatmap_height=heatmap_size,
        heatmap_width=heatmap_size,
        sigma_pixels=sigma_pixels,
    )
    return generate_center_tip_heatmaps(
        center_x_norm,
        center_y_norm,
        tip_x_norm,
        tip_y_norm,
        config=config,
    )


def heatmap_index_to_crop_pixel(
    index_value: float,
    *,
    heatmap_size: int,
    crop_size: int = 224,
) -> float:
    """Map a heatmap index coordinate to crop pixel space."""

    return float(index_value) * float(crop_size) / float(heatmap_size - 1)

