"""Geometry crop manifest types and helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from embedded_gauge_reading_tinyml.gauge_geometry import (
    angle_degrees_from_center_to_tip,
    celsius_from_inner_dial_angle_degrees,
)


@dataclass(frozen=True, slots=True)
class SourceGeometryExample:
    """One labeled image row from a geometry manifest."""

    image_path: str
    temperature_c: float
    split: str
    source_width: int
    source_height: int
    loose_crop_x1: int
    loose_crop_y1: int
    loose_crop_x2: int
    loose_crop_y2: int
    center_x_source: float
    center_y_source: float
    tip_x_source: float
    tip_y_source: float
    dial_radius_source: float
    label_quality: str
    source_manifest: str
    notes: str
    angle_degrees_from_labels: float
    deterministic_temperature_c: float
    absolute_temperature_difference_c: float
    center_tip_distance_pixels: float
    quality_flag: str

    @property
    def source_image_path(self) -> str:
        """Return the source image path using the name expected by helpers."""

        return self.image_path


@dataclass(frozen=True, slots=True)
class JitterParams:
    """A simple crop jitter configuration."""

    shift_x: int = 0
    shift_y: int = 0
    scale: float = 1.0
    aspect: float = 1.0


def generate_jitter_params(
    rng: random.Random,
    *,
    max_shift_px: int = 16,
    scale_min: float = 0.92,
    scale_max: float = 1.08,
    aspect_min: float = 0.92,
    aspect_max: float = 1.08,
) -> JitterParams:
    """Sample a small crop jitter that stays close to the annotated box.

    The legacy geometry point trainers expect this helper to exist, and the
    sampled jitter is intentionally conservative so the model learns mild
    robustness without drifting off the labeled gauge region.
    """

    shift_range = int(max(0, max_shift_px))
    return JitterParams(
        shift_x=int(rng.randint(-shift_range, shift_range)),
        shift_y=int(rng.randint(-shift_range, shift_range)),
        scale=float(rng.uniform(scale_min, scale_max)),
        aspect=float(rng.uniform(aspect_min, aspect_max)),
    )


@dataclass(frozen=True, slots=True)
class JitteredCrop:
    """The crop coordinates and derived labels after jittering."""

    source_image_path: str
    split: str
    temperature_c: float
    source_manifest: str
    source_width: int
    source_height: int
    quality_flag: str
    dial_radius_source: float
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    jitter_shift_x: int
    jitter_shift_y: int
    jitter_scale: float
    jitter_aspect: float
    center_x_normalized: float
    center_y_normalized: float
    tip_x_normalized: float
    tip_y_normalized: float
    center_x_224: float
    center_y_224: float
    tip_x_224: float
    tip_y_224: float
    angle_degrees: float
    deterministic_temperature_c: float
    absolute_temperature_difference_c: float
    accepted: bool


def _resolve_path(manifest_path: Path, raw_path: str) -> str:
    """Normalize a manifest image path."""

    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "ml":
        return str(manifest_path.parent.parent.parent / path)
    return str(manifest_path.parent.parent.parent / "ml" / path)


def load_geometry_manifest(manifest_path: Path) -> list[SourceGeometryExample]:
    """Load geometry-style samples from CSV."""

    import csv

    samples: list[SourceGeometryExample] = []
    if not manifest_path.exists():
        return samples
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            quality = str(row.get("quality_flag", "")).strip().lower()
            if quality == "exclude":
                continue
            try:
                image_path = _resolve_path(manifest_path, str(row["image_path"]).strip())
                samples.append(
                    SourceGeometryExample(
                        image_path=image_path,
                        temperature_c=float(row.get("temperature_c", "nan")),
                        split=str(row.get("split", "train")),
                        source_width=int(float(row["source_width"])),
                        source_height=int(float(row["source_height"])),
                        loose_crop_x1=int(float(row["loose_crop_x1"])),
                        loose_crop_y1=int(float(row["loose_crop_y1"])),
                        loose_crop_x2=int(float(row["loose_crop_x2"])),
                        loose_crop_y2=int(float(row["loose_crop_y2"])),
                        center_x_source=float(row["center_x_source"]),
                        center_y_source=float(row["center_y_source"]),
                        tip_x_source=float(row["tip_x_source"]),
                        tip_y_source=float(row["tip_y_source"]),
                        dial_radius_source=float(row.get("dial_radius_source", 0.0)),
                        label_quality=str(row.get("label_quality", "")),
                        source_manifest=str(row.get("source_manifest", "")),
                        notes=str(row.get("notes", "")),
                        angle_degrees_from_labels=float(row.get("angle_degrees_from_labels", "nan")),
                        deterministic_temperature_c=float(row.get("deterministic_temperature_c", "nan")),
                        absolute_temperature_difference_c=float(row.get("absolute_temperature_difference_c", "nan")),
                        center_tip_distance_pixels=float(row.get("center_tip_distance_pixels", "nan")),
                        quality_flag=str(row.get("quality_flag", "clean")),
                    )
                )
            except (TypeError, ValueError, KeyError):
                continue
    return samples


def _make_crop_box(
    example: SourceGeometryExample,
    jitter: JitterParams,
) -> tuple[int, int, int, int]:
    """Build a jittered crop box in source-image coordinates."""

    x1 = float(example.loose_crop_x1)
    y1 = float(example.loose_crop_y1)
    x2 = float(example.loose_crop_x2)
    y2 = float(example.loose_crop_y2)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = x1 + 0.5 * width
    center_y = y1 + 0.5 * height

    jitter_width = width * float(jitter.scale) * float(jitter.aspect)
    jitter_height = height * float(jitter.scale) / max(float(jitter.aspect), 1e-6)
    jitter_center_x = center_x + float(jitter.shift_x)
    jitter_center_y = center_y + float(jitter.shift_y)

    crop_x1 = int(round(jitter_center_x - 0.5 * jitter_width))
    crop_y1 = int(round(jitter_center_y - 0.5 * jitter_height))
    crop_x2 = int(round(jitter_center_x + 0.5 * jitter_width))
    crop_y2 = int(round(jitter_center_y + 0.5 * jitter_height))

    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(example.source_width, max(crop_x1 + 1, crop_x2))
    crop_y2 = min(example.source_height, max(crop_y1 + 1, crop_y2))
    return crop_x1, crop_y1, crop_x2, crop_y2


def create_jittered_crop(
    example: SourceGeometryExample,
    jitter: JitterParams,
) -> JitteredCrop:
    """Create a jittered crop and its normalized labels."""

    crop_x1, crop_y1, crop_x2, crop_y2 = _make_crop_box(example, jitter)
    crop_width = max(1.0, float(crop_x2 - crop_x1))
    crop_height = max(1.0, float(crop_y2 - crop_y1))

    center_x_norm = (float(example.center_x_source) - float(crop_x1)) / crop_width
    center_y_norm = (float(example.center_y_source) - float(crop_y1)) / crop_height
    tip_x_norm = (float(example.tip_x_source) - float(crop_x1)) / crop_width
    tip_y_norm = (float(example.tip_y_source) - float(crop_y1)) / crop_height
    accepted = all(0.0 <= value <= 1.0 for value in (center_x_norm, center_y_norm, tip_x_norm, tip_y_norm))

    center_x_224 = center_x_norm * 223.0
    center_y_224 = center_y_norm * 223.0
    tip_x_224 = tip_x_norm * 223.0
    tip_y_224 = tip_y_norm * 223.0
    angle_degrees = angle_degrees_from_center_to_tip(center_x_224, center_y_224, tip_x_224, tip_y_224)
    deterministic_temperature_c = celsius_from_inner_dial_angle_degrees(angle_degrees)

    return JitteredCrop(
        source_image_path=example.image_path,
        split=example.split,
        temperature_c=example.temperature_c,
        source_manifest=example.source_manifest,
        source_width=example.source_width,
        source_height=example.source_height,
        quality_flag=example.quality_flag,
        dial_radius_source=example.dial_radius_source,
        crop_x1=crop_x1,
        crop_y1=crop_y1,
        crop_x2=crop_x2,
        crop_y2=crop_y2,
        jitter_shift_x=jitter.shift_x,
        jitter_shift_y=jitter.shift_y,
        jitter_scale=jitter.scale,
        jitter_aspect=jitter.aspect,
        center_x_normalized=center_x_norm,
        center_y_normalized=center_y_norm,
        tip_x_normalized=tip_x_norm,
        tip_y_normalized=tip_y_norm,
        center_x_224=center_x_224,
        center_y_224=center_y_224,
        tip_x_224=tip_x_224,
        tip_y_224=tip_y_224,
        angle_degrees=angle_degrees,
        deterministic_temperature_c=deterministic_temperature_c,
        absolute_temperature_difference_c=abs(deterministic_temperature_c - float(example.temperature_c)),
        accepted=accepted,
    )
