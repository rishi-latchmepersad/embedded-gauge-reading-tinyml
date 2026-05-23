"""
Geometry crop dataset utilities for the inner Celsius dial reader.

This module provides utilities for loading geometry labels from the manifest,
generating jittered loose crops, and transforming center/tip coordinates
from source-image space into normalized crop coordinates.

Why coordinate transforms matter:
- Labels are annotated in source-image coordinates (e.g., 3472x4624)
- Model trains on crops (e.g., 224x224)
- Center/tip must be transformed correctly or training will learn wrong positions
- This is the most common source of bugs in keypoint-based gauge reading
"""

import csv
import numpy as np
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum


class CropRejectionReason(Enum):
    """Reasons why a jittered crop may be rejected."""
    CROP_OUTSIDE_IMAGE = "crop_outside_image_bounds"
    CENTER_OUTSIDE_CROP = "center_outside_crop"
    TIP_OUTSIDE_CROP = "tip_outside_crop"
    CROP_TOO_SMALL = "crop_too_small"
    CROP_ASPECT_UNREASONABLE = "crop_aspect_unreasonable"
    COORDINATES_NOT_FINITE = "coordinates_not_finite"
    INVALID_JITTER_PARAMS = "invalid_jitter_parameters"


@dataclass
class SourceGeometryExample:
    """
    Represents one geometry-labeled example from the manifest.

    All coordinates are in source-image space (original image coordinates).
    """
    image_path: str
    temperature_c: float
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
    split: str
    label_quality: str = "manual"
    source_manifest: str = ""
    notes: str = ""
    quality_flag: str = "clean"


@dataclass
class JitteredCrop:
    """
    Represents a jittered crop with transformed coordinates.

    Normalized coordinates are in [0, 1] relative to the crop.
    Pixel coordinates are relative to the 224x224 resized crop.
    """
    # Source image info
    source_image_path: str
    split: str
    temperature_c: float
    
    # Crop bounds in source-image coordinates
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    
    # Normalized coordinates in [0, 1]
    center_x_normalized: float
    center_y_normalized: float
    tip_x_normalized: float
    tip_y_normalized: float
    
    # Pixel coordinates in 224x224 crop
    center_x_224: float
    center_y_224: float
    tip_x_224: float
    tip_y_224: float
    
    # Jitter parameters used
    jitter_shift_x: int
    jitter_shift_y: int
    jitter_scale: float
    jitter_aspect: float
    
    # Validity
    accepted: bool
    rejection_reason: Optional[str] = None
    
    # Derived quantities
    angle_degrees: Optional[float] = None
    deterministic_temperature_c: Optional[float] = None
    absolute_temperature_difference_c: Optional[float] = None


@dataclass
class JitterParams:
    """
    Parameters for jittering a loose crop.

    For v1, we prioritize geometric jitter (shift, scale, aspect)
    over photometric jitter (brightness, blur, noise).
    """
    shift_x: int = 0
    shift_y: int = 0
    scale: float = 1.0
    aspect: float = 1.0


def load_geometry_manifest(manifest_path: Path) -> List[SourceGeometryExample]:
    """
    Load geometry examples from the manifest CSV.

    Args:
        manifest_path: Path to geometry_reader_manifest_v1.csv

    Returns:
        List of SourceGeometryExample objects
    """
    examples = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing critical fields
            if not row.get('image_path'):
                continue
            if not row.get('temperature_c'):
                continue
            if not row.get('center_x_source') or not row.get('center_y_source'):
                continue
            if not row.get('tip_x_source') or not row.get('tip_y_source'):
                continue
            
            try:
                example = SourceGeometryExample(
                    image_path=row['image_path'],
                    temperature_c=float(row['temperature_c']),
                    source_width=int(row['source_width']),
                    source_height=int(row['source_height']),
                    loose_crop_x1=int(float(row['loose_crop_x1'])),
                    loose_crop_y1=int(float(row['loose_crop_y1'])),
                    loose_crop_x2=int(float(row['loose_crop_x2'])),
                    loose_crop_y2=int(float(row['loose_crop_y2'])),
                    center_x_source=float(row['center_x_source']),
                    center_y_source=float(row['center_y_source']),
                    tip_x_source=float(row['tip_x_source']),
                    tip_y_source=float(row['tip_y_source']),
                    dial_radius_source=float(row['dial_radius_source']),
                    split=row['split'],
                    label_quality=row.get('label_quality', 'unknown'),
                    source_manifest=row.get('source_manifest', ''),
                    notes=row.get('notes', ''),
                    quality_flag=row.get('quality_flag', 'clean'),
                )
                examples.append(example)
            except (ValueError, KeyError) as e:
                # Skip malformed rows
                continue
    
    return examples


def generate_jitter_params(
    rng: np.random.Generator,
    shift_range: int = 20,
    scale_range: Tuple[float, float] = (0.85, 1.25),
    aspect_range: Tuple[float, float] = (0.90, 1.10),
) -> JitterParams:
    """
    Generate random jitter parameters.

    Jitter ranges (v1 defaults):
    - x shift: -20 to +20 px
    - y shift: -20 to +20 px
    - scale: 0.85 to 1.25
    - aspect ratio: 0.90 to 1.10
    - rotation: omitted for v1 (documented below)
    - brightness/blur/noise: omitted for v1

    Why rotation is omitted in v1:
    - Rotation requires careful handling of coordinate transforms
    - Inner dial rotation is already captured by the needle angle
    - Adding rotation jitter complicates the transform chain
    - Can be added in v2 once the base pipeline is verified

    Args:
        rng: Random number generator (for reproducibility)
        shift_range: Maximum shift in pixels
        scale_range: (min, max) scale factor
        aspect_range: (min, max) aspect ratio multiplier

    Returns:
        JitterParams object
    """
    shift_x = rng.integers(-shift_range, shift_range)
    shift_y = rng.integers(-shift_range, shift_range)
    scale = rng.uniform(scale_range[0], scale_range[1])
    aspect = rng.uniform(aspect_range[0], aspect_range[1])
    
    return JitterParams(
        shift_x=shift_x,
        shift_y=shift_y,
        scale=scale,
        aspect=aspect,
    )


def apply_jitter_to_crop(
    example: SourceGeometryExample,
    jitter: JitterParams,
) -> Tuple[int, int, int, int]:
    """
    Apply jitter parameters to the loose crop box.

    The jitter is applied as follows:
    1. Shift: translate the crop box by (shift_x, shift_y)
    2. Scale: scale the crop dimensions around the center
    3. Aspect: adjust width relative to height

    Args:
        example: Source geometry example
        jitter: Jitter parameters

    Returns:
        (crop_x1, crop_y1, crop_x2, crop_y2) in source-image coordinates
    """
    # Original crop bounds
    x1 = example.loose_crop_x1
    y1 = example.loose_crop_y1
    x2 = example.loose_crop_x2
    y2 = example.loose_crop_y2
    
    # Original dimensions
    width = x2 - x1
    height = y2 - y1
    
    # Crop center
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    # Apply shift
    x1 += jitter.shift_x
    y1 += jitter.shift_y
    x2 += jitter.shift_x
    y2 += jitter.shift_y
    
    # Apply scale (around the new center)
    new_center_x = (x1 + x2) / 2.0
    new_center_y = (y1 + y2) / 2.0
    new_width = width * jitter.scale
    new_height = height * jitter.scale
    
    # Apply aspect ratio adjustment (adjust width only)
    new_width *= jitter.aspect
    
    # Compute new bounds
    crop_x1 = int(new_center_x - new_width / 2.0)
    crop_y1 = int(new_center_y - new_height / 2.0)
    crop_x2 = int(new_center_x + new_width / 2.0)
    crop_y2 = int(new_center_y + new_height / 2.0)
    
    return (crop_x1, crop_y1, crop_x2, crop_y2)


def transform_point_to_crop(
    point_x_source: float,
    point_y_source: float,
    crop_x1: int,
    crop_y1: int,
    crop_x2: int,
    crop_y2: int,
) -> Tuple[float, float]:
    """
    Transform a point from source-image coordinates to normalized crop coordinates.

    Normalized coordinates are in [0, 1] where:
    - (0, 0) is the top-left of the crop
    - (1, 1) is the bottom-right of the crop

    Args:
        point_x_source: X coordinate in source-image space
        point_y_source: Y coordinate in source-image space
        crop_x1, crop_y1: Top-left of crop in source-image space
        crop_x2, crop_y2: Bottom-right of crop in source-image space

    Returns:
        (x_normalized, y_normalized) in [0, 1] range
    """
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    
    # Normalize to [0, 1]
    x_norm = (point_x_source - crop_x1) / crop_width
    y_norm = (point_y_source - crop_y1) / crop_height
    
    return (x_norm, y_norm)


def transform_normalized_to_224(
    x_normalized: float,
    y_normalized: float,
    target_size: int = 224,
) -> Tuple[float, float]:
    """
    Transform normalized coordinates to pixel coordinates in a 224x224 crop.

    Args:
        x_normalized: X coordinate in [0, 1]
        y_normalized: Y coordinate in [0, 1]
        target_size: Target crop size (default 224)

    Returns:
        (x_224, y_224) pixel coordinates
    """
    x_224 = x_normalized * target_size
    y_224 = y_normalized * target_size
    return (x_224, y_224)


def validate_crop(
    example: SourceGeometryExample,
    crop_x1: int,
    crop_y1: int,
    crop_x2: int,
    crop_y2: int,
    center_x_norm: float,
    center_y_norm: float,
    tip_x_norm: float,
    tip_y_norm: float,
) -> Tuple[bool, Optional[CropRejectionReason]]:
    """
    Validate a jittered crop and its transformed coordinates.

    Rejection criteria:
    - Crop leaves source image bounds
    - Center or tip falls outside the crop (normalized coords not in [0, 1])
    - Crop width or height is too small (< 32 px)
    - Crop aspect ratio is unreasonable (< 0.5 or > 2.0)
    - Transformed coordinates are not finite

    Args:
        example: Source geometry example
        crop_x1, crop_y1, crop_x2, crop_y2: Crop bounds in source-image space
        center_x_norm, center_y_norm: Normalized center coordinates
        tip_x_norm, tip_y_norm: Normalized tip coordinates

    Returns:
        (accepted, rejection_reason) tuple
    """
    # Check crop is within source image bounds
    if crop_x1 < 0 or crop_y1 < 0:
        return (False, CropRejectionReason.CROP_OUTSIDE_IMAGE)
    if crop_x2 > example.source_width or crop_y2 > example.source_height:
        return (False, CropRejectionReason.CROP_OUTSIDE_IMAGE)
    
    # Check crop dimensions
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    
    if crop_width < 32 or crop_height < 32:
        return (False, CropRejectionReason.CROP_TOO_SMALL)
    
    # Check crop aspect ratio
    aspect_ratio = crop_width / crop_height
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return (False, CropRejectionReason.CROP_ASPECT_UNREASONABLE)
    
    # Check center is inside crop
    if not (0.0 <= center_x_norm <= 1.0 and 0.0 <= center_y_norm <= 1.0):
        return (False, CropRejectionReason.CENTER_OUTSIDE_CROP)
    
    # Check tip is inside crop
    if not (0.0 <= tip_x_norm <= 1.0 and 0.0 <= tip_y_norm <= 1.0):
        return (False, CropRejectionReason.TIP_OUTSIDE_CROP)
    
    # Check coordinates are finite
    import math
    coords = [center_x_norm, center_y_norm, tip_x_norm, tip_y_norm]
    if any(not math.isfinite(c) for c in coords):
        return (False, CropRejectionReason.COORDINATES_NOT_FINITE)
    
    return (True, None)


def create_jittered_crop(
    example: SourceGeometryExample,
    jitter: JitterParams,
) -> JitteredCrop:
    """
    Create a jittered crop from a source geometry example.

    This is the main entry point for generating augmented training samples.

    Args:
        example: Source geometry example
        jitter: Jitter parameters

    Returns:
        JitteredCrop object (accepted=False if validation fails)
    """
    # Import geometry utilities for angle/temperature computation
    from embedded_gauge_reading_tinyml.gauge_geometry import (
        angle_degrees_from_center_to_tip,
        celsius_from_inner_dial_angle_degrees,
    )
    
    # Apply jitter to get crop bounds
    crop_x1, crop_y1, crop_x2, crop_y2 = apply_jitter_to_crop(example, jitter)
    
    # Transform center to normalized crop coordinates
    center_x_norm, center_y_norm = transform_point_to_crop(
        example.center_x_source,
        example.center_y_source,
        crop_x1, crop_y1,
        crop_x2, crop_y2,
    )
    
    # Transform tip to normalized crop coordinates
    tip_x_norm, tip_y_norm = transform_point_to_crop(
        example.tip_x_source,
        example.tip_y_source,
        crop_x1, crop_y1,
        crop_x2, crop_y2,
    )
    
    # Validate crop
    accepted, rejection_reason = validate_crop(
        example,
        crop_x1, crop_y1, crop_x2, crop_y2,
        center_x_norm, center_y_norm,
        tip_x_norm, tip_y_norm,
    )
    
    # Transform to 224x224 coordinates
    center_x_224, center_y_224 = transform_normalized_to_224(center_x_norm, center_y_norm)
    tip_x_224, tip_y_224 = transform_normalized_to_224(tip_x_norm, tip_y_norm)
    
    # Compute angle and deterministic temperature (if accepted)
    angle_degrees = None
    deterministic_temp = None
    temp_diff = None
    
    if accepted:
        # Compute angle from center/tip in crop coordinates
        # Note: We use the 224x224 coordinates for angle computation
        # This is equivalent to using normalized coords (linear transform)
        angle_degrees = angle_degrees_from_center_to_tip(
            center_x_224, center_y_224,
            tip_x_224, tip_y_224,
        )
        
        # Compute deterministic temperature from angle
        deterministic_temp = celsius_from_inner_dial_angle_degrees(angle_degrees)
        
        # Compute temperature difference
        temp_diff = abs(deterministic_temp - example.temperature_c)
    
    return JitteredCrop(
        source_image_path=example.image_path,
        split=example.split,
        temperature_c=example.temperature_c,
        crop_x1=crop_x1,
        crop_y1=crop_y1,
        crop_x2=crop_x2,
        crop_y2=crop_y2,
        center_x_normalized=center_x_norm,
        center_y_normalized=center_y_norm,
        tip_x_normalized=tip_x_norm,
        tip_y_normalized=tip_y_norm,
        center_x_224=center_x_224,
        center_y_224=center_y_224,
        tip_x_224=tip_x_224,
        tip_y_224=tip_y_224,
        jitter_shift_x=jitter.shift_x,
        jitter_shift_y=jitter.shift_y,
        jitter_scale=jitter.scale,
        jitter_aspect=jitter.aspect,
        accepted=accepted,
        rejection_reason=rejection_reason.value if rejection_reason else None,
        angle_degrees=angle_degrees,
        deterministic_temperature_c=deterministic_temp,
        absolute_temperature_difference_c=temp_diff,
    )


def generate_jittered_crops_for_example(
    example: SourceGeometryExample,
    num_crops: int = 5,
    seed: Optional[int] = None,
) -> List[JitteredCrop]:
    """
    Generate multiple jittered crops for a single source example.

    Args:
        example: Source geometry example
        num_crops: Number of jittered crops to generate
        seed: Random seed for reproducibility

    Returns:
        List of JitteredCrop objects (some may be rejected)
    """
    rng = np.random.default_rng(seed)
    crops = []
    
    for i in range(num_crops):
        jitter = generate_jitter_params(rng)
        crop = create_jittered_crop(example, jitter)
        crops.append(crop)
    
    return crops


def load_and_generate_crops(
    manifest_path: Path,
    num_crops_per_example: int = 5,
    seed: Optional[int] = 42,
    max_examples: Optional[int] = None,
) -> Tuple[List[JitteredCrop], Dict[str, int]]:
    """
    Load manifest and generate jittered crops for all examples.

    Args:
        manifest_path: Path to geometry_reader_manifest_v1.csv
        num_crops_per_example: Number of jittered crops per example
        seed: Random seed for reproducibility
        max_examples: Maximum number of examples to process (None = all)

    Returns:
        (all_crops, rejection_counts) tuple
    """
    # Load examples
    examples = load_geometry_manifest(manifest_path)
    
    if max_examples:
        examples = examples[:max_examples]
    
    # Generate crops
    all_crops = []
    rejection_counts: Dict[str, int] = {}
    
    for i, example in enumerate(examples):
        # Use deterministic seed per example
        example_seed = seed + i if seed is not None else None
        crops = generate_jittered_crops_for_example(
            example,
            num_crops=num_crops_per_example,
            seed=example_seed,
        )
        
        for crop in crops:
            all_crops.append(crop)
            
            # Count rejections
            if not crop.accepted and crop.rejection_reason:
                rejection_counts[crop.rejection_reason] = \
                    rejection_counts.get(crop.rejection_reason, 0) + 1
    
    return (all_crops, rejection_counts)
