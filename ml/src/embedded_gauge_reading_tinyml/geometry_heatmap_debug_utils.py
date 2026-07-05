"""Small debug helpers shared by the geometry heatmap scripts."""

from __future__ import annotations

from pathlib import Path

from embedded_gauge_reading_tinyml.geometry_crop_dataset import load_geometry_manifest


def heatmap_index_to_crop_pixel(index_value: float, *, heatmap_size: int, crop_size: int = 224) -> float:
    """Map a heatmap index back into crop pixel coordinates."""

    if heatmap_size <= 1:
        return 0.0
    return float(index_value) * float(crop_size - 1) / float(heatmap_size - 1)


def load_clean_geometry_examples(manifest_path: Path):
    """Load the manifest and keep only rows that are not explicitly excluded."""

    return [example for example in load_geometry_manifest(manifest_path) if example.quality_flag == "clean"]

