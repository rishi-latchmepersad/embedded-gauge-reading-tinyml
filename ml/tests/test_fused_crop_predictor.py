"""Tests for crop-fusion path resolution helpers."""

from __future__ import annotations

from pathlib import Path

from embedded_gauge_reading_tinyml.fused_crop_predictor import (
    REPO_ROOT,
    resolve_dataset_image_path,
)


def test_resolve_dataset_image_path_handles_raw_paths() -> None:
    """Raw-image manifests should resolve to the existing file on disk."""
    image_path = resolve_dataset_image_path(
        "ml/data/raw/PXL_20260125_114517176.jpg",
        repo_root=REPO_ROOT,
    )

    assert image_path.exists()
    assert image_path.name == "PXL_20260125_114517176.jpg"


def test_resolve_dataset_image_path_handles_captured_image_paths() -> None:
    """Captured-image manifests should resolve to the repo copy on disk."""
    image_path = resolve_dataset_image_path(
        "ml/data/captured_images/capture_0001.yuv422",
        repo_root=REPO_ROOT,
    )

    assert image_path.exists()
    assert image_path.name == "capture_0001.yuv422"


def test_resolve_dataset_image_path_preserves_absolute_paths(tmp_path: Path) -> None:
    """Already-absolute image paths should pass through unchanged."""
    image_path = tmp_path / "demo.jpg"
    image_path.write_bytes(b"demo")

    resolved = resolve_dataset_image_path(image_path, repo_root=REPO_ROOT)

    assert resolved == image_path
