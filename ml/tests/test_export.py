"""Tests for board artifact export helpers."""

from __future__ import annotations

from pathlib import Path

from embedded_gauge_reading_tinyml.export import (
    _resolve_repo_path,
    build_export_metadata,
)


def test_resolve_repo_path_uses_repo_root_for_relative_paths() -> None:
    """Relative export paths should resolve from the repository root."""
    repo_root = Path("/repo")
    resolved = _resolve_repo_path(Path("captured_images/capture_0c_preview.png"), repo_root)
    assert resolved == Path("/repo/captured_images/capture_0c_preview.png")


def test_build_export_metadata_includes_board_input_size() -> None:
    """Export metadata should record the quantization and board input size."""
    metadata = build_export_metadata(
        source_model_path=Path("/repo/model.keras"),
        tflite_path=Path("/repo/model_int8.tflite"),
        input_shape=(1, 224, 224, 3),
        output_shape=(1, 1),
        input_scale=0.5,
        input_zero_point=-128,
        output_scale=0.25,
        output_zero_point=7,
        representative_examples=39,
        hard_case_manifest=Path("/repo/ml/data/hard_cases.csv"),
    )

    assert metadata["board_input_size"] == {"height": 224, "width": 224}
    assert metadata["input_scale"] == 0.5
    assert metadata["output_zero_point"] == 7
    assert metadata["representative_examples"] == 39
