"""Tests for the manifest builder script.

This module contains pytest tests for the build_canonical_manifest.py script,
covering path normalization, duplicate resolution, conflict detection,
missing file filtering, and output schema correctness.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Import the module under test
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_canonical_manifest as builder


class TestNormalizePath:
    """Tests for path normalization behavior."""

    def test_forward_slashes(self, tmp_path):
        """Test that backslashes are converted to forward slashes."""
        repo_root = tmp_path
        input_path = "ml\\data\\raw\\image.jpg"
        result = builder.normalize_path(input_path, repo_root)
        assert "/" in result
        assert "\\" not in result

    def test_relative_path_preserved(self, tmp_path):
        """Test that relative paths stay relative."""
        repo_root = tmp_path
        input_path = "ml/data/raw/image.jpg"
        result = builder.normalize_path(input_path, repo_root)
        assert result == "ml/data/raw/image.jpg"

    def test_absolute_path_under_repo(self, tmp_path):
        """Test that absolute paths under repo root are made relative."""
        repo_root = tmp_path
        input_path = str(tmp_path / "ml" / "data" / "raw" / "image.jpg")
        result = builder.normalize_path(input_path, repo_root)
        assert result == "ml/data/raw/image.jpg"

    def test_absolute_path_outside_repo(self, tmp_path):
        """Test that absolute paths outside repo root are preserved."""
        repo_root = tmp_path
        input_path = "/some/other/path/image.jpg"
        result = builder.normalize_path(input_path, repo_root)
        assert result == "/some/other/path/image.jpg"


class TestLoadManifest:
    """Tests for manifest loading functionality."""

    def test_load_basic_csv(self, tmp_path):
        """Test loading a basic CSV with image_path and label columns."""
        repo_root = tmp_path
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a test CSV
        csv_path = data_dir / "test_manifest.csv"
        df = pd.DataFrame(
            {
                "image_path": ["ml/data/raw/image1.jpg", "ml/data/raw/image2.jpg"],
                "label": [-29.0, 30.0],
            }
        )
        df.to_csv(csv_path, index=False)

        # Create dummy image files
        (tmp_path / "ml" / "data" / "raw").mkdir(parents=True)
        (tmp_path / "ml" / "data" / "raw" / "image1.jpg").touch()
        (tmp_path / "ml" / "data" / "raw" / "image2.jpg").touch()

        result = builder.load_manifest(csv_path, repo_root, "test_manifest")

        assert result is not None
        assert len(result) == 2
        assert "value" in result.columns
        assert "source_tag" in result.columns

    def test_load_csv_with_value_column(self, tmp_path):
        """Test loading a CSV that already uses 'value' instead of 'label'."""
        repo_root = tmp_path
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        csv_path = data_dir / "test_manifest.csv"
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg"],
                "value": [25.0],
            }
        )
        df.to_csv(csv_path, index=False)

        (tmp_path / "image1.jpg").touch()

        result = builder.load_manifest(csv_path, repo_root, "test_manifest")

        assert result is not None
        assert result["value"].iloc[0] == 25.0

    def test_missing_file_returns_none(self, tmp_path):
        """Test that loading a non-existent file returns None."""
        repo_root = tmp_path
        csv_path = tmp_path / "nonexistent.csv"

        result = builder.load_manifest(csv_path, repo_root, "test")

        assert result is None

    def test_empty_file_returns_none(self, tmp_path):
        """Test that loading an empty CSV returns None."""
        repo_root = tmp_path
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")

        result = builder.load_manifest(csv_path, repo_root, "test")

        assert result is None


class TestFilterValidRows:
    """Tests for filtering rows with missing files or invalid labels."""

    def test_filter_missing_files(self, tmp_path):
        """Test that rows with missing files are filtered out."""
        repo_root = tmp_path

        # Create a DataFrame with one existing and one missing file
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg", "image2.jpg"],
                "value": [25.0, 30.0],
                "source_tag": ["core", "core"],
                "hardness_tag": ["normal", "normal"],
                "crop_x_min": [pd.NA, pd.NA],
                "crop_y_min": [pd.NA, pd.NA],
                "crop_x_max": [pd.NA, pd.NA],
                "crop_y_max": [pd.NA, pd.NA],
                "source_file": ["test", "test"],
            }
        )

        # Create only the first image file
        (tmp_path / "image1.jpg").touch()

        result = builder.filter_valid_rows(df, repo_root)

        assert len(result) == 1
        assert result["image_path"].iloc[0] == "image1.jpg"

    def test_filter_invalid_labels(self, tmp_path):
        """Test that rows with non-numeric labels are filtered out."""
        repo_root = tmp_path

        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg", "image2.jpg", "image3.jpg"],
                "value": [25.0, "invalid", None],
                "source_tag": ["core", "core", "core"],
                "hardness_tag": ["normal", "normal", "normal"],
                "crop_x_min": [pd.NA, pd.NA, pd.NA],
                "crop_y_min": [pd.NA, pd.NA, pd.NA],
                "crop_x_max": [pd.NA, pd.NA, pd.NA],
                "crop_y_max": [pd.NA, pd.NA, pd.NA],
                "source_file": ["test", "test", "test"],
            }
        )

        (tmp_path / "image1.jpg").touch()
        (tmp_path / "image2.jpg").touch()
        (tmp_path / "image3.jpg").touch()

        result = builder.filter_valid_rows(df, repo_root)

        assert len(result) == 1
        assert result["value"].iloc[0] == 25.0


class TestDeduplicateAndResolveConflicts:
    """Tests for duplicate resolution and conflict detection."""

    def test_no_duplicates_kept_as_is(self):
        """Test that rows without duplicates are kept as-is."""
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg", "image2.jpg"],
                "value": [25.0, 30.0],
                "source_file": ["source1", "source1"],
                "source_tag": ["core", "core"],
                "hardness_tag": ["normal", "normal"],
            }
        )

        canonical, conflicts = builder.deduplicate_and_resolve_conflicts(df)

        assert len(canonical) == 2
        assert len(conflicts) == 0

    def test_duplicate_resolution_precedence(self):
        """Test that duplicates are resolved by source precedence."""
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg", "image1.jpg"],
                "value": [25.0, 25.5],  # Within threshold
                "source_file": [
                    "unified_manifest_with_crops_v2",
                    "hard_cases_plus_board30_valid_with_new6",
                ],
                "source_tag": ["core", "hard_case"],
                "hardness_tag": ["normal", "hard_case"],
            }
        )

        canonical, conflicts = builder.deduplicate_and_resolve_conflicts(df)

        assert len(canonical) == 1
        assert len(conflicts) == 0
        # Should keep the higher precedence source
        assert (
            canonical["source_file"].iloc[0]
            == "hard_cases_plus_board30_valid_with_new6"
        )

    def test_conflict_detection(self):
        """Test that label differences above threshold are flagged as conflicts."""
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg", "image1.jpg"],
                "value": [25.0, 30.0],  # Difference of 5.0 > threshold
                "source_file": ["source1", "source2"],
                "source_tag": ["core", "core"],
                "hardness_tag": ["normal", "normal"],
            }
        )

        canonical, conflicts = builder.deduplicate_and_resolve_conflicts(
            df, conflict_threshold=1.0
        )

        assert len(canonical) == 0
        assert len(conflicts) == 2  # Both rows moved to conflicts

    def test_conflict_threshold_custom(self):
        """Test custom conflict threshold."""
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg", "image1.jpg"],
                "value": [25.0, 26.5],  # Difference of 1.5
                "source_file": ["source1", "source2"],
                "source_tag": ["core", "core"],
                "hardness_tag": ["normal", "normal"],
            }
        )

        # With threshold 1.0, should be a conflict
        canonical, conflicts = builder.deduplicate_and_resolve_conflicts(
            df, conflict_threshold=1.0
        )
        assert len(conflicts) == 2

        # With threshold 2.0, should be resolved
        canonical, conflicts = builder.deduplicate_and_resolve_conflicts(
            df, conflict_threshold=2.0
        )
        assert len(canonical) == 1
        assert len(conflicts) == 0


class TestOutputSchema:
    """Tests for output schema correctness."""

    def test_canonical_columns(self, tmp_path):
        """Test that canonical output has expected columns."""
        repo_root = tmp_path
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a minimal test manifest
        csv_path = data_dir / "test_manifest.csv"
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg"],
                "label": [25.0],
            }
        )
        df.to_csv(csv_path, index=False)

        (tmp_path / "image1.jpg").touch()

        # Load and process
        loaded_df = builder.load_manifest(csv_path, repo_root, "test_manifest")
        filtered_df = builder.filter_valid_rows(loaded_df, repo_root)
        canonical_df, _ = builder.deduplicate_and_resolve_conflicts(filtered_df)

        expected_columns = [
            "image_path",
            "value",
            "source_tag",
            "hardness_tag",
            "crop_x_min",
            "crop_y_min",
            "crop_x_max",
            "crop_y_max",
            "source_file",
        ]

        for col in expected_columns:
            assert col in canonical_df.columns, f"Missing column: {col}"

    def test_value_is_numeric(self, tmp_path):
        """Test that value column is numeric."""
        repo_root = tmp_path
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        csv_path = data_dir / "test_manifest.csv"
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg", "image2.jpg"],
                "label": [25.0, 30.0],
            }
        )
        df.to_csv(csv_path, index=False)

        (tmp_path / "image1.jpg").touch()
        (tmp_path / "image2.jpg").touch()

        loaded_df = builder.load_manifest(csv_path, repo_root, "test_manifest")
        filtered_df = builder.filter_valid_rows(loaded_df, repo_root)

        assert pd.api.types.is_numeric_dtype(filtered_df["value"])


class TestSourceTagMapping:
    """Tests for source tag mapping."""

    def test_core_source_tag(self, tmp_path):
        """Test that core manifests get the 'core' source_tag."""
        repo_root = tmp_path
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        csv_path = data_dir / "test.csv"
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg"],
                "label": [25.0],
            }
        )
        df.to_csv(csv_path, index=False)
        (tmp_path / "image1.jpg").touch()

        result = builder.load_manifest(
            csv_path, repo_root, "unified_manifest_with_crops_v2"
        )

        assert result["source_tag"].iloc[0] == "core"

    def test_hard_case_source_tag(self, tmp_path):
        """Test that hard case manifests get the 'hard_case' source_tag."""
        repo_root = tmp_path
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        csv_path = data_dir / "test.csv"
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg"],
                "label": [25.0],
            }
        )
        df.to_csv(csv_path, index=False)
        (tmp_path / "image1.jpg").touch()

        result = builder.load_manifest(
            csv_path, repo_root, "hard_cases_plus_board30_valid_with_new6"
        )

        assert result["source_tag"].iloc[0] == "hard_case"
        assert result["hardness_tag"].iloc[0] == "hard_case"

    def test_board_capture_source_tag(self, tmp_path):
        """Test that board capture manifests get the 'board_capture' source_tag."""
        repo_root = tmp_path
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        csv_path = data_dir / "test.csv"
        df = pd.DataFrame(
            {
                "image_path": ["image1.jpg"],
                "label": [25.0],
            }
        )
        df.to_csv(csv_path, index=False)
        (tmp_path / "image1.jpg").touch()

        result = builder.load_manifest(csv_path, repo_root, "new_labelled_captures4")

        assert result["source_tag"].iloc[0] == "board_capture"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
