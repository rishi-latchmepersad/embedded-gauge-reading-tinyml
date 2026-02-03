"""Unit tests for dataset utilities (CVAT zip parsing and dataset assembly)."""

from __future__ import annotations  # Use future annotations for forward refs.

from pathlib import Path  # Use Path for type-safe filesystem paths.
from zipfile import ZipFile  # Use ZipFile to build CVAT-style test zips.

import pytest  # Use pytest for assertions and exception checks.

from embedded_gauge_reading_tinyml import dataset  # Import module under test.


def _write_cvat_zip(zip_path: Path, xml_text: str) -> None:
    """Create a CVAT-style zip at the given path with annotations.xml content."""
    # Write a minimal zip with the expected file name so parser can find it.
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("annotations.xml", xml_text)  # Store XML at expected path.


def _xml_one_image(
    *,
    image_name: str,
    include_rotation: bool = True,
    include_tip: bool = True,
) -> str:
    """Build a minimal CVAT annotations.xml string for a single image."""
    # Conditionally include rotation to exercise the default branch.
    rotation_attr: str = ' rotation="15.0"' if include_rotation else ""
    # Conditionally include the tip label to test missing-label error.
    tip_points: str = (
        '<points label="temp_tip" points="150.0,160.0" />' if include_tip else ""
    )
    # Return a compact CVAT-style XML payload for one image.
    xml_text: str = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<annotations>"
        f'<image id="0" name="{image_name}" width="640" height="480">'
        f'<ellipse label="temp_dial" cx="100.0" cy="110.0" rx="50.0" ry="60.0"{rotation_attr} />'
        '<points label="temp_center" points="120.0,130.0" />'
        f"{tip_points}"
        "</image>"
        "</annotations>"
    )
    return xml_text  # Return the constructed XML string.


def test_list_labelled_zips_sorts_and_filters(tmp_path: Path) -> None:
    """list_labelled_zips returns only .zip files and sorts by name."""
    # Create two zip files and one non-zip to validate filtering.
    (tmp_path / "b.zip").write_text("x", encoding="utf-8")  # Unsorted zip.
    (tmp_path / "a.zip").write_text("y", encoding="utf-8")  # Unsorted zip.
    (tmp_path / "note.txt").write_text("z", encoding="utf-8")  # Non-zip.

    # Call the function under test.
    result: list[Path] = dataset.list_labelled_zips(tmp_path)

    # Expect only zip files in sorted order.
    assert result == [tmp_path / "a.zip", tmp_path / "b.zip"]


def test_parse_cvat_zip_happy_path(tmp_path: Path) -> None:
    """parse_cvat_zip returns a fully populated Sample with correct types."""
    # Build a CVAT-style zip with all required labels.
    zip_path: Path = tmp_path / "batch_1.zip"
    xml_text: str = _xml_one_image(image_name="img_001.jpg", include_rotation=True)
    _write_cvat_zip(zip_path, xml_text)  # Write annotations.xml into zip.

    # Provide a raw directory so paths are resolved deterministically.
    raw_dir: Path = tmp_path / "raw"
    samples: list[dataset.Sample] = dataset.parse_cvat_zip(zip_path, raw_dir=raw_dir)

    # Ensure we parsed exactly one sample.
    assert len(samples) == 1
    sample: dataset.Sample = samples[0]

    # Verify image path resolution.
    assert sample.image_path == raw_dir / "img_001.jpg"

    # Verify labels exist and match expected names.
    assert sample.dial.label == "temp_dial"
    assert sample.center.label == "temp_center"
    assert sample.tip.label == "temp_tip"

    # Verify numeric parsing from strings.
    assert sample.dial.rotation == 15.0
    assert sample.center.x == 120.0
    assert sample.center.y == 130.0
    assert sample.tip.x == 150.0
    assert sample.tip.y == 160.0


def test_parse_cvat_zip_rotation_default_is_zero(tmp_path: Path) -> None:
    """parse_cvat_zip defaults ellipse rotation to 0.0 when absent."""
    # Build a zip without the rotation attribute.
    zip_path: Path = tmp_path / "batch_2.zip"
    xml_text: str = _xml_one_image(image_name="img_002.jpg", include_rotation=False)
    _write_cvat_zip(zip_path, xml_text)  # Write annotations.xml into zip.

    # Parse and check that rotation falls back to 0.0.
    samples: list[dataset.Sample] = dataset.parse_cvat_zip(zip_path, raw_dir=tmp_path)
    assert samples[0].dial.rotation == 0.0


def test_parse_cvat_zip_missing_label_raises(tmp_path: Path) -> None:
    """parse_cvat_zip raises ValueError when required labels are missing."""
    # Build a zip without the tip label to trigger validation.
    zip_path: Path = tmp_path / "batch_3.zip"
    xml_text: str = _xml_one_image(image_name="img_003.jpg", include_tip=False)
    _write_cvat_zip(zip_path, xml_text)  # Write annotations.xml into zip.

    # Expect a ValueError when required labels are missing.
    with pytest.raises(ValueError, match="Missing labels"):
        dataset.parse_cvat_zip(zip_path, raw_dir=tmp_path)


def test_load_dataset_combines_multiple_zips(tmp_path: Path) -> None:
    """load_dataset merges samples from all zip files in sorted order."""
    # Create two zips to validate aggregation across files.
    xml_a: str = _xml_one_image(image_name="a.jpg", include_rotation=True)
    xml_b: str = _xml_one_image(image_name="b.jpg", include_rotation=True)
    _write_cvat_zip(tmp_path / "a.zip", xml_a)  # First zip.
    _write_cvat_zip(tmp_path / "b.zip", xml_b)  # Second zip.

    # Load the dataset from the temp directory.
    raw_dir: Path = tmp_path / "raw"
    samples: list[dataset.Sample] = dataset.load_dataset(
        labelled_dir=tmp_path,
        raw_dir=raw_dir,
    )

    # Ensure sample order follows zip sort order.
    assert [s.image_path.name for s in samples] == ["a.jpg", "b.jpg"]
