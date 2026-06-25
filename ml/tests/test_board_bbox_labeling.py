from __future__ import annotations

from pathlib import Path

from PIL import Image

from embedded_gauge_reading_tinyml.board_bbox_labeling import (
    BoardBBoxRecord,
    load_board_bbox_candidates,
    load_board_bbox_records,
    write_board_bbox_records,
)


def test_board_bbox_record_roundtrip(tmp_path: Path) -> None:
    """A saved bbox CSV should round-trip back into the same record."""

    image_path = Path("ml/data/captured_images/_live_rectified_probe/demo/board_crop.png")
    record = BoardBBoxRecord(
        image_path=image_path,
        source_width=320,
        source_height=240,
        crop_x_min=12.5,
        crop_y_min=14.0,
        crop_x_max=210.0,
        crop_y_max=200.5,
        quality_flag="review",
        label_source="manual_gui",
        origin_manifest="ml/data/board_rectified_probe_20260422.csv",
    )
    output_csv = tmp_path / "labels.csv"
    write_board_bbox_records(output_csv, [record])

    loaded = load_board_bbox_records(output_csv)
    assert image_path.as_posix() in loaded
    restored = loaded[image_path.as_posix()]
    assert restored.source_width == 320
    assert restored.source_height == 240
    assert restored.crop_x_min == 12.5
    assert restored.crop_y_max == 200.5
    assert restored.has_box


def test_board_bbox_candidate_loader_reads_manifest(tmp_path: Path) -> None:
    """The manifest loader should resolve board images and their size."""

    image_dir = tmp_path / "board"
    image_dir.mkdir()
    image_path = image_dir / "board_crop.png"
    Image.new("RGB", (128, 96), color=(10, 20, 30)).save(image_path)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,value\n" f"{image_path.as_posix()},14\n", encoding="utf-8")

    candidates = load_board_bbox_candidates(manifest, limit=1)
    assert len(candidates) == 1
    assert candidates[0].source_width == 128
    assert candidates[0].source_height == 96
    assert candidates[0].image_path.as_posix().endswith("board_crop.png")
