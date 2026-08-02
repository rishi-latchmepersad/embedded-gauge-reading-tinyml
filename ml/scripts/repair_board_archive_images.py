#!/usr/bin/env python3
"""Re-inject missing image bytes into the labelled board archives.

The ``initial_temp_gauge`` archives under ``ml/data/labelled`` carry full CVAT
``annotations.xml`` files, but the image files themselves were stripped from
most archives when the dataset was first committed.  The image bytes still
exist on disk:

- ``ml/data/raw`` holds the ``PXL_*.jpg`` phone captures referenced by
  ``gauge_1_batch_*.zip``.
- ``ml/data/captured_images/clean_board_captures`` holds the
  ``capture_*.png`` board frames referenced by ``board_captures_3.zip`` and
  ``board_captures_4.zip``.

This script rewrites each affected archive as a proper CVAT zip that contains
``annotations.xml`` plus every referenced image, so the shared loaders
(``load_zips`` in ``train_ellipse_robust_384.py``) can actually consume the
labelled data.  Archives that already contain their images are left untouched.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "data" / "labelled" / "initial_temp_gauge"
RAW = ROOT / "data" / "raw"
CLEAN_BOARD = ROOT / "data" / "captured_images" / "clean_board_captures"

# The archives that currently hold only annotations.xml and no image bytes.
TARGET_ARCHIVES = [
    "board_captures_3.zip",
    "board_captures_4.zip",
    "gauge_1_batch_1.zip",
    "gauge_1_batch_2.zip",
    "gauge_1_batch_3.zip",
    "gauge_1_batch_4.zip",
    "gauge_1_batch_5.zip",
    "gauge_1_batch_6.zip",
    "gauge_1_batch_7.zip",
    "gauge_1_batch_8.zip",
]


def _image_roots() -> Iterable[Path]:
    """Yield the directories that may hold archive image bytes."""
    for path in (RAW, CLEAN_BOARD):
        if path.is_dir():
            yield path


def resolve_image(name: str) -> Path | None:
    """Locate one referenced image file by basename across the raw folders."""
    for root in _image_roots():
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def rebuild_archive(archive_name: str) -> tuple[int, int]:
    """Rewrite one archive with annotations.xml plus every referenced image.

    Returns (images_written, images_missing) so callers can verify coverage.
    """
    archive_path = LABELLED / archive_name
    with zipfile.ZipFile(archive_path) as source:
        xml_bytes = source.read("annotations.xml")
    root = ET.fromstring(xml_bytes)

    written = 0
    missing = 0
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as target:
        target.writestr("annotations.xml", xml_bytes)
        for image_node in root.findall("image"):
            name = image_node.get("name", "")
            if not name:
                continue
            image_path = resolve_image(name)
            if image_path is None:
                missing += 1
                print(f"  MISSING {archive_name}: {name}", flush=True)
                continue
            target.write(image_path, arcname=name)
            written += 1
    return written, missing


def main() -> None:
    """Rebuild every image-less archive and report coverage."""
    total_written = 0
    total_missing = 0
    for archive_name in TARGET_ARCHIVES:
        with zipfile.ZipFile(LABELLED / archive_name) as probe:
            has_images = any(
                member.lower().endswith((".png", ".jpg", ".jpeg"))
                for member in probe.namelist()
            )
        if has_images:
            print(f"SKIP  {archive_name} (already contains images)", flush=True)
            continue
        print(f"REBUILD {archive_name}", flush=True)
        written, missing = rebuild_archive(archive_name)
        total_written += written
        total_missing += missing
        print(f"  wrote {written} images, {missing} missing", flush=True)
    print(f"TOTAL: {total_written} written, {total_missing} missing", flush=True)
    sys.exit(1 if total_missing else 0)


if __name__ == "__main__":
    main()
