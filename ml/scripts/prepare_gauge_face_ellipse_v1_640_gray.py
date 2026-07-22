"""Build the 640x640 grayscale YOLO OBB dataset for gauge_face_ellipse_v1."""

from __future__ import annotations

import io
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image
from tqdm import tqdm


SIZE = 640
CLASS_ID = 0


@dataclass(frozen=True)
class Ellipse:
    """Represent one GaugeFace ellipse in source-image pixels."""

    cx: float
    cy: float
    rx: float
    ry: float


def _records(archive_path: Path) -> list[tuple[int, str, tuple[Ellipse, ...]]]:
    """Read GaugeFace ellipses from one CVAT image archive."""
    with zipfile.ZipFile(archive_path) as archive:
        root = ElementTree.fromstring(archive.read("annotations.xml"))
    result: list[tuple[int, str, tuple[Ellipse, ...]]] = []
    for index, image in enumerate(root.findall("./image")):
        ellipses = tuple(
            Ellipse(
                float(node.attrib["cx"]),
                float(node.attrib["cy"]),
                float(node.attrib["rx"]),
                float(node.attrib["ry"]),
            )
            for node in image.findall("./ellipse")
            if node.attrib.get("label") == "GaugeFace"
        )
        if ellipses:
            result.append((index, image.attrib["name"], ellipses))
    return result


def _entry(archive: zipfile.ZipFile, name: str) -> zipfile.ZipInfo:
    """Resolve a CVAT image name against the archive’s image entries."""
    basename = Path(name).name
    matches = [item for item in archive.infolist() if Path(item.filename).name == basename]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely resolve {name!r} in {archive.filename}")
    return matches[0]


def _prepare(image: Image.Image) -> tuple[Image.Image, int, int, int]:
    """Center-crop a grayscale image to square and resize it to 640 pixels."""
    image = image.convert("L")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    return cropped.resize((SIZE, SIZE), Image.Resampling.BILINEAR), left, top, side


def _label(ellipse: Ellipse, left: int, top: int, side: int) -> str:
    """Convert a GaugeFace ellipse into clipped normalized OBB corners."""
    scale = SIZE / side
    points = (
        (ellipse.cx - ellipse.rx, ellipse.cy - ellipse.ry),
        (ellipse.cx + ellipse.rx, ellipse.cy - ellipse.ry),
        (ellipse.cx + ellipse.rx, ellipse.cy + ellipse.ry),
        (ellipse.cx - ellipse.rx, ellipse.cy + ellipse.ry),
    )
    values: list[float] = []
    for x, y in points:
        values.extend(
            (
                min(1.0, max(0.0, (x - left) / side)),
                min(1.0, max(0.0, (y - top) / side)),
            )
        )
    return f"{CLASS_ID} " + " ".join(f"{value:.6f}" for value in values)


def _split(archives: tuple[Path, ...], split: str, output: Path) -> int:
    """Extract one train/validation/test split without reopening archives per image."""
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for archive_path in archives:
        records = _records(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            for index, name, ellipses in tqdm(records, desc=archive_path.name):
                # Keep the archive open across the split; reopening it per frame
                # makes a 7k-image export needlessly take many minutes.
                with Image.open(io.BytesIO(archive.read(_entry(archive, name)))) as source:
                    image, left, top, side = _prepare(source)
                stem = f"{archive_path.stem}_{index:06d}_{Path(name).stem}"
                # Low compression keeps preparation practical; the training input
                # remains lossless grayscale and is decoded identically on-device.
                image.save(image_dir / f"{stem}.png", compress_level=1)
                labels = [_label(ellipse, left, top, side) for ellipse in ellipses]
                (label_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
                count += 1
    return count


def main() -> None:
    """Create the versioned 640x640 grayscale dataset and YAML manifest."""
    repo = Path(__file__).resolve().parents[1]
    labelled = repo / "data" / "labelled"
    output = repo / "data" / "gauge_face_ellipse_v1_640_gray"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    groups = {
        "train": (labelled / "train_1.zip", labelled / "train_2.zip"),
        "val": (labelled / "val_1.zip", labelled / "val_2.zip"),
        "test": (labelled / "test_1.zip", labelled / "test_2.zip"),
    }
    counts = {split: _split(paths, split, output) for split, paths in groups.items()}
    (output / "dataset.yaml").write_text(
        f"path: {output.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: gauge_face\n",
        encoding="utf-8",
    )
    print(f"Prepared gauge_face_ellipse_v1_640_gray: {counts}")


if __name__ == "__main__":
    main()
