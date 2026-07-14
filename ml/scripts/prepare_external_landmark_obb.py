"""Validate the local Roboflow landmark OBB dataset and write a local manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CLASS_NAMES: tuple[str, ...] = ("Center", "End", "Start", "Tip")
SPLITS: tuple[str, ...] = ("train", "valid", "test")
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "ml/data/external/gauge_meter_detection_v2_yolov8_obb"


@dataclass(frozen=True)
class SplitSummary:
    """Summarize image, label, and object counts for one dataset split."""

    images: int
    labels: int
    objects: int
    objects_by_class: dict[str, int]


def _iter_images(image_dir: Path) -> Iterable[Path]:
    """Yield supported image files in deterministic order."""

    yield from sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def _read_label_file(label_path: Path) -> list[int]:
    """Validate one YOLO-OBB label file and return its class ids."""

    class_ids: list[int] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw_line.split()
        if len(fields) != 9:
            raise ValueError(
                f"{label_path}:{line_number}: expected class plus 8 coordinates, "
                f"got {len(fields)} fields"
            )

        class_id = int(fields[0])
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"{label_path}:{line_number}: invalid class {class_id}")

        coordinates = [float(value) for value in fields[1:]]
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: coordinates must be normalized")

        class_ids.append(class_id)
    return class_ids


def summarize_split(dataset_root: Path, split: str) -> SplitSummary:
    """Validate image/label pairing and summarize one split."""

    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    images = list(_iter_images(image_dir))
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in labels}
    if image_stems != label_stems:
        raise ValueError(f"{split}: image/label stems do not match")

    class_counts: Counter[int] = Counter()
    for label_path in labels:
        class_counts.update(_read_label_file(label_path))

    return SplitSummary(
        images=len(images),
        labels=len(labels),
        objects=sum(class_counts.values()),
        objects_by_class={
            CLASS_NAMES[class_id]: class_counts[class_id]
            for class_id in range(len(CLASS_NAMES))
        },
    )


def write_local_yaml(dataset_root: Path, yaml_path: Path) -> None:
    """Write a path-correct Ultralytics YAML without modifying the source archive."""

    # why: the downloaded Roboflow YAML points at a cloud-export path that does not
    # exist in this checkout, so generate a relocatable local configuration.
    root = dataset_root.resolve().as_posix()
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Validate the dataset and write its local training manifest/YAML."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument("--yaml", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    yaml_path = args.yaml or dataset_root / "data_local.yaml"
    summaries = {split: summarize_split(dataset_root, split) for split in SPLITS}
    write_local_yaml(dataset_root, yaml_path)
    manifest_path = dataset_root / "manifest_local.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classes": list(CLASS_NAMES),
                "dataset_root": dataset_root.as_posix(),
                "splits": {name: asdict(summary) for name, summary in summaries.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({name: asdict(summary) for name, summary in summaries.items()}, indent=2))
    print(f"Wrote {yaml_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
