"""Prepare generic-corpus crops with the same wider runtime geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_center_tip_v1_160_gray"
OUTPUT = ROOT / "tmp" / "generic_conditioned_wide_v1"
SIZE = 160
SOURCE_CROP_SCALE = 1.18
RUNTIME_CROP_SCALE = 1.60


def make_sample(row: dict[str, object], split: str) -> tuple[np.ndarray, np.ndarray]:
    """Build one wide crop and map its point labels into local coordinates."""
    image = np.asarray(Image.open(DATA / str(row["image"])).convert("L"), dtype=np.float32) / 255.0
    fixed = np.asarray(row["ellipse"], dtype=np.float32)
    # Metadata ellipses are in the original 640 frame; the stored image is 160.
    fixed *= SIZE / float(row.get("source_width", 640))
    local_points = np.asarray((row["center_xy_norm"], row["tip_xy_norm"]), dtype=np.float32)
    source_side = max(2.0 * fixed[2], 2.0 * fixed[3]) * SOURCE_CROP_SCALE
    source_points = fixed[:2] - source_side / 2.0 + local_points * source_side
    runtime_side = max(2.0 * fixed[2], 2.0 * fixed[3]) * RUNTIME_CROP_SCALE
    left, top = fixed[:2] - runtime_side / 2.0
    crop = Image.fromarray(np.rint(image * 255.0).astype(np.uint8)).crop((float(left), float(top), float(left + runtime_side), float(top + runtime_side))).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    gray = np.asarray(crop, dtype=np.float32) / 255.0
    axis = (np.arange(SIZE, dtype=np.float32) + 0.5) / SIZE * runtime_side
    xx, yy = np.meshgrid(axis + left, axis + top)
    cx, cy, rx, ry = fixed
    mask = (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0).astype(np.float32)
    inputs = np.stack((gray * 2.0 - 1.0, mask * 2.0 - 1.0), axis=-1)
    points = np.clip((source_points - np.asarray((left, top), dtype=np.float32)) / runtime_side, 0.0, 1.0)
    return inputs.astype(np.float32), points.astype(np.float32)


def main() -> None:
    """Write one wide conditioned NPZ for each generic split."""
    metadata = json.loads((DATA / "metadata.json").read_text())["splits"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split, rows in metadata.items():
        inputs, points = zip(*(make_sample(row, split) for row in rows))
        np.savez_compressed(OUTPUT / f"{split}.npz", inputs=np.stack(inputs), points=np.stack(points))
        print(split, len(rows), "written")


if __name__ == "__main__":
    main()
