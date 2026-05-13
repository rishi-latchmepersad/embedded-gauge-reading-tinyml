"""Generate polar needle masks from labeled gauge data for segmentation supervision.

For each labeled image with center and tip points, we compute the needle angle
and generate a soft mask in polar space. These masks serve as ground truth for
training the polar needle-segmentation model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Add ml/src to path.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.dataset import load_dataset
from embedded_gauge_reading_tinyml.gauge.processing import (
    GaugeSpec,
    load_gauge_specs,
    needle_value,
)
from embedded_gauge_reading_tinyml.polar_projection import (
    needle_mask_from_polar,
    polar_project_image_path,
)


def _compute_needle_angle_deg(sample: Any, spec: GaugeSpec) -> float:
    """Compute the needle angle in degrees from a labeled sample."""
    dx = sample.tip.x - sample.center.x
    dy = sample.tip.y - sample.center.y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg % 360.0


def generate_polar_masks(
    output_dir: Path,
    gauge_id: str = "littlegood_home_temp_gauge_c",
    polar_size: int = 224,
    mask_sigma: float = 3.0,
) -> None:
    """Generate polar-projected images and needle masks from labeled data.

    Also generates masks for all scalar manifest rows that have image files.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    specs = load_gauge_specs()
    spec = specs[gauge_id]

    # Load labeled samples.
    labeled_samples = load_dataset()
    print(f"Loaded {len(labeled_samples)} labeled samples")

    rows: list[dict[str, str]] = []

    for idx, sample in enumerate(labeled_samples):
        image_path = sample.image_path
        if not image_path.exists():
            print(f"  Skip missing image: {image_path}")
            continue

        value = needle_value(sample, spec, strict=False)
        angle_deg = _compute_needle_angle_deg(sample, spec)

        # Polar project.
        try:
            polar_img = polar_project_image_path(
                image_path,
                polar_size=polar_size,
            )
        except Exception as exc:
            print(f"  Skip polar projection failed for {image_path}: {exc}")
            continue

        # Generate mask.
        mask = needle_mask_from_polar(
            polar_img,
            needle_angle_deg=angle_deg,
            mask_sigma=mask_sigma,
        )

        # Save.
        stem = image_path.stem
        img_out = images_dir / f"{stem}_polar.png"
        mask_out = masks_dir / f"{stem}_mask.png"

        # Save polar image as uint8.
        img_uint8 = np.clip(polar_img * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(img_uint8).save(img_out)

        # Save mask as uint8.
        mask_uint8 = np.clip(mask[..., 0] * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(mask_uint8).save(mask_out)

        rows.append(
            {
                "image_path": str(Path("ml") / img_out.relative_to(PROJECT_ROOT)),
                "mask_path": str(Path("ml") / mask_out.relative_to(PROJECT_ROOT)),
                "original_path": str(image_path),
                "value": f"{value:.2f}",
                "angle_deg": f"{angle_deg:.2f}",
            }
        )

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(labeled_samples)}")

    # Write manifest.
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "mask_path",
                "original_path",
                "value",
                "angle_deg",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate polar needle masks.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/data/polar_masks"),
        help="Output directory for polar images and masks.",
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default="littlegood_home_temp_gauge_c",
        help="Gauge ID for calibration.",
    )
    parser.add_argument(
        "--polar-size",
        type=int,
        default=224,
        help="Size of polar projection.",
    )
    parser.add_argument(
        "--mask-sigma",
        type=float,
        default=3.0,
        help="Width of Gaussian needle mask in pixels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_polar_masks(
        output_dir=args.output_dir,
        gauge_id=args.gauge_id,
        polar_size=args.polar_size,
        mask_sigma=args.mask_sigma,
    )


if __name__ == "__main__":
    main()
