"""Baseline luma bright-centroid crop detector.

This module emulates the firmware's AppBaselineRuntime_EstimateCenterFromBrightPixels()
from app_baseline_runtime.c, producing a dynamic crop centered on the detected
bright centroid of the gauge face.

Algorithm (matching firmware exactly):
1. Compute inner dial center from training crop ratios
2. Compute dial radius from training crop height * 0.56
3. Scan 1.5x dial radius around inner dial center for bright pixels
4. Threshold: luma >= 150 (BRIGHT_THRESHOLD), exclude > 235 (SATURATION)
5. Require >= 1024 bright pixels (MIN_BRIGHT_PIXELS)
6. Compute centroid, bias Y upward by 11% of bright bbox height [8..18] px
7. Center a dynamic crop (training crop dimensions) on the biased centroid

Usage:
    python ml/scripts/luma_crop_detector.py --input image.png --output-dir tmp/crop_test/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

# ---------------------------------------------------------------------------
# Firmware constants from app_baseline_runtime.c and app_gauge_geometry.h
# ---------------------------------------------------------------------------

# Bright-centroid detection (app_baseline_runtime.c:66-86)
BRIGHT_THRESHOLD: Final[int] = 150
SATURATION_THRESHOLD: Final[int] = 235
MIN_BRIGHT_PIXELS: Final[int] = 1024
DIAL_RADIUS_FROM_CROP_HEIGHT_RATIO: Final[float] = 0.56
SCAN_RADIUS_RATIO: Final[float] = 1.5

# Inner dial center ratios (app_gauge_geometry.h:28-29)
INNER_DIAL_CENTER_X_RATIO: Final[float] = 0.50
INNER_DIAL_CENTER_Y_RATIO: Final[float] = 0.446

# Training crop ratios (app_gauge_geometry.h:16-19)
TRAINING_CROP_X_MIN_RATIO: Final[float] = 0.1027
TRAINING_CROP_Y_MIN_RATIO: Final[float] = 0.2573
TRAINING_CROP_X_MAX_RATIO: Final[float] = 0.7987
TRAINING_CROP_Y_MAX_RATIO: Final[float] = 0.8071

# Y bias (app_baseline_runtime.c:1413-1416)
CENTER_Y_BIAS_RATIO: Final[float] = 0.11
CENTER_Y_BIAS_MIN_PIXELS: Final[int] = 8
CENTER_Y_BIAS_MAX_PIXELS: Final[int] = 18

# Minimum crop size relative to frame
MIN_CROP_FRACTION: Final[float] = 0.25

DEFAULT_IMAGE_SIZE: Final[int] = 224


@dataclass(frozen=True)
class BrightCentroidResult:
    """Result of the bright-centroid detection."""

    center_x: float
    center_y: float
    bright_count: int
    bright_bbox_height: int
    bias_y: int
    dial_radius: float
    scan_radius: float
    detected: bool


@dataclass(frozen=True)
class CropBox:
    """Integer crop geometry."""

    x_min: int
    y_min: int
    width: int
    height: int

    @property
    def x_max(self) -> int:
        return self.x_min + self.width

    @property
    def y_max(self) -> int:
        return self.y_min + self.height


def rgb_to_luma(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert RGB to luma using BT.601 weights.

    Matches the firmware's AppBaselineRuntime_ReadLuma() which reads the Y
    channel from YUV422. For RGB inputs, we compute:
        luma = 0.299*R + 0.587*G + 0.114*B

    Args:
        image: RGB image array, shape (H, W, 3), dtype uint8

    Returns:
        Luma array, shape (H, W), dtype uint8
    """
    rgb = image.astype(np.float32)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.rint(luma), 0.0, 255.0).astype(np.uint8)


def estimate_dial_radius(height: int) -> float:
    """Estimate dial radius from image height.

    Matches AppBaselineRuntime_EstimateDialRadiusPixels():
        radius = crop_height * DIAL_RADIUS_FROM_CROP_HEIGHT_RATIO

    Args:
        height: Image height in pixels

    Returns:
        Estimated dial radius in pixels, clamped to [16, 0.49 * min_dim]
    """
    crop_height = height * (TRAINING_CROP_Y_MAX_RATIO - TRAINING_CROP_Y_MIN_RATIO)
    radius = crop_height * DIAL_RADIUS_FROM_CROP_HEIGHT_RATIO
    min_dim = min(height, height)
    frame_limit = 0.49 * min_dim
    return max(16.0, min(radius, frame_limit))


def estimate_bright_centroid(
    image: NDArray[np.uint8],
    *,
    bright_threshold: int = BRIGHT_THRESHOLD,
    saturation_threshold: int = SATURATION_THRESHOLD,
    min_bright_pixels: int = MIN_BRIGHT_PIXELS,
) -> BrightCentroidResult:
    """Estimate the bright centroid of the gauge face.

    Matches AppBaselineRuntime_EstimateCenterFromBrightPixels() from
    app_baseline_runtime.c:1317-1422 exactly.

    Args:
        image: RGB image array, shape (H, W, 3), dtype uint8

    Returns:
        BrightCentroidResult with detected center and metadata
    """
    height, width = image.shape[:2]

    # Step 1: Compute inner dial center (training crop center)
    inner_center_x = int(width * INNER_DIAL_CENTER_X_RATIO)
    inner_center_y = int(height * INNER_DIAL_CENTER_Y_RATIO)

    # Step 2: Compute dial radius and scan area
    dial_radius = estimate_dial_radius(height)
    scan_radius = int(dial_radius * SCAN_RADIUS_RATIO)

    scan_x_min = max(0, inner_center_x - scan_radius)
    scan_x_max = min(width, inner_center_x + scan_radius)
    scan_y_min = max(0, inner_center_y - scan_radius)
    scan_y_max = min(height, inner_center_y + scan_radius)

    # Step 3: Compute luma
    luma = rgb_to_luma(image)

    # Step 4: Scan for bright pixels
    bright_y_coords: list[int] = []
    bright_x_coords: list[int] = []

    for y in range(scan_y_min, scan_y_max):
        for x in range(scan_x_min, scan_x_max):
            luma_val = int(luma[y, x])

            # Skip if below bright threshold
            if luma_val < bright_threshold:
                continue

            # Skip saturated/glare pixels
            if luma_val > saturation_threshold:
                continue

            bright_x_coords.append(x)
            bright_y_coords.append(y)

    bright_count = len(bright_x_coords)

    # Step 5: Check minimum bright pixels
    if bright_count < min_bright_pixels:
        return BrightCentroidResult(
            center_x=float(inner_center_x),
            center_y=float(inner_center_y),
            bright_count=bright_count,
            bright_bbox_height=0,
            bias_y=0,
            dial_radius=dial_radius,
            scan_radius=float(scan_radius),
            detected=False,
        )

    # Step 6: Compute centroid
    raw_center_x = float(sum(bright_x_coords) / bright_count)
    raw_center_y = float(sum(bright_y_coords) / bright_count)

    # Step 7: Compute Y bias (upward shift to land on inner Celsius dial)
    bright_y_min = min(bright_y_coords)
    bright_y_max = max(bright_y_coords)
    crop_height = bright_y_max - bright_y_min
    bias = int(0.11 * crop_height + 0.5)
    clamped_bias = max(CENTER_Y_BIAS_MIN_PIXELS, min(bias, CENTER_Y_BIAS_MAX_PIXELS))
    biased_center_y = max(0.0, raw_center_y - clamped_bias)

    return BrightCentroidResult(
        center_x=raw_center_x,
        center_y=biased_center_y,
        bright_count=bright_count,
        bright_bbox_height=crop_height,
        bias_y=clamped_bias,
        dial_radius=dial_radius,
        scan_radius=float(scan_radius),
        detected=True,
    )


def compute_dynamic_crop(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
) -> CropBox | None:
    """Compute a dynamic crop box centered on the detected bright centroid.

    Uses the training crop dimensions scaled to 85% (matching the AI-side
    crop size) or the full training crop, centered on the detected centroid.

    Args:
        width: Image width
        height: Image height
        center_x: Detected bright centroid X
        center_y: Detected bright centroid Y

    Returns:
        CropBox or None if crop is too small
    """
    # Use the training crop dimensions
    crop_width = int(width * (TRAINING_CROP_X_MAX_RATIO - TRAINING_CROP_X_MIN_RATIO))
    crop_height = int(height * (TRAINING_CROP_Y_MAX_RATIO - TRAINING_CROP_Y_MIN_RATIO))

    # Center on the detected centroid
    left = int(center_x - crop_width / 2)
    top = int(center_y - crop_height / 2)

    # Clamp to image bounds
    left = max(0, min(left, width - crop_width))
    top = max(0, min(top, height - crop_height))

    # Ensure positive dimensions
    crop_width = min(crop_width, width - left)
    crop_height = min(crop_height, height - top)

    # Minimum size check
    if crop_width < width * MIN_CROP_FRACTION or crop_height < height * MIN_CROP_FRACTION:
        return None

    return CropBox(x_min=left, y_min=top, width=crop_width, height=crop_height)


def crop_and_resize(
    image: NDArray[np.uint8],
    crop_box: CropBox,
    target_size: int = DEFAULT_IMAGE_SIZE,
) -> NDArray[np.uint8]:
    """Crop and resize an image to the target size with padding.

    Uses bilinear resize with padding to preserve aspect ratio, matching
    the firmware's resize_with_pad behavior.

    Args:
        image: Input image (H, W, 3)
        crop_box: Crop region
        target_size: Output size (target_size x target_size)

    Returns:
        Resized image, shape (target_size, target_size, 3)
    """
    # Crop
    cropped = image[
        crop_box.y_min : crop_box.y_max,
        crop_box.x_min : crop_box.x_max,
    ]

    # Convert to PIL for resize with pad
    pil_image = Image.fromarray(cropped, mode="RGB")

    # Compute scale to fit within target_size while preserving aspect ratio
    crop_h, crop_w = cropped.shape[:2]
    scale = min(target_size / crop_w, target_size / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized = pil_image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    # Center on canvas
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return np.asarray(canvas, dtype=np.uint8)


def run_crop_detector(
    image_path: Path,
    output_dir: Path | None = None,
) -> dict:
    """Run the full crop detection pipeline on one image.

    Args:
        image_path: Path to input image
        output_dir: Optional directory for output files

    Returns:
        Dict with detection results and crop metadata
    """
    # Load image
    with Image.open(image_path) as img:
        rgb_image = np.asarray(img.convert("RGB"), dtype=np.uint8)

    height, width = rgb_image.shape[:2]

    # Run bright-centroid detection
    centroid = estimate_bright_centroid(rgb_image)

    # Compute dynamic crop
    crop_box = None
    if centroid.detected:
        crop_box = compute_dynamic_crop(width, height, centroid.center_x, centroid.center_y)

    # Fall back to fixed training crop if detection failed
    if crop_box is None:
        x_min = int(width * TRAINING_CROP_X_MIN_RATIO)
        y_min = int(height * TRAINING_CROP_Y_MIN_RATIO)
        crop_w = int(width * (TRAINING_CROP_X_MAX_RATIO - TRAINING_CROP_X_MIN_RATIO))
        crop_h = int(height * (TRAINING_CROP_Y_MAX_RATIO - TRAINING_CROP_Y_MIN_RATIO))
        crop_box = CropBox(x_min=x_min, y_min=y_min, width=crop_w, height=crop_h)
        source = "fixed_training_crop"
    else:
        source = "bright_centroid"

    # Crop and resize
    resized = crop_and_resize(rgb_image, crop_box)

    # Save outputs if output_dir specified
    results = {
        "image_path": str(image_path),
        "image_size": f"{width}x{height}",
        "detection_source": source,
        "centroid": {
            "x": round(centroid.center_x, 1),
            "y": round(centroid.center_y, 1),
            "bright_count": centroid.bright_count,
            "detected": centroid.detected,
        },
        "dial_radius": round(centroid.dial_radius, 1),
        "crop_box": {
            "x_min": crop_box.x_min,
            "y_min": crop_box.y_min,
            "width": crop_box.width,
            "height": crop_box.height,
        },
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save cropped image
        crop_path = output_dir / f"{image_path.stem}_crop.png"
        Image.fromarray(resized, mode="RGB").save(crop_path)
        results["crop_path"] = str(crop_path)

        # Save visualization (if matplotlib available)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Original with crop overlay
            axes[0].imshow(rgb_image)
            rect = patches.Rectangle(
                (crop_box.x_min, crop_box.y_min),
                crop_box.width,
                crop_box.height,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
                label=source,
            )
            axes[0].add_patch(rect)
            if centroid.detected:
                axes[0].plot(
                    centroid.center_x,
                    centroid.center_y,
                    "r+",
                    markersize=12,
                    markeredgewidth=2,
                    label="centroid",
                )
            axes[0].set_title(f"{image_path.name} ({source})")
            axes[0].legend(loc="lower right")
            axes[0].axis("off")

            # Cropped result
            axes[1].imshow(resized)
            axes[1].set_title(f"Crop {crop_box.width}x{crop_box.height}")
            axes[1].axis("off")

            fig.tight_layout()
            vis_path = output_dir / f"{image_path.stem}_vis.png"
            fig.savefig(vis_path, dpi=120)
            plt.close(fig)
            results["vis_path"] = str(vis_path)
        except ImportError:
            pass  # Skip visualization if matplotlib not available

        # Save JSON report
        report_path = output_dir / f"{image_path.stem}_report.json"
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        results["report_path"] = str(report_path)

    return results


def main() -> None:
    """CLI entry point for the luma crop detector."""
    parser = argparse.ArgumentParser(
        description="Baseline luma bright-centroid crop detector"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for crop, visualization, and report",
    )
    args = parser.parse_args()

    results = run_crop_detector(args.input, args.output_dir)

    print(f"Image: {results['image_path']}")
    print(f"Size: {results['image_size']}")
    print(f"Source: {results['detection_source']}")
    print(f"Dial radius: {results['dial_radius']} px")
    print(f"Crop: {results['crop_box']}")
    if "crop_path" in results:
        print(f"Crop saved: {results['crop_path']}")
    if "vis_path" in results:
        print(f"Visualization: {results['vis_path']}")


if __name__ == "__main__":
    main()
