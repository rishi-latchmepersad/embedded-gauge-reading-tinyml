"""Compare a labeled training image against the board-style crop path.

This module keeps the comparison logic in one place so the CLI script and the
tests can both reuse the exact same crop detection and resize steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Final

import matplotlib

# Use a headless backend so the report works in WSL and CI.
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import tensorflow as tf

from embedded_gauge_reading_tinyml.dataset import Sample

RGBImage = NDArray[np.uint8]

BOARD_BRIGHT_THRESHOLD: Final[int] = 80
BOARD_BORDER_PIXELS: Final[int] = 16
TRAINING_CROP_X_MIN_RATIO: Final[float] = 0.1027
TRAINING_CROP_Y_MIN_RATIO: Final[float] = 0.2573
TRAINING_CROP_X_MAX_RATIO: Final[float] = 0.7987
TRAINING_CROP_Y_MAX_RATIO: Final[float] = 0.8071
DEFAULT_IMAGE_SIZE: Final[int] = 224


@dataclass(frozen=True)
class CropBox:
    """Integer crop geometry used for board-style comparisons."""

    x_min: int
    y_min: int
    width: int
    height: int
    centroid_x: int
    centroid_y: int
    bright_count: int

    @property
    def x_max(self) -> int:
        """Return the exclusive right edge."""
        return self.x_min + self.width

    @property
    def y_max(self) -> int:
        """Return the exclusive bottom edge."""
        return self.y_min + self.height


@dataclass(frozen=True)
class BoardCropEstimate:
    """Summary of the board-style crop selected from the image luma."""

    crop_box: CropBox
    center_luma: int
    mean_luma: float
    min_luma: int
    max_luma: int


@dataclass(frozen=True)
class ComparisonReport:
    """Paths and statistics written by the comparison command."""

    source_image_path: Path
    output_dir: Path
    training_crop_box: tuple[float, float, float, float]
    board_crop_estimate: BoardCropEstimate
    training_crop_path: Path
    board_crop_path: Path
    comparison_figure_path: Path
    report_json_path: Path
    mean_abs_diff: float


@dataclass(frozen=True)
class BoardCaptureReport:
    """Artifacts produced from one raw board capture."""

    capture_path: Path
    output_dir: Path
    board_crop_estimate: BoardCropEstimate
    capture_preview_path: Path
    capture_crop_path: Path
    comparison_figure_path: Path
    report_json_path: Path


def load_rgb_image(image_path: Path) -> RGBImage:
    """Load one labeled image as a uint8 RGB array."""
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.uint8)


def load_yuv422_capture_as_rgb(
    capture_path: Path,
    *,
    image_width: int = DEFAULT_IMAGE_SIZE,
    image_height: int = DEFAULT_IMAGE_SIZE,
) -> RGBImage:
    """Load a packed YUV422 YUYV capture and expose it as grayscale RGB."""
    raw_bytes = capture_path.read_bytes()
    expected_bytes = image_width * image_height * 2
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Unexpected YUV422 size for {capture_path}: "
            f"got {len(raw_bytes)} bytes, expected {expected_bytes}."
        )

    yuyv_pairs = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
        image_height, image_width // 2, 4
    )
    luma = np.empty((image_height, image_width), dtype=np.uint8)
    luma[:, 0::2] = yuyv_pairs[:, :, 0]
    luma[:, 1::2] = yuyv_pairs[:, :, 2]
    return np.repeat(luma[:, :, None], 3, axis=2)


def load_yuv422_luma(
    capture_path: Path,
    *,
    image_width: int = DEFAULT_IMAGE_SIZE,
    image_height: int = DEFAULT_IMAGE_SIZE,
) -> NDArray[np.uint8]:
    """Load a packed YUV422 YUYV capture and return only the luma plane."""
    raw_bytes = capture_path.read_bytes()
    expected_bytes = image_width * image_height * 2
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Unexpected YUV422 size for {capture_path}: "
            f"got {len(raw_bytes)} bytes, expected {expected_bytes}."
        )

    yuyv_pairs = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
        image_height, image_width // 2, 4
    )
    luma = np.empty((image_height, image_width), dtype=np.uint8)
    luma[:, 0::2] = yuyv_pairs[:, :, 0]
    luma[:, 1::2] = yuyv_pairs[:, :, 2]
    return luma


def rgb_to_luma(image: RGBImage) -> NDArray[np.uint8]:
    """Convert RGB to an 8-bit luminance image using BT.601-style weights."""
    rgb = image.astype(np.float32)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.rint(luma), 0.0, 255.0).astype(np.uint8)


def find_latest_board_capture(captured_dir: Path) -> Path:
    """Return the newest board capture file from the capture directory."""
    candidates = sorted(
        captured_dir.glob("*.yuv422"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"No .yuv422 captures found in {captured_dir}.")
    return candidates[-1]


def compute_training_crop_box(sample: Sample, pad_ratio: float) -> tuple[float, float, float, float]:
    """Recreate the training crop box used to label and resize the gauge ROI."""
    pad_x: float = sample.dial.rx * pad_ratio
    pad_y: float = sample.dial.ry * pad_ratio
    x_min: float = sample.dial.cx - sample.dial.rx - pad_x
    y_min: float = sample.dial.cy - sample.dial.ry - pad_y
    x_max: float = sample.dial.cx + sample.dial.rx + pad_x
    y_max: float = sample.dial.cy + sample.dial.ry + pad_y
    return (x_min, y_min, x_max, y_max)


def _crop_image_with_xyxy(image: tf.Tensor, crop_box_xyxy: tuple[float, float, float, float]) -> tf.Tensor:
    """Crop with float xyxy coordinates while clipping safely to image bounds."""
    shape = tf.shape(image)
    img_h = shape[0]
    img_w = shape[1]

    crop_box = tf.convert_to_tensor(crop_box_xyxy, dtype=tf.float32)
    x_min_f = tf.clip_by_value(crop_box[0], 0.0, tf.cast(img_w - 1, tf.float32))
    y_min_f = tf.clip_by_value(crop_box[1], 0.0, tf.cast(img_h - 1, tf.float32))
    x_max_f = tf.clip_by_value(crop_box[2], x_min_f + 1.0, tf.cast(img_w, tf.float32))
    y_max_f = tf.clip_by_value(crop_box[3], y_min_f + 1.0, tf.cast(img_h, tf.float32))

    x_min = tf.cast(tf.math.floor(x_min_f), tf.int32)
    y_min = tf.cast(tf.math.floor(y_min_f), tf.int32)
    x_max = tf.cast(tf.math.ceil(x_max_f), tf.int32)
    y_max = tf.cast(tf.math.ceil(y_max_f), tf.int32)

    crop_w = tf.maximum(1, x_max - x_min)
    crop_h = tf.maximum(1, y_max - y_min)
    crop_w = tf.minimum(crop_w, img_w - x_min)
    crop_h = tf.minimum(crop_h, img_h - y_min)
    return tf.image.crop_to_bounding_box(image, y_min, x_min, crop_h, crop_w)


def resize_with_pad_rgb(
    image: RGBImage,
    crop_box_xyxy: tuple[float, float, float, float],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> RGBImage:
    """Crop and resize an RGB image the same way the training pipeline does."""
    image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    cropped = _crop_image_with_xyxy(image_tensor, crop_box_xyxy)
    # Resize with pad to preserve aspect ratio and avoid warping the needle angle.
    resized = tf.image.resize_with_pad(
        tf.cast(cropped, tf.float32),
        image_size,
        image_size,
        method="bilinear",
    )
    return np.clip(np.rint(resized.numpy()), 0.0, 255.0).astype(np.uint8)


def _summarize_luma(luma: NDArray[np.uint8], box: CropBox) -> BoardCropEstimate:
    """Compute luma statistics for a crop box so the report can explain itself."""
    crop = luma[box.y_min : box.y_max, box.x_min : box.x_max]
    center_x = min(max(box.x_min + box.width // 2, box.x_min), box.x_max - 1)
    center_y = min(max(box.y_min + box.height // 2, box.y_min), box.y_max - 1)
    center_luma = int(luma[center_y, center_x])
    return BoardCropEstimate(
        crop_box=box,
        center_luma=center_luma,
        mean_luma=float(crop.mean()),
        min_luma=int(crop.min()),
        max_luma=int(crop.max()),
    )


def estimate_board_crop_from_rgb(
    image: RGBImage,
    *,
    bright_threshold: int = BOARD_BRIGHT_THRESHOLD,
    border_pixels: int = BOARD_BORDER_PIXELS,
) -> BoardCropEstimate | None:
    """Estimate the board crop using the same bright-luma heuristic as firmware."""
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image, got shape {image.shape}.")
    if height <= 2 * border_pixels or width <= 2 * border_pixels:
        return None

    luma = rgb_to_luma(image)
    inner = luma[
        border_pixels : height - border_pixels,
        border_pixels : width - border_pixels,
    ]
    bright_mask = inner >= bright_threshold
    if not np.any(bright_mask):
        return None

    bright_y, bright_x = np.nonzero(bright_mask)
    bright_x = bright_x + border_pixels
    bright_y = bright_y + border_pixels
    centroid_x = int(np.mean(bright_x))
    centroid_y = int(np.mean(bright_y))
    bright_count = int(bright_x.size)

    crop_width = max(
        1,
        int(round(width * (TRAINING_CROP_X_MAX_RATIO - TRAINING_CROP_X_MIN_RATIO))),
    )
    crop_height = max(
        1,
        int(round(height * (TRAINING_CROP_Y_MAX_RATIO - TRAINING_CROP_Y_MIN_RATIO))),
    )

    left = max(0, centroid_x - (crop_width // 2))
    top = max(0, centroid_y - (crop_height // 2))
    right = left + crop_width
    bottom = top + crop_height
    if right > width:
        right = width
        left = max(0, right - crop_width)
    if bottom > height:
        bottom = height
        top = max(0, bottom - crop_height)

    if right <= left or bottom <= top:
        return None

    box = CropBox(
        x_min=left,
        y_min=top,
        width=right - left,
        height=bottom - top,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        bright_count=bright_count,
    )
    return _summarize_luma(luma, box)


def _save_rgb_image(image: RGBImage, path: Path) -> None:
    """Save one RGB image without introducing extra plotting noise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def _render_comparison_figure(
    *,
    source_image: RGBImage,
    training_crop_box: tuple[float, float, float, float],
    board_estimate: BoardCropEstimate,
    training_crop: RGBImage,
    board_crop: RGBImage,
    comparison_figure_path: Path,
) -> None:
    """Write a side-by-side figure that shows both crop paths and their delta."""
    comparison_figure_path.parent.mkdir(parents=True, exist_ok=True)
    diff = np.abs(training_crop.astype(np.int16) - board_crop.astype(np.int16))
    diff_gray = diff.mean(axis=2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    source_ax = axes[0, 0]
    source_ax.imshow(source_image)
    source_ax.set_title("Source image")
    source_ax.axis("off")

    training_rect = patches.Rectangle(
        (training_crop_box[0], training_crop_box[1]),
        training_crop_box[2] - training_crop_box[0],
        training_crop_box[3] - training_crop_box[1],
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
        label="training crop",
    )
    board_box = board_estimate.crop_box
    board_rect = patches.Rectangle(
        (board_box.x_min, board_box.y_min),
        board_box.width,
        board_box.height,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        label="board crop",
    )
    source_ax.add_patch(training_rect)
    source_ax.add_patch(board_rect)
    source_ax.legend(loc="lower right")

    training_ax = axes[0, 1]
    training_ax.imshow(training_crop)
    training_ax.set_title("Training crop")
    training_ax.axis("off")

    board_ax = axes[1, 0]
    board_ax.imshow(board_crop)
    board_ax.set_title("Board crop")
    board_ax.axis("off")

    diff_ax = axes[1, 1]
    diff_im = diff_ax.imshow(diff_gray, cmap="magma")
    diff_ax.set_title("Absolute diff")
    diff_ax.axis("off")
    fig.colorbar(diff_im, ax=diff_ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(comparison_figure_path, dpi=160)
    plt.close(fig)


def compare_labelled_sample(
    sample: Sample,
    output_dir: Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_pad_ratio: float = 0.1,
) -> ComparisonReport:
    """Run both crop paths on one labeled image and write a visual report."""
    source_image = load_rgb_image(sample.image_path)
    training_crop_box = compute_training_crop_box(sample, crop_pad_ratio)
    board_estimate = estimate_board_crop_from_rgb(source_image)
    if board_estimate is None:
        raise ValueError(
            f"Board crop estimate failed for {sample.image_path}. "
            "The image did not produce a bright enough dial region."
        )

    training_crop = resize_with_pad_rgb(source_image, training_crop_box, image_size)
    board_crop = resize_with_pad_rgb(
        source_image,
        (
            float(board_estimate.crop_box.x_min),
            float(board_estimate.crop_box.y_min),
            float(board_estimate.crop_box.x_max),
            float(board_estimate.crop_box.y_max),
        ),
        image_size,
    )

    comparison_dir = output_dir / sample.image_path.stem
    comparison_dir.mkdir(parents=True, exist_ok=True)
    training_crop_path = comparison_dir / "training_crop.png"
    board_crop_path = comparison_dir / "board_crop.png"
    comparison_figure_path = comparison_dir / "comparison.png"
    report_json_path = comparison_dir / "report.json"

    _save_rgb_image(training_crop, training_crop_path)
    _save_rgb_image(board_crop, board_crop_path)
    _render_comparison_figure(
        source_image=source_image,
        training_crop_box=training_crop_box,
        board_estimate=board_estimate,
        training_crop=training_crop,
        board_crop=board_crop,
        comparison_figure_path=comparison_figure_path,
    )

    mean_abs_diff = float(
        np.mean(
            np.abs(training_crop.astype(np.int16) - board_crop.astype(np.int16))
        )
    )
    report = ComparisonReport(
        source_image_path=sample.image_path,
        output_dir=comparison_dir,
        training_crop_box=training_crop_box,
        board_crop_estimate=board_estimate,
        training_crop_path=training_crop_path,
        board_crop_path=board_crop_path,
        comparison_figure_path=comparison_figure_path,
        report_json_path=report_json_path,
        mean_abs_diff=mean_abs_diff,
    )

    report_json = {
        "source_image_path": str(report.source_image_path),
        "output_dir": str(report.output_dir),
        "training_crop_box": list(report.training_crop_box),
        "board_crop_estimate": {
            "crop_box": asdict(report.board_crop_estimate.crop_box),
            "center_luma": report.board_crop_estimate.center_luma,
            "mean_luma": report.board_crop_estimate.mean_luma,
            "min_luma": report.board_crop_estimate.min_luma,
            "max_luma": report.board_crop_estimate.max_luma,
        },
        "training_crop_path": str(report.training_crop_path),
        "board_crop_path": str(report.board_crop_path),
        "comparison_figure_path": str(report.comparison_figure_path),
        "mean_abs_diff": report.mean_abs_diff,
    }
    report_json_path.write_text(
        json.dumps(report_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _render_board_capture_figure(
    *,
    source_image: RGBImage,
    board_estimate: BoardCropEstimate,
    board_crop: RGBImage,
    comparison_figure_path: Path,
) -> None:
    """Write a compact figure for a raw board capture and the detected crop."""
    comparison_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    source_ax = axes[0]
    source_ax.imshow(source_image)
    source_ax.set_title("Board capture")
    source_ax.axis("off")

    board_box = board_estimate.crop_box
    board_rect = patches.Rectangle(
        (board_box.x_min, board_box.y_min),
        board_box.width,
        board_box.height,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        label="board crop",
    )
    source_ax.add_patch(board_rect)
    source_ax.legend(loc="lower right")

    crop_ax = axes[1]
    crop_ax.imshow(board_crop)
    crop_ax.set_title("Board crop")
    crop_ax.axis("off")

    fig.tight_layout()
    fig.savefig(comparison_figure_path, dpi=160)
    plt.close(fig)


def compare_board_capture(
    capture_path: Path,
    output_dir: Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    image_width: int = DEFAULT_IMAGE_SIZE,
    image_height: int = DEFAULT_IMAGE_SIZE,
) -> BoardCaptureReport:
    """Run the board crop estimator on a raw YUV422 board capture."""
    source_image = load_yuv422_capture_as_rgb(
        capture_path,
        image_width=image_width,
        image_height=image_height,
    )
    board_estimate = estimate_board_crop_from_rgb(source_image)
    if board_estimate is None:
        raise ValueError(
            f"Board crop estimate failed for {capture_path}. "
            "The capture did not produce a bright enough dial region."
        )

    board_crop = resize_with_pad_rgb(
        source_image,
        (
            float(board_estimate.crop_box.x_min),
            float(board_estimate.crop_box.y_min),
            float(board_estimate.crop_box.x_max),
            float(board_estimate.crop_box.y_max),
        ),
        image_size,
    )

    capture_dir = output_dir / capture_path.stem
    capture_dir.mkdir(parents=True, exist_ok=True)
    capture_preview_path = capture_dir / "capture_preview.png"
    capture_crop_path = capture_dir / "board_crop.png"
    comparison_figure_path = capture_dir / "comparison.png"
    report_json_path = capture_dir / "report.json"

    _save_rgb_image(source_image, capture_preview_path)
    _save_rgb_image(board_crop, capture_crop_path)
    _render_board_capture_figure(
        source_image=source_image,
        board_estimate=board_estimate,
        board_crop=board_crop,
        comparison_figure_path=comparison_figure_path,
    )

    report = BoardCaptureReport(
        capture_path=capture_path,
        output_dir=capture_dir,
        board_crop_estimate=board_estimate,
        capture_preview_path=capture_preview_path,
        capture_crop_path=capture_crop_path,
        comparison_figure_path=comparison_figure_path,
        report_json_path=report_json_path,
    )

    report_json = {
        "capture_path": str(report.capture_path),
        "output_dir": str(report.output_dir),
        "board_crop_estimate": {
            "crop_box": asdict(report.board_crop_estimate.crop_box),
            "center_luma": report.board_crop_estimate.center_luma,
            "mean_luma": report.board_crop_estimate.mean_luma,
            "min_luma": report.board_crop_estimate.min_luma,
            "max_luma": report.board_crop_estimate.max_luma,
        },
        "capture_preview_path": str(report.capture_preview_path),
        "capture_crop_path": str(report.capture_crop_path),
        "comparison_figure_path": str(report.comparison_figure_path),
    }
    report_json_path.write_text(
        json.dumps(report_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
