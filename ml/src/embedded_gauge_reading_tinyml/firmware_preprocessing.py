"""Firmware-style preprocessing helpers for the STM32 polar-vote path.

This module mirrors the live board path closely enough to replay the current
firmware tensor on laptop captures:
- fixed training crop,
- inverse resize-with-pad geometry,
- polar projection around the inner dial center,
- 7-channel int8 tensor construction, and
- circular vote decoding back into Celsius.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
import math
import sys
from typing import Final, Literal
import zlib
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from embedded_gauge_reading_tinyml.baseline_classical_cv import run_classical_baseline
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec
from embedded_gauge_reading_tinyml.board_crop_compare import resize_with_pad_rgb
from embedded_gauge_reading_tinyml.polar_projection import polar_project_image

RGBImage = NDArray[np.uint8]
FirmwareTensor = NDArray[np.int8]
PolarInputMode = Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"]

DEFAULT_IMAGE_SIZE: Final[int] = 224
CAPTURE_YUV422_SUFFIX: Final[str] = ".yuv422"
CAPTURE_RAW16_SUFFIX: Final[str] = ".raw16"

TRAINING_CROP_X_MIN_RATIO: Final[float] = 0.1027
TRAINING_CROP_Y_MIN_RATIO: Final[float] = 0.2573
TRAINING_CROP_X_MAX_RATIO: Final[float] = 0.7987
TRAINING_CROP_Y_MAX_RATIO: Final[float] = 0.8071

POLAR_VOTE_BINS: Final[int] = 224
POLAR_VOTE_MIN_ANGLE_RAD: Final[float] = 2.356
POLAR_VOTE_SWEEP_RAD: Final[float] = 4.712
POLAR_VOTE_MIN_VALUE_C: Final[float] = -30.0
POLAR_VOTE_MAX_VALUE_C: Final[float] = 50.0
POLAR_OUTPUT_SCALE: Final[float] = 0.093767159
POLAR_OUTPUT_ZERO_POINT: Final[int] = 16
POLAR_MASK_LOGIT: Final[float] = -1.0e9


@dataclass(frozen=True, slots=True)
class TensorProbe:
    """Compact firmware-style metadata for a tensor buffer."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_length: int
    crc32_hex: str
    first8: tuple[int, ...]
    mid8: tuple[int, ...]
    last8: tuple[int, ...]


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
    """Load a packed YUV422 YUYV capture and convert to full-color RGB."""
    raw_bytes = capture_path.read_bytes()
    expected_bytes = image_width * image_height * 2
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Unexpected YUV422 size for {capture_path}: "
            f"got {len(raw_bytes)} bytes, expected {expected_bytes}."
        )

    yuyv = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
        image_height, image_width // 2, 4
    )
    # YUYV: [Y0, U, Y1, V] per 4 bytes — U/V shared across pixel pair
    y0 = yuyv[:, :, 0].astype(np.float32)
    y1 = yuyv[:, :, 2].astype(np.float32)
    u  = yuyv[:, :, 1].astype(np.float32) - 128.0
    v  = yuyv[:, :, 3].astype(np.float32) - 128.0

    r0 = np.clip(y0 + 1.402 * v, 0, 255).astype(np.uint8)
    g0 = np.clip(y0 - 0.344 * u - 0.714 * v, 0, 255).astype(np.uint8)
    b0 = np.clip(y0 + 1.772 * u, 0, 255).astype(np.uint8)
    r1 = np.clip(y1 + 1.402 * v, 0, 255).astype(np.uint8)
    g1 = np.clip(y1 - 0.344 * u - 0.714 * v, 0, 255).astype(np.uint8)
    b1 = np.clip(y1 + 1.772 * u, 0, 255).astype(np.uint8)

    rgb = np.empty((image_height, image_width, 3), dtype=np.uint8)
    rgb[:, 0::2, 0] = r0
    rgb[:, 0::2, 1] = g0
    rgb[:, 0::2, 2] = b0
    rgb[:, 1::2, 0] = r1
    rgb[:, 1::2, 1] = g1
    rgb[:, 1::2, 2] = b1
    return rgb


def resize_with_pad_rgb(
    image: RGBImage,
    crop_box_xyxy: tuple[float, float, float, float],
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> RGBImage:
    """Crop and resize with padding using Pillow."""
    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = crop_box_xyxy
    x_min_i = max(0, int(math.floor(x_min)))
    y_min_i = max(0, int(math.floor(y_min)))
    x_max_i = min(image_width, int(math.ceil(x_max)))
    y_max_i = min(image_height, int(math.ceil(y_max)))
    if x_max_i <= x_min_i:
        x_max_i = min(image_width, x_min_i + 1)
    if y_max_i <= y_min_i:
        y_max_i = min(image_height, y_min_i + 1)

    crop = image[y_min_i:y_max_i, x_min_i:x_max_i]
    crop_height, crop_width = crop.shape[:2]
    if crop_height == image_size and crop_width == image_size:
        return np.ascontiguousarray(crop)

    scale = min(float(image_size) / float(crop_width), float(image_size) / float(crop_height))
    resized_width = max(1, int(round(float(crop_width) * scale)))
    resized_height = max(1, int(round(float(crop_height) * scale)))
    resized = Image.fromarray(crop, mode="RGB").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    offset_x = max(0, (image_size - resized_width) // 2)
    offset_y = max(0, (image_size - resized_height) // 2)
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = np.asarray(resized, dtype=np.uint8)
    return canvas


def firmware_training_crop_box(
    width_pixels: int,
    height_pixels: int,
) -> tuple[float, float, float, float]:
    """Return the fixed training crop used by the live firmware."""
    x_min = int(float(width_pixels) * TRAINING_CROP_X_MIN_RATIO)
    y_min = int(float(height_pixels) * TRAINING_CROP_Y_MIN_RATIO)
    width = max(
        1,
        int(float(width_pixels) * (TRAINING_CROP_X_MAX_RATIO - TRAINING_CROP_X_MIN_RATIO)),
    )
    height = max(
        1,
        int(float(height_pixels) * (TRAINING_CROP_Y_MAX_RATIO - TRAINING_CROP_Y_MIN_RATIO)),
    )
    return (
        float(x_min),
        float(y_min),
        float(x_min + width),
        float(y_min + height),
    )


def load_capture_image(
    capture_path: Path,
    *,
    image_width: int = DEFAULT_IMAGE_SIZE,
    image_height: int = DEFAULT_IMAGE_SIZE,
) -> tuple[RGBImage, Literal["rgb", "yuv422", "raw16"]]:
    """Load one board capture or labeled RGB image as RGB."""
    suffix = capture_path.suffix.lower()
    if suffix == CAPTURE_YUV422_SUFFIX:
        return (
            load_yuv422_capture_as_rgb(
                capture_path,
                image_width=image_width,
                image_height=image_height,
            ),
            "yuv422",
        )
    if suffix in {CAPTURE_RAW16_SUFFIX, ".raw"}:
        raw_bytes = capture_path.read_bytes()
        pixel_count = image_width * image_height
        if len(raw_bytes) == pixel_count:
            gray = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
                image_height,
                image_width,
            )
        elif len(raw_bytes) == pixel_count * 2:
            gray16 = np.frombuffer(raw_bytes, dtype="<u2").reshape(
                image_height,
                image_width,
            )
            gray = np.right_shift(gray16, 8).astype(np.uint8)
        else:
            raise ValueError(
                f"Unexpected raw capture size for {capture_path}: "
                f"got {len(raw_bytes)} bytes, expected {pixel_count} or {pixel_count * 2}."
            )
        return np.repeat(gray[:, :, None], 3, axis=2), "raw16"
    return load_rgb_image(capture_path), "rgb"


def _sample_bilinear_rgb(
    source_rgb: NDArray[np.float32],
    src_x: NDArray[np.float32],
    src_y: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Bilinearly sample an RGB image at floating-point coordinates."""
    source_height, source_width = source_rgb.shape[:2]
    x = np.clip(src_x, 0.0, float(source_width - 1))
    y = np.clip(src_y, 0.0, float(source_height - 1))

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, source_width - 1)
    y1 = np.clip(y0 + 1, 0, source_height - 1)

    x0_f = x0.astype(np.float32)
    y0_f = y0.astype(np.float32)
    dx = x - x0_f
    dy = y - y0_f

    top_left = source_rgb[y0, x0]
    top_right = source_rgb[y0, x1]
    bottom_left = source_rgb[y1, x0]
    bottom_right = source_rgb[y1, x1]

    top = top_left * (1.0 - dx)[..., None] + top_right * dx[..., None]
    bottom = bottom_left * (1.0 - dx)[..., None] + bottom_right * dx[..., None]
    return top * (1.0 - dy)[..., None] + bottom * dy[..., None]


def _quantize_unit_float_to_int8(channel: NDArray[np.float32]) -> NDArray[np.int8]:
    """Match the firmware's q = round(x * 255) - 128 quantization."""
    quantized = np.floor(channel * 255.0 + 0.5) - 128.0
    return np.clip(quantized, -128.0, 127.0).astype(np.int8)


def _normalize_polar_channel(channel: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize one polar channel the same way the training pipeline does."""
    hi = float(np.percentile(channel, 99.0))
    if hi <= 1e-6:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip(channel / hi, 0.0, 1.0).astype(np.float32)


def _build_polar_vote_prior_channel(
    luma: NDArray[np.float32],
    grad_theta: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Build the vote-prior channel used by the offline V28 training path."""
    if luma.ndim != 2 or grad_theta.ndim != 2:
        raise ValueError("luma and grad_theta must be 2D arrays.")
    if luma.shape != grad_theta.shape:
        raise ValueError("luma and grad_theta must share the same shape.")

    height, width = luma.shape
    row_positions = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
    radial_mask = ((row_positions >= 0.18) & (row_positions <= 0.95)).astype(np.float32)
    radial_mask = radial_mask[:, None]

    darkness = np.clip((0.62 - luma) / 0.62, 0.0, 1.0).astype(np.float32)
    edge = np.clip(grad_theta, 0.0, 1.0).astype(np.float32)
    evidence = (0.65 * darkness + 0.35 * edge) * radial_mask

    denom = float(np.sum(radial_mask))
    if denom <= 1e-6:
        return np.zeros((height, width, 1), dtype=np.float32)
    mean_evidence = np.sum(evidence, axis=0) / np.float32(denom)

    darkness_threshold = float(np.quantile(luma, 0.38))
    binary_dark = ((luma <= darkness_threshold).astype(np.float32) * radial_mask).astype(
        np.float32
    )
    continuity = np.zeros(width, dtype=np.float32)
    for column in range(width):
        support = binary_dark[:, column]
        run = 0
        best = 0
        for value in support:
            if value > 0.5:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        continuity[column] = np.float32(best / max(1, height))

    score = (0.70 * mean_evidence + 0.30 * continuity).astype(np.float32)
    score_smoothed = (
        0.50 * score
        + 0.25 * np.roll(score, 1)
        + 0.25 * np.roll(score, -1)
    ).astype(np.float32)
    score_smoothed -= np.min(score_smoothed)
    peak = float(np.max(score_smoothed))
    if peak > 1e-6:
        score_smoothed = score_smoothed / np.float32(peak)
    score_map = np.repeat(score_smoothed[None, :], height, axis=0)
    return score_map[..., None].astype(np.float32)


def _score_polar_alignment(polar_rgb: NDArray[np.float32]) -> float:
    """Score how well a polar crop aligns with a dark radial needle trace."""
    luma = (
        0.299 * polar_rgb[..., 0]
        + 0.587 * polar_rgb[..., 1]
        + 0.114 * polar_rgb[..., 2]
    ).astype(np.float32)
    height = luma.shape[0]
    row_positions = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
    radial_mask = ((row_positions >= 0.18) & (row_positions <= 0.95)).astype(np.float32)
    radial_mask = radial_mask[:, None]

    darkness_threshold = float(np.quantile(luma, 0.38))
    binary_dark = ((luma <= darkness_threshold).astype(np.float32) * radial_mask).astype(np.float32)

    continuity_sum = 0.0
    for column in range(binary_dark.shape[1]):
        support = binary_dark[:, column]
        run = 0
        best = 0
        for value in support:
            if value > 0.5:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        continuity_sum += float(best)
    continuity = continuity_sum / float(max(1, binary_dark.shape[1] * height))
    return continuity


def _polar_rgb_to_vote_channels(
    polar_rgb: NDArray[np.float32],
    *,
    input_mode: PolarInputMode,
) -> NDArray[np.float32]:
    """Convert a polar RGB image into the float32 representation used by training."""
    if polar_rgb.ndim != 3 or polar_rgb.shape[2] != 3:
        raise ValueError("polar_rgb must be a 3-channel image.")

    polar_rgb = np.clip(np.asarray(polar_rgb, dtype=np.float32), 0.0, 1.0)
    if input_mode == "rgb":
        return polar_rgb.astype(np.float32)

    luma = (
        0.299 * polar_rgb[..., 0]
        + 0.587 * polar_rgb[..., 1]
        + 0.114 * polar_rgb[..., 2]
    ).astype(np.float32)
    luma = np.clip(luma, 0.0, 1.0)
    grad_r, grad_theta = np.gradient(luma)
    grad_theta = np.abs(grad_theta).astype(np.float32)
    grad_r = np.abs(grad_r).astype(np.float32)

    edge3 = np.stack(
        [
            _normalize_polar_channel(luma),
            _normalize_polar_channel(grad_theta),
            _normalize_polar_channel(grad_r),
        ],
        axis=-1,
    )
    if input_mode == "edge3":
        return edge3.astype(np.float32)

    if input_mode == "rgb_edge6":
        return np.concatenate([polar_rgb, edge3], axis=-1).astype(np.float32)

    vote_prior = _build_polar_vote_prior_channel(luma, edge3[..., 1])
    return np.concatenate([polar_rgb, edge3, vote_prior], axis=-1).astype(np.float32)


def polar_rgb_to_training_features(
    polar_rgb: NDArray[np.float32],
    *,
    input_mode: PolarInputMode = "rgb_edge6_vote7",
) -> NDArray[np.float32]:
    """Public wrapper for the polar training feature stack used by the board path.

    The training script uses this helper so the model sees the same feature
    representation that the firmware will construct at inference time.
    """

    return _polar_rgb_to_vote_channels(polar_rgb, input_mode=input_mode)


def polar_rgb_batch_to_training_features(
    polar_rgb_batch: NDArray[np.float32],
    *,
    input_mode: PolarInputMode = "rgb_edge6_vote7",
) -> NDArray[np.float32]:
    """Convert a batch of polar RGB tensors into the requested feature stack."""

    batch = np.asarray(polar_rgb_batch, dtype=np.float32)
    if batch.ndim != 4 or batch.shape[-1] != 3:
        raise ValueError("polar_rgb_batch must have shape (N, H, W, 3).")

    features = [
        polar_rgb_to_training_features(polar_rgb, input_mode=input_mode)
        for polar_rgb in batch
    ]
    return np.stack(features, axis=0).astype(np.float32)


def _build_polar_vote_tensor_from_polar_rgb(
    polar_rgb: NDArray[np.float32],
) -> FirmwareTensor:
    """Build the 7-channel V28 tensor once the polar RGB image is available."""
    if polar_rgb.ndim != 3 or polar_rgb.shape[2] != 3:
        raise ValueError("polar_rgb must be a 3-channel image.")

    luma = (
        0.299 * polar_rgb[..., 0]
        + 0.587 * polar_rgb[..., 1]
        + 0.114 * polar_rgb[..., 2]
    ).astype(np.float32)
    luma = np.clip(luma, 0.0, 1.0)

    grad_r, grad_theta = np.gradient(luma)
    grad_theta = np.abs(grad_theta).astype(np.float32)
    grad_r = np.abs(grad_r).astype(np.float32)

    edge3 = np.stack(
        [
            _normalize_polar_channel(luma),
            _normalize_polar_channel(grad_theta),
            _normalize_polar_channel(grad_r),
        ],
        axis=-1,
    )
    vote_prior = _build_polar_vote_prior_channel(luma, edge3[..., 1])

    tensor = np.concatenate([polar_rgb, edge3, vote_prior], axis=-1)
    return np.ascontiguousarray(
        np.stack(
            [
                _quantize_unit_float_to_int8(tensor[..., 0]),
                _quantize_unit_float_to_int8(tensor[..., 1]),
                _quantize_unit_float_to_int8(tensor[..., 2]),
                _quantize_unit_float_to_int8(tensor[..., 3]),
                _quantize_unit_float_to_int8(tensor[..., 4]),
                _quantize_unit_float_to_int8(tensor[..., 5]),
                _quantize_unit_float_to_int8(tensor[..., 6]),
            ],
            axis=-1,
        )
    )


def build_training_style_polar_vote_tensor(
    source_image: RGBImage,
    *,
    crop_box_xyxy: tuple[float, float, float, float] | None = None,
    output_dim: int = DEFAULT_IMAGE_SIZE,
) -> FirmwareTensor:
    """Build the exact offline crop -> resize_with_pad -> polar V28 tensor."""
    if source_image.ndim != 3 or source_image.shape[2] != 3:
        raise ValueError("source_image must be an RGB image.")
    if output_dim != DEFAULT_IMAGE_SIZE:
        raise ValueError("The firmware path is hard-coded for 224x224 tensors.")

    source_height, source_width = source_image.shape[:2]
    if crop_box_xyxy is None:
        crop_box_xyxy = firmware_training_crop_box(source_width, source_height)

    cropped = resize_with_pad_rgb(source_image, crop_box_xyxy, image_size=output_dim)
    polar = polar_project_image(
        cropped,
        center_xy=(float(output_dim) * 0.5, float(output_dim) * 0.5),
        max_radius=float(output_dim) * 0.5,
        polar_size=output_dim,
    )
    return _build_polar_vote_tensor_from_polar_rgb(np.asarray(polar, dtype=np.float32))


def build_training_style_polar_vote_float32(
    source_image: RGBImage,
    *,
    crop_box_xyxy: tuple[float, float, float, float] | None = None,
    output_dim: int = DEFAULT_IMAGE_SIZE,
    input_mode: Literal["rgb", "edge3", "rgb_edge6", "rgb_edge6_vote7"] = "rgb_edge6_vote7",
    center_search_px: int = 5,
    center_mode: Literal["image_center", "classical_baseline"] = "image_center",
    gauge_spec: GaugeSpec | None = None,
) -> NDArray[np.float32]:
    """Build the exact float32 training tensor used by the offline V28 runs."""
    training_module = _load_training_script_module()
    image = np.asarray(source_image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("source_image must be an RGB image.")
    if crop_box_xyxy is not None:
        crop_box = tuple(float(value) for value in crop_box_xyxy)
    else:
        crop_box = None
    return np.asarray(
        training_module._crop_and_polar_image(  # type: ignore[attr-defined]
            image_path=_array_to_temp_path(image),
            crop_box=crop_box,
            polar_size=output_dim,
            input_mode=input_mode,
            center_search_px=center_search_px,
            center_mode=center_mode,
            gauge_spec=gauge_spec,
        ),
        dtype=np.float32,
    )


@lru_cache(maxsize=1)
def _load_training_script_module() -> ModuleType:
    """Load the training CLI as a module so we can reuse its exact helper."""
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "ml" / "scripts" / "train_polar_angle_classifier_manifest.py"
    spec = importlib.util.spec_from_file_location("embedded_gauge_training_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _array_to_temp_path(image: NDArray[np.uint8]) -> Path:
    """Store an RGB array in a temporary PNG so the training helper can reuse it."""
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temp_path = Path(handle.name)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(temp_path)
    return temp_path


def build_legacy_firmware_polar_vote_tensor(
    source_image: RGBImage,
    *,
    crop_box_xyxy: tuple[float, float, float, float] | None = None,
    output_dim: int = DEFAULT_IMAGE_SIZE,
) -> FirmwareTensor:
    """Build the legacy direct-sampled 224x224x7 int8 polar-vote tensor."""
    if source_image.ndim != 3 or source_image.shape[2] != 3:
        raise ValueError("source_image must be an RGB image.")
    if output_dim != DEFAULT_IMAGE_SIZE:
        raise ValueError("The firmware path is hard-coded for 224x224 tensors.")

    source_height, source_width = source_image.shape[:2]
    if crop_box_xyxy is None:
        crop_box_xyxy = firmware_training_crop_box(source_width, source_height)

    crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop_box_xyxy
    crop_width = max(crop_x_max - crop_x_min, 1.0)
    crop_height = max(crop_y_max - crop_y_min, 1.0)

    # Mirror the firmware's inverse resize_with_pad math exactly.
    resize_scale = float(output_dim) / max(crop_width, crop_height)
    resized_w = crop_width * resize_scale
    resized_h = crop_height * resize_scale
    pad_x = (float(output_dim) - resized_w) * 0.5
    pad_y = (float(output_dim) - resized_h) * 0.5

    source_rgb = np.asarray(source_image, dtype=np.float32) / 255.0

    rows = np.arange(output_dim, dtype=np.float32)
    cols = np.arange(output_dim, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    two_pi = np.float32(2.0 * math.pi)
    angles = two_pi * col_grid / np.float32(POLAR_VOTE_BINS)
    radii = (row_grid / np.float32(output_dim)) * np.float32(output_dim // 2)

    src_x_resized = np.float32(output_dim // 2) + (radii * np.cos(angles))
    src_y_resized = np.float32(output_dim // 2) - (radii * np.sin(angles))
    in_pad = (
        (src_x_resized >= pad_x)
        & (src_x_resized < (pad_x + resized_w))
        & (src_y_resized >= pad_y)
        & (src_y_resized < (pad_y + resized_h))
    )

    src_x_crop = (src_x_resized - pad_x + 0.5) / resize_scale - 0.5
    src_y_crop = (src_y_resized - pad_y + 0.5) / resize_scale - 0.5
    src_x = src_x_crop + crop_x_min
    src_y = src_y_crop + crop_y_min

    polar_rgb = _sample_bilinear_rgb(source_rgb, src_x, src_y)
    polar_rgb[~in_pad] = 0.0
    polar_rgb = np.clip(polar_rgb, 0.0, 1.0)

    return _build_polar_vote_tensor_from_polar_rgb(np.asarray(polar_rgb, dtype=np.float32))


def build_firmware_polar_vote_tensor(
    source_image: RGBImage,
    *,
    crop_box_xyxy: tuple[float, float, float, float] | None = None,
    output_dim: int = DEFAULT_IMAGE_SIZE,
) -> FirmwareTensor:
    """Build the current board-style 224x224x7 int8 polar-vote tensor."""
    return build_training_style_polar_vote_tensor(
        source_image,
        crop_box_xyxy=crop_box_xyxy,
        output_dim=output_dim,
    )


def decode_circular_vote_logits(
    logits: NDArray[np.float32] | NDArray[np.int8] | NDArray[np.uint8],
    gauge_spec: GaugeSpec,
    *,
    output_scale: float = POLAR_OUTPUT_SCALE,
    output_zero_point: int = POLAR_OUTPUT_ZERO_POINT,
    bins: int = POLAR_VOTE_BINS,
) -> float:
    """Decode firmware circular vote logits into engineering units."""
    flat = np.asarray(logits).reshape(-1)
    if flat.size < bins:
        raise ValueError(f"Expected at least {bins} logits, got {flat.size}.")

    if np.issubdtype(flat.dtype, np.floating):
        dequantized = flat[:bins].astype(np.float32)
    else:
        dequantized = (
            (flat[:bins].astype(np.float32) - np.float32(output_zero_point))
            * np.float32(output_scale)
        )

    angles = (2.0 * math.pi * np.arange(bins, dtype=np.float32)) / np.float32(bins)
    shifted = np.mod(angles - np.float32(gauge_spec.min_angle_rad), 2.0 * math.pi)
    masked = np.where(
        shifted > (np.float32(gauge_spec.sweep_rad) + 1.0e-6),
        np.float32(POLAR_MASK_LOGIT),
        dequantized,
    )

    max_logit = float(np.max(masked))
    if not math.isfinite(max_logit):
        raise ValueError("Circular vote logits did not contain a finite maximum.")

    exp_vals = np.exp(masked - np.float32(max_logit))
    sum_exp = float(np.sum(exp_vals))
    if sum_exp <= 0.0 or not math.isfinite(sum_exp):
        raise ValueError("Circular vote softmax underflowed.")

    probs = exp_vals / np.float32(sum_exp)
    sin_sum = float(np.sum(probs * np.sin(angles)))
    cos_sum = float(np.sum(probs * np.cos(angles)))
    mean_angle = math.atan2(sin_sum, cos_sum)
    if mean_angle < 0.0:
        mean_angle += 2.0 * math.pi

    fraction = (mean_angle - gauge_spec.min_angle_rad) / gauge_spec.sweep_rad
    if fraction < 0.0:
        fraction += (2.0 * math.pi) / gauge_spec.sweep_rad
    fraction = min(max(fraction, 0.0), 1.0)
    return gauge_spec.min_value + fraction * (gauge_spec.max_value - gauge_spec.min_value)


def probe_tensor(name: str, tensor: NDArray[np.generic]) -> TensorProbe:
    """Create a firmware-style checksum and byte summary for one tensor."""
    array = np.ascontiguousarray(np.asarray(tensor))
    raw = array.tobytes()
    raw_bytes = np.frombuffer(raw, dtype=np.uint8)
    byte_length = int(raw_bytes.size)
    first8 = tuple(int(value) for value in raw_bytes[:8].tolist())
    mid_start = max(0, (byte_length // 2) - 4)
    mid8 = tuple(int(value) for value in raw_bytes[mid_start : mid_start + 8].tolist())
    last8 = tuple(int(value) for value in raw_bytes[-8:].tolist()) if byte_length >= 8 else first8
    return TensorProbe(
        name=name,
        dtype=str(array.dtype),
        shape=tuple(int(dim) for dim in array.shape),
        byte_length=byte_length,
        crc32_hex=f"0x{zlib.crc32(raw) & 0xFFFFFFFF:08X}",
        first8=first8,
        mid8=mid8,
        last8=last8,
    )
