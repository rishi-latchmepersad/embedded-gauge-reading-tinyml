"""Training sub-module for embedded gauge reading.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf


def _compute_fullframe_obb_params(
    source_w: int,
    source_h: int,
    dial_cx: float,
    dial_cy: float,
    dial_rx: float,
    dial_ry: float,
    dial_rotation_deg: float,
    canvas_h: int = 224,
    canvas_w: int = 224,
) -> np.ndarray:
    """Map dial ellipse from source-image coords -> 224x224 canvas normalized.

    Unlike the crop-based pipeline (where cx,cy are always ~0.5), this maps
    into the full-frame canvas so cx,cy vary with photo composition. The model
    must actually detect the gauge rather than predict the mean center.
    """
    scale: float = min(canvas_w / source_w, canvas_h / source_h)
    scaled_w: float = source_w * scale
    scaled_h: float = source_h * scale
    pad_x: float = (canvas_w - scaled_w) * 0.5
    pad_y: float = (canvas_h - scaled_h) * 0.5

    canvas_cx: float = dial_cx * scale + pad_x
    canvas_cy: float = dial_cy * scale + pad_y
    canvas_w_: float = 2.0 * dial_rx * scale
    canvas_h_: float = 2.0 * dial_ry * scale

    rotation_rad: float = math.radians(dial_rotation_deg)
    return np.array(
        [
            np.clip(canvas_cx / canvas_w, 0.0, 1.0),
            np.clip(canvas_cy / canvas_h, 0.0, 1.0),
            np.clip(canvas_w_ / canvas_w, 0.0, 1.0),
            np.clip(canvas_h_ / canvas_h, 0.0, 1.0),
            math.cos(2.0 * rotation_rad),
            math.sin(2.0 * rotation_rad),
        ],
        dtype=np.float32,
    )



def _map_point_to_resized_crop_xy(
    *,
    point_xy: tuple[float, float],
    crop_box_xyxy: tuple[float, float, float, float],
    image_height: int,
    image_width: int,
) -> tuple[float, float]:
    """Map a point from the original image into the resized crop frame.

    We mirror the crop + resize_with_pad geometry used by the input pipeline so
    keypoint heatmaps line up with the network input coordinates.
    """
    x_min, y_min, x_max, y_max = crop_box_xyxy
    crop_w: float = max(x_max - x_min, 1.0)
    crop_h: float = max(y_max - y_min, 1.0)
    scale: float = min(image_width / crop_w, image_height / crop_h)
    resized_w: float = crop_w * scale
    resized_h: float = crop_h * scale
    pad_x: float = 0.5 * float(image_width - resized_w)
    pad_y: float = 0.5 * float(image_height - resized_h)

    point_x, point_y = point_xy
    out_x: float = (point_x - x_min) * scale + pad_x
    out_y: float = (point_y - y_min) * scale + pad_y
    return out_x, out_y


def _make_gaussian_heatmap(
    *,
    point_xy: tuple[float, float],
    heatmap_size: int,
    sigma: float = 1.5,
) -> np.ndarray:
    """Build a single Gaussian heatmap centered on the given point."""
    if heatmap_size < 4:
        raise ValueError("heatmap_size must be >= 4.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")

    center_x, center_y = point_xy
    yy, xx = np.meshgrid(
        np.arange(heatmap_size, dtype=np.float32),
        np.arange(heatmap_size, dtype=np.float32),
        indexing="ij",
    )
    heatmap: np.ndarray = np.exp(
        -((xx - np.float32(center_x)) ** 2 + (yy - np.float32(center_y)) ** 2)
        / (2.0 * np.float32(sigma) ** 2)
    )
    max_value: float = float(np.max(heatmap))
    if max_value > 0.0:
        heatmap = heatmap / max_value
    return heatmap.astype(np.float32)


def _make_keypoint_heatmaps(
    *,
    center_xy: tuple[float, float],
    tip_xy: tuple[float, float],
    heatmap_size: int,
    sigma: float = 1.5,
    extra_keypoints: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """Build the heatmap target used by the keypoint-style models.

    The first two channels stay centered on the dial pivot and the pointer tip.
    Optional extra channels can supervise additional sweep landmarks so the
    reader learns the full gauge arc instead of only the local pointer pose.
    """
    keypoints: list[tuple[float, float]] = [center_xy, tip_xy]
    keypoints.extend(extra_keypoints)
    heatmap_channels: list[np.ndarray] = [
        _make_gaussian_heatmap(
            point_xy=point_xy,
            heatmap_size=heatmap_size,
            sigma=sigma,
        )
        for point_xy in keypoints
    ]
    return np.stack(heatmap_channels, axis=-1)


def _make_pointer_mask(
    *,
    center_xy: tuple[float, float],
    tip_xy: tuple[float, float],
    mask_size: int,
    sigma: float = 1.6,
) -> np.ndarray:
    """Build a soft pointer-shaft mask from the dial center to the tip."""
    if mask_size < 4:
        raise ValueError("mask_size must be >= 4.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")

    center_x, center_y = center_xy
    tip_x, tip_y = tip_xy
    yy, xx = np.meshgrid(
        np.arange(mask_size, dtype=np.float32),
        np.arange(mask_size, dtype=np.float32),
        indexing="ij",
    )
    dx = np.float32(tip_x - center_x)
    dy = np.float32(tip_y - center_y)
    denom = dx * dx + dy * dy
    if float(denom) <= 0.0:
        return np.zeros((mask_size, mask_size, 1), dtype=np.float32)

    # Project each pixel onto the needle segment, then convert the distance to
    # a smooth confidence mask. This is a cheap proxy for a true pointer mask.
    proj_t = ((xx - np.float32(center_x)) * dx + (yy - np.float32(center_y)) * dy) / denom
    proj_t = np.clip(proj_t, 0.0, 1.0)
    proj_x = np.float32(center_x) + proj_t * dx
    proj_y = np.float32(center_y) + proj_t * dy
    dist_sq = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
    mask: np.ndarray = np.exp(-dist_sq / (2.0 * np.float32(sigma) ** 2))
    max_value: float = float(np.max(mask))
    if max_value > 0.0:
        mask = mask / max_value
    return mask[..., np.newaxis].astype(np.float32)



def _make_source_crop_corner_targets(
    *,
    crop_box_xyxy: tuple[float, float, float, float],
    source_image_height: int,
    source_image_width: int,
    image_height: int,
    image_width: int,
    heatmap_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the corner heatmaps, coordinates, and normalized box target.

    The crop box corners are first mapped onto the full-frame canvas used by the
    network input and then normalized so the box head can learn a compact
    0-1 representation.
    """
    source_x0, source_y0, source_x1, source_y1 = crop_box_xyxy
    source_corners: tuple[tuple[float, float], ...] = (
        (source_x0, source_y0),
        (source_x1, source_y0),
        (source_x1, source_y1),
        (source_x0, source_y1),
    )
    canvas_corners = [
        _map_point_to_resized_crop_xy(
            point_xy=corner,
            crop_box_xyxy=(
                0.0,
                0.0,
                float(source_image_width),
                float(source_image_height),
            ),
            image_height=image_height,
            image_width=image_width,
        )
        for corner in source_corners
    ]
    scale_x: float = (heatmap_size - 1.0) / max(image_width - 1.0, 1.0)
    scale_y: float = (heatmap_size - 1.0) / max(image_height - 1.0, 1.0)
    heatmap_channels: list[np.ndarray] = [
        _make_gaussian_heatmap(
            point_xy=(canvas_x * scale_x, canvas_y * scale_y),
            heatmap_size=heatmap_size,
        )
        for canvas_x, canvas_y in canvas_corners
    ]
    corner_heatmaps = np.stack(heatmap_channels, axis=-1)
    corner_coords = np.array(
        [
            [canvas_x * scale_x, canvas_y * scale_y]
            for canvas_x, canvas_y in canvas_corners
        ],
        dtype=np.float32,
    )
    canvas_x_values = np.array([corner[0] for corner in canvas_corners], dtype=np.float32)
    canvas_y_values = np.array([corner[1] for corner in canvas_corners], dtype=np.float32)
    normalized_box = np.array(
        [
            float(np.min(canvas_x_values) / max(image_width, 1)),
            float(np.min(canvas_y_values) / max(image_height, 1)),
            float(np.max(canvas_x_values) / max(image_width, 1)),
            float(np.max(canvas_y_values) / max(image_height, 1)),
        ],
        dtype=np.float32,
    )
    return corner_heatmaps, corner_coords, normalized_box



def _coerce_keypoint_heatmaps(
    heatmaps: np.ndarray | None,
    *,
    heatmap_size: int,
    num_keypoints: int,
) -> np.ndarray:
    """Return a keypoint heatmap tensor with a stable channel count.

    This keeps the OBB-mask and sequence-geometry dataset builders from mixing
    legacy 2-keypoint examples with the new 4-keypoint sweep examples.
    """
    if heatmaps is None:
        return np.zeros((heatmap_size, heatmap_size, num_keypoints), dtype=np.float32)

    arr: np.ndarray = np.asarray(heatmaps, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.shape[:2] != (heatmap_size, heatmap_size):
        raise ValueError(
            "Keypoint heatmap spatial size mismatch: "
            f"expected {(heatmap_size, heatmap_size)}, got {arr.shape[:2]}."
        )
    channels = int(arr.shape[-1])
    if channels == num_keypoints:
        return arr
    if channels > num_keypoints:
        return arr[..., :num_keypoints]

    padded = np.zeros((heatmap_size, heatmap_size, num_keypoints), dtype=np.float32)
    padded[..., :channels] = arr
    return padded



def _coerce_keypoint_coords(
    coords: np.ndarray | None,
    *,
    num_keypoints: int,
) -> np.ndarray:
    """Return a coordinate tensor with a stable keypoint count."""
    if coords is None:
        return np.zeros((num_keypoints, 2), dtype=np.float32)

    arr: np.ndarray = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 2:
        raise ValueError(
            "Keypoint coordinate tensor must have shape (N, 2); "
            f"got {arr.shape}."
        )
    keypoint_count = int(arr.shape[0])
    if keypoint_count == num_keypoints:
        return arr
    if keypoint_count > num_keypoints:
        return arr[:num_keypoints, :]

    padded = np.zeros((num_keypoints, 2), dtype=np.float32)
    padded[:keypoint_count, :] = arr
    return padded


