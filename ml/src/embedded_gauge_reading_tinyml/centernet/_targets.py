"""Target generation for CenterNet gauge center detection.

Provides utilities for computing Gaussian radius from bounding box size
and drawing Gaussian heatmap peaks, following the Objects as Points paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class GaussianTargetConfig:
    """Configuration for Gaussian heatmap target generation.

    Attributes:
        heatmap_height: Output heatmap height in pixels.
        heatmap_width: Output heatmap width in pixels.
        input_height: Input image height.
        input_width: Input image width.
        sigma_pixels: Gaussian sigma in output heatmap pixels.
        min_overlap: IOU threshold for adaptive Gaussian radius (default 0.7).
    """

    heatmap_height: int = 96
    heatmap_width: int = 96
    input_height: int = 384
    input_width: int = 384
    sigma_pixels: float = 2.0
    min_overlap: float = 0.7

    @property
    def output_stride(self) -> int:
        """Stride from input image to output heatmap."""
        return self.input_height // self.heatmap_height


def gaussian_radius_from_bbox(
    bbox_h: float,
    bbox_w: float,
    min_overlap: float = 0.7,
) -> float:
    """Compute adaptive Gaussian radius from object bounding box size.

    From CornerNet / CenterNet: the radius should be such that a pair of
    points within the radius produce an IOU of at least min_overlap with
    the ground truth bounding box.

    We use the dial_radius_source * 2 as the effective bounding box to
    determine the Gaussian spread.

    Returns sigma in pixels at the output heatmap resolution.
    """
    # Three candidate radii based on the overlap computation.
    # radius = min(r1, r2, r3) where each is derived from IOU geometry.
    height = bbox_h
    width = bbox_w

    # Case 1: corner of predicted bbox inside ground truth bbox.
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = max(b1**2 - 4.0 * a1 * c1, 0.0)
    r1 = (b1 + np.sqrt(sq1)) / (2.0 * a1)

    # Case 2: corner of ground truth bbox inside predicted bbox.
    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    sq2 = max(b2**2 - 4.0 * a2 * c2, 0.0)
    r2 = (b2 + np.sqrt(sq2)) / (2.0 * a2)

    # Case 3: both corners aligned on x or y.
    a3 = 4.0 * min_overlap
    b3 = 2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = max(b3**2 - 4.0 * a3 * c3, 0.0)
    r3 = (b3 + np.sqrt(sq3)) / (2.0 * a3)

    return float(min(r1, r2, r3))


def draw_gaussian_heatmap(
    height: int,
    width: int,
    cx: float,
    cy: float,
    sigma: float,
) -> np.ndarray:
    """Draw a single 2D Gaussian peak at (cx, cy) with pixel sigma.

    Args:
        height: Heatmap height in pixels.
        width: Heatmap width in pixels.
        cx: Center x in heatmap pixel coordinates.
        cy: Center y in heatmap pixel coordinates.
        sigma: Gaussian sigma in heatmap pixels.

    Returns:
        np.ndarray of shape (height, width) with Gaussian peak.
    """
    # Clip center to valid range.
    cx = np.clip(cx, 0, width - 1)
    cy = np.clip(cy, 0, height - 1)

    y = np.arange(height, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    heatmap = np.exp(-dist_sq / (2.0 * sigma**2))
    # Normalize so that the peak is 1.0.
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap


def build_centernet_targets(
    center_x_canvas: float,
    center_y_canvas: float,
    dial_radius_canvas: float,
    config: GaussianTargetConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build CenterNet targets (heatmap, offset) from canvas-space center.

    Args:
        center_x_canvas: Center x in input canvas pixels.
        center_y_canvas: Center y in input canvas pixels.
        dial_radius_canvas: Dial radius in canvas pixels (for sigma).
        config: Gaussian target configuration.

    Returns:
        (heatmap (H_out, W_out), offset_map (H_out, W_out, 2)) as float32 arrays.
    """
    if config is None:
        config = GaussianTargetConfig()

    stride = config.output_stride

    # Map to heatmap grid.
    cx_hm = center_x_canvas / stride
    cy_hm = center_y_canvas / stride

    # Adaptive sigma based on dial radius.
    sigma = max(config.sigma_pixels, dial_radius_canvas / stride * 0.1)

    # Draw heatmap.
    heatmap = draw_gaussian_heatmap(
        config.heatmap_height,
        config.heatmap_width,
        cx_hm,
        cy_hm,
        sigma,
    )

    # Build offset map (only at the integer peak location).
    cx_int = int(np.floor(cx_hm))
    cy_int = int(np.floor(cy_hm))
    offset_map = np.zeros((config.heatmap_height, config.heatmap_width, 2), dtype=np.float32)

    if 0 <= cx_int < config.heatmap_width and 0 <= cy_int < config.heatmap_height:
        offset_map[cy_int, cx_int, 0] = cx_hm - cx_int  # x offset
        offset_map[cy_int, cx_int, 1] = cy_hm - cy_int  # y offset

    return heatmap, offset_map


def build_centernet_targets_tf(
    center_x_canvas: tf.Tensor,
    center_y_canvas: tf.Tensor,
    dial_radius_canvas: tf.Tensor,
    output_stride: int,
    heatmap_h: int,
    heatmap_w: int,
    sigma_pixels: float = 2.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """TensorFlow version of build_centernet_targets for tf.data pipeline.

    Returns (heatmap, offset_map) as tf.Tensors.
    """
    cx_hm = center_x_canvas / tf.cast(output_stride, tf.float32)
    cy_hm = center_y_canvas / tf.cast(output_stride, tf.float32)
    sigma = tf.cast(sigma_pixels, tf.float32)

    # Build Gaussian heatmap using tf ops.
    y = tf.range(tf.cast(heatmap_h, tf.float32))
    x = tf.range(tf.cast(heatmap_w, tf.float32))
    yy, xx = tf.meshgrid(y, x, indexing="ij")
    dist_sq = tf.square(xx - cx_hm) + tf.square(yy - cy_hm)
    heatmap = tf.exp(-dist_sq / (2.0 * tf.square(sigma)))
    heatmap = heatmap / tf.reduce_max(heatmap)

    # Offset map.
    cx_int = tf.cast(tf.math.floor(cx_hm), tf.int32)
    cy_int = tf.cast(tf.math.floor(cy_hm), tf.int32)
    cx_int = tf.clip_by_value(cx_int, 0, heatmap_w - 1)
    cy_int = tf.clip_by_value(cy_int, 0, heatmap_h - 1)

    offset_map = tf.zeros([heatmap_h, heatmap_w, 2], dtype=tf.float32)
    indices = tf.expand_dims(tf.stack([cy_int, cx_int]), axis=0)
    updates = tf.expand_dims(
        tf.stack([cx_hm - tf.cast(cx_int, tf.float32), cy_hm - tf.cast(cy_int, tf.float32)]),
        axis=0,
    )
    offset_map = tf.tensor_scatter_nd_update(offset_map, indices, updates)

    return heatmap, offset_map
