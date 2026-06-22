"""Decoding utilities for CenterNet heatmap predictions.

Converts heatmap + offset outputs to center point coordinates with
sub-pixel refinement, following the Objects as Points decoding procedure.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def decode_centernet_heatmap(
    heatmap: np.ndarray,
    offset: np.ndarray | None = None,
    *,
    method: str = "argmax_topk",
    topk: int = 1,
    min_score: float = 0.1,
) -> list[tuple[float, float, float]]:
    """Decode a single CenterNet heatmap into (x, y, score) detections.

    Args:
        heatmap: Predicted heatmap, shape (H, W) or (H, W, 1).
        offset: Predicted offset map, shape (H, W, 2) or None.
        method: Decoding method — 'argmax_topk' uses the top-k peaks.
        topk: Maximum number of detections to return.
        min_score: Minimum heatmap score threshold.

    Returns:
        List of (cx_hm, cy_hm, score) in heatmap pixel coordinates.
        If offset is provided, (cx, cy) includes sub-pixel refinement.
    """
    if heatmap.ndim == 3:
        heatmap = np.squeeze(heatmap, axis=-1)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")

    # 3x3 max-pool to find local peaks (CenterNet uses 3x3 NMS).
    pooled = _max_pool_2d_numpy(heatmap, kernel_size=3)

    # Find local peaks that match the max-pooled values.
    peak_mask = np.equal(heatmap, pooled)
    peak_mask = np.logical_and(peak_mask, heatmap >= min_score)

    # Extract peak coordinates and scores.
    ys, xs = np.where(peak_mask)
    scores = heatmap[ys, xs]

    # Sort by score descending and take topk.
    order = np.argsort(-scores)
    ys = ys[order][:topk]
    xs = xs[order][:topk]
    scores = scores[order][:topk]

    results = []
    for i in range(len(xs)):
        cx = float(xs[i])
        cy = float(ys[i])
        # Apply offset refinement if available.
        if offset is not None and offset.ndim >= 2:
            off_x = offset[int(cy), int(cx), 0]
            off_y = offset[int(cy), int(cx), 1]
            cx += off_x
            cy += off_y
        results.append((cx, cy, float(scores[i])))

    return results


def _max_pool_2d_numpy(arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Simple 2D max pooling using numpy for CenterNet NMS.

    Uses a sliding-window approach without scipy to avoid an extra dependency.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    pad = kernel_size // 2
    padded = np.pad(arr, pad, mode="constant", constant_values=0)
    out = np.zeros_like(arr)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            out = np.maximum(
                out, padded[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
            )
    return out


def decode_centernet_batch(
    heatmaps: np.ndarray,
    offsets: np.ndarray | None = None,
    *,
    topk: int = 1,
    min_score: float = 0.1,
) -> list[list[tuple[float, float, float]]]:
    """Decode a batch of CenterNet heatmaps.

    Args:
        heatmaps: Predicted heatmaps, shape (B, H, W, 1).
        offsets: Predicted offsets, shape (B, H, W, 2) or None.
        topk: Maximum detections per image.
        min_score: Minimum heatmap score threshold.

    Returns:
        List of lists: one list of (cx, cy, score) per batch element.
    """
    batch_size = heatmaps.shape[0]
    results = []
    for i in range(batch_size):
        hm = heatmaps[i]
        off = offsets[i] if offsets is not None else None
        detections = decode_centernet_heatmap(
            hm, off, topk=topk, min_score=min_score
        )
        results.append(detections)
    return results


def centernet_nms(
    detections: list[tuple[float, float, float]],
    nms_threshold: float = 0.5,
    kernel_size: int = 3,
) -> list[tuple[float, float, float]]:
    """Simple distance-based NMS for CenterNet detections.

    For gauge center detection, we typically only need the top-1 detection,
    but this is provided for multi-center scenarios.

    Args:
        detections: List of (cx, cy, score) tuples.
        nms_threshold: Distance threshold in heatmap pixels for suppression.
        kernel_size: Not used for distance NMS (kept for API compatibility).

    Returns:
        Filtered list of detections.
    """
    if len(detections) <= 1:
        return detections

    # Sort by score descending.
    detections = sorted(detections, key=lambda d: d[2], reverse=True)
    kept = []
    suppressed = set()

    for i, (cx_i, cy_i, score_i) in enumerate(detections):
        if i in suppressed:
            continue
        kept.append((cx_i, cy_i, score_i))
        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue
            cx_j, cy_j, _ = detections[j]
            dist = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
            if dist < nms_threshold:
                suppressed.add(j)

    return kept


def heatmap_to_canvas_coords(
    cx_hm: float,
    cy_hm: float,
    output_stride: int = 4,
) -> tuple[float, float]:
    """Convert heatmap pixel coordinates to input canvas pixel coordinates.

    Args:
        cx_hm: Center x in heatmap pixel space.
        cy_hm: Center y in heatmap pixel space.
        output_stride: Stride from input to heatmap (e.g. 4 for 384→96).

    Returns:
        (cx_canvas, cy_canvas) in input canvas pixel space.
    """
    return cx_hm * output_stride, cy_hm * output_stride


def canvas_to_source_coords(
    cx_canvas: float,
    cy_canvas: float,
    crop_box_xyxy: tuple[float, float, float, float],
    canvas_h: int = 384,
    canvas_w: int = 384,
) -> tuple[float, float]:
    """Invert the crop+resize mapping back to source image coordinates.

    Args:
        cx_canvas: Center x in canvas pixel space.
        cy_canvas: Center y in canvas pixel space.
        crop_box_xyxy: (x1, y1, x2, y2) crop box in source coords.
        canvas_h: Canvas height.
        canvas_w: Canvas width.

    Returns:
        (cx_source, cy_source) in original source image pixel space.
    """
    x1, y1, x2, y2 = crop_box_xyxy
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)

    scale = min(canvas_w / crop_w, canvas_h / crop_h)
    pad_x = (canvas_w - crop_w * scale) * 0.5
    pad_y = (canvas_h - crop_h * scale) * 0.5

    cx_in_crop = (cx_canvas - pad_x) / scale
    cy_in_crop = (cy_canvas - pad_y) / scale

    return cx_in_crop + x1, cy_in_crop + y1
