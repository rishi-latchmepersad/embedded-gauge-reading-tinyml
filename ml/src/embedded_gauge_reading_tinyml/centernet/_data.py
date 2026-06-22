"""Data loading utilities for CenterNet gauge center detection.

Reads the geometry heatmap manifest (CSV with center_x_source,
center_y_source, dial_radius_source) and builds tf.data pipelines
that produce (image, center_heatmap, center_offset) tuples.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

# Resolve repo root relative to this file for path portability.
# _data.py → centernet/ → embedded_gauge_reading_tinyml/ → src/ → ml/ → <repo_root>
_ML_ROOT: Path = Path(__file__).resolve().parents[3]  # ml/ directory
_REPO_ROOT: Path = _ML_ROOT.parent  # git repo root


@dataclass
class GeometryManifestRow:
    """One row of the geometry manifest with the fields needed for CenterNet."""

    image_path: str
    split: str  # train, val, or test
    source_width: int
    source_height: int
    # Loose crop box in source image coords (used for gauge ROI).
    loose_crop_x1: float
    loose_crop_y1: float
    loose_crop_x2: float
    loose_crop_y2: float
    # Center point in source image coords.
    center_x_source: float
    center_y_source: float
    # Dial radius for adaptive Gaussian sigma.
    dial_radius_source: float
    # Temperature value for downstream quality checks.
    temperature_c: float


def load_geometry_manifest(
    csv_path: str | Path,
    splits: tuple[str, ...] = ("train",),
) -> list[GeometryManifestRow]:
    """Load geometry manifest CSV filtered to the requested splits.

    Args:
        csv_path: Path to the geometry heatmap manifest CSV.
                  Relative paths resolve against the ml/ directory.
        splits: Which split values to include (e.g. ('train',)).

    Returns:
        List of GeometryManifestRow for the requested splits.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        # Resolve relative paths from the ml/ directory (where scripts run from).
        csv_path = _ML_ROOT / csv_path
    rows: list[GeometryManifestRow] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "").strip() not in splits:
                continue
            # Skip rows missing center coordinates (can't train CenterNet).
            if not row.get("center_x_source") or not row.get("center_y_source"):
                continue
            rows.append(
                GeometryManifestRow(
                    image_path=row["image_path"],
                    split=row.get("split", "").strip(),
                    source_width=int(row.get("source_width", 0)),
                    source_height=int(row.get("source_height", 0)),
                    loose_crop_x1=float(row.get("loose_crop_x1", 0)),
                    loose_crop_y1=float(row.get("loose_crop_y1", 0)),
                    loose_crop_x2=float(row.get("loose_crop_x2", 0)),
                    loose_crop_y2=float(row.get("loose_crop_y2", 0)),
                    center_x_source=float(row["center_x_source"]),
                    center_y_source=float(row["center_y_source"]),
                    dial_radius_source=float(row.get("dial_radius_source", 100)),
                    temperature_c=float(row.get("temperature_c", 0)),
                )
            )
    return rows


def _resolve_image_path(rel_path: str) -> str:
    """Resolve a manifest-relative image path to an absolute filesystem path.

    Image paths in the manifest are relative to the git repo root
    (e.g. 'ml/data/raw/PXL_...jpg').
    """
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    return str(_REPO_ROOT / rel_path)


def _crop_and_resize(
    image: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    target_h: int,
    target_w: int,
) -> tf.Tensor:
    """Crop image to loose crop box, then resize with pad preserving aspect ratio.

    The loose_crop box is in source-image pixel coordinates. We crop to it
    and then resize the crop into the target canvas using aspect-ratio-preserving
    pad, so the dial circle remains circular and the center point maps linearly.
    """
    shape = tf.shape(image)
    img_h = tf.cast(shape[0], tf.float32)
    img_w = tf.cast(shape[1], tf.float32)

    x1 = tf.clip_by_value(crop_box_xyxy[0], 0.0, img_w - 1.0)
    y1 = tf.clip_by_value(crop_box_xyxy[1], 0.0, img_h - 1.0)
    x2 = tf.clip_by_value(crop_box_xyxy[2], x1 + 1.0, img_w)
    y2 = tf.clip_by_value(crop_box_xyxy[3], y1 + 1.0, img_h)

    ix1 = tf.cast(tf.math.floor(x1), tf.int32)
    iy1 = tf.cast(tf.math.floor(y1), tf.int32)
    ix2 = tf.cast(tf.math.ceil(x2), tf.int32)
    iy2 = tf.cast(tf.math.ceil(y2), tf.int32)

    crop_w = tf.maximum(1, ix2 - ix1)
    crop_h = tf.maximum(1, iy2 - iy1)

    cropped = tf.image.crop_to_bounding_box(image, iy1, ix1, crop_h, crop_w)
    # Preserve aspect ratio so dial circle stays circular.
    resized = tf.image.resize_with_pad(cropped, target_h, target_w)
    return resized


def _map_center_to_canvas(
    center_x_source: tf.Tensor,
    center_y_source: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    source_w: tf.Tensor,
    source_h: tf.Tensor,
    canvas_h: int,
    canvas_w: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map center point from source coords through crop+resize into canvas coords.

    Returns (center_x_canvas, center_y_canvas) in pixel units on the canvas.
    """
    x1 = tf.clip_by_value(crop_box_xyxy[0], 0.0, source_w - 1.0)
    y1 = tf.clip_by_value(crop_box_xyxy[1], 0.0, source_h - 1.0)
    x2 = tf.clip_by_value(crop_box_xyxy[2], x1 + 1.0, source_w)
    y2 = tf.clip_by_value(crop_box_xyxy[3], y1 + 1.0, source_h)

    crop_w = tf.maximum(x2 - x1, 1.0)
    crop_h = tf.maximum(y2 - y1, 1.0)

    # Center relative to crop origin.
    cx_in_crop = center_x_source - x1
    cy_in_crop = center_y_source - y1

    # resize_with_pad scaling: scale = min(canvas_w/crop_w, canvas_h/crop_h).
    scale = tf.minimum(
        tf.cast(canvas_w, tf.float32) / crop_w,
        tf.cast(canvas_h, tf.float32) / crop_h,
    )
    pad_x = (tf.cast(canvas_w, tf.float32) - crop_w * scale) * 0.5
    pad_y = (tf.cast(canvas_h, tf.float32) - crop_h * scale) * 0.5

    cx_canvas = cx_in_crop * scale + pad_x
    cy_canvas = cy_in_crop * scale + pad_y
    return cx_canvas, cy_canvas


def _gaussian_2d(
    height: int,
    width: int,
    cx: tf.Tensor,
    cy: tf.Tensor,
    sigma: tf.Tensor,
) -> tf.Tensor:
    """Build a 2D Gaussian heatmap peak at (cx, cy) with given sigma (pixels).

    Uses tf operations so it runs inside the tf.data graph.
    """
    y = tf.range(tf.cast(height, tf.float32))
    x = tf.range(tf.cast(width, tf.float32))
    yy, xx = tf.meshgrid(y, x, indexing="ij")
    dist_sq = tf.square(xx - cx) + tf.square(yy - cy)
    heatmap = tf.exp(-dist_sq / (2.0 * tf.square(sigma)))
    # Normalize peak to 1.0 so focal loss behaves correctly.
    heatmap = heatmap / tf.reduce_max(heatmap)
    return heatmap


def build_centernet_tf_dataset(
    manifest_rows: list[GeometryManifestRow],
    input_height: int = 384,
    input_width: int = 384,
    heatmap_height: int = 96,
    heatmap_width: int = 96,
    sigma_pixels: float = 2.0,
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = True,
    board_style_prob: float = 0.4,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset producing (image, heatmap, offset) batches.

    Each element:
      image:   (B, H, W, 3) float32 in [0, 1]
      heatmap: (B, H_out, W_out, 1) float32 with Gaussian peak at center
      offset:  (B, H_out, W_out, 2) float32 sub-pixel refinement

    The pipeline:
      1. Read JPEG/PNG from disk.
      2. Crop to loose_crop box, resize with pad to (H, W).
      3. Normalize to [0, 1].
      4. Compute target heatmap + offset at output stride 4.
    """
    # Materialize to Python lists for tf.data.Dataset.from_tensor_slices.
    image_paths = [_resolve_image_path(r.image_path) for r in manifest_rows]
    cx_sources = [r.center_x_source for r in manifest_rows]
    cy_sources = [r.center_y_source for r in manifest_rows]
    radii = [r.dial_radius_source for r in manifest_rows]
    crop_x1s = [r.loose_crop_x1 for r in manifest_rows]
    crop_y1s = [r.loose_crop_y1 for r in manifest_rows]
    crop_x2s = [r.loose_crop_x2 for r in manifest_rows]
    crop_y2s = [r.loose_crop_y2 for r in manifest_rows]
    src_ws = [r.source_width for r in manifest_rows]
    src_hs = [r.source_height for r in manifest_rows]

    # Determine output stride (input / heatmap sizes).
    output_stride = input_height // heatmap_height

    ds = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            cx_sources,
            cy_sources,
            radii,
            crop_x1s,
            crop_y1s,
            crop_x2s,
            crop_y2s,
            src_ws,
            src_hs,
        )
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(manifest_rows), seed=seed)

    def _process(
        img_path: tf.Tensor,
        cx_src: tf.Tensor,
        cy_src: tf.Tensor,
        radius: tf.Tensor,
        cx1: tf.Tensor,
        cy1: tf.Tensor,
        cx2: tf.Tensor,
        cy2: tf.Tensor,
        src_w: tf.Tensor,
        src_h: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Load, crop, resize image and build CenterNet targets."""
        # Read and decode image.
        img_bytes = tf.io.read_file(img_path)
        image = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        image = tf.ensure_shape(image, [None, None, 3])

        crop_box = tf.stack(
            [
                tf.cast(cx1, tf.float32),
                tf.cast(cy1, tf.float32),
                tf.cast(cx2, tf.float32),
                tf.cast(cy2, tf.float32),
            ]
        )
        src_w_f = tf.cast(src_w, tf.float32)
        src_h_f = tf.cast(src_h, tf.float32)

        # Crop and resize image to input canvas.
        image = _crop_and_resize(image, crop_box, input_height, input_width)
        image = tf.cast(image, tf.float32) / 255.0

        # Map center to canvas pixel coords, then to heatmap grid.
        cx_canvas, cy_canvas = _map_center_to_canvas(
            tf.cast(cx_src, tf.float32),
            tf.cast(cy_src, tf.float32),
            crop_box,
            src_w_f,
            src_h_f,
            input_height,
            input_width,
        )

        # Heatmap grid coordinates (at output stride).
        cx_hm = cx_canvas / tf.cast(output_stride, tf.float32)
        cy_hm = cy_canvas / tf.cast(output_stride, tf.float32)

        # Build Gaussian heatmap.
        sigma = tf.cast(sigma_pixels, tf.float32)
        heatmap = _gaussian_2d(
            heatmap_height, heatmap_width, cx_hm, cy_hm, sigma
        )
        heatmap = tf.expand_dims(heatmap, axis=-1)  # (H_out, W_out, 1)

        # Build offset target: sub-pixel refinement at the center location.
        # offset = (cx_hm - floor(cx_hm), cy_hm - floor(cy_hm)) at peak position.
        cx_hm_int = tf.cast(tf.math.floor(cx_hm), tf.int32)
        cy_hm_int = tf.cast(tf.math.floor(cy_hm), tf.int32)
        cx_offset = cx_hm - tf.cast(cx_hm_int, tf.float32)
        cy_offset = cy_hm - tf.cast(cy_hm_int, tf.float32)

        # Offset map: only the integer peak location gets the offset values.
        # We use scatter_nd to place the offset at the quantized center.
        offset_map = tf.zeros([heatmap_height, heatmap_width, 2], dtype=tf.float32)
        # Clamp the integer center to valid heatmap range.
        cx_hm_int = tf.clip_by_value(cx_hm_int, 0, heatmap_width - 1)
        cy_hm_int = tf.clip_by_value(cy_hm_int, 0, heatmap_height - 1)
        indices = tf.stack([cy_hm_int, cx_hm_int])
        indices = tf.expand_dims(indices, axis=0)
        updates = tf.expand_dims(tf.stack([cx_offset, cy_offset]), axis=0)
        offset_map = tf.tensor_scatter_nd_update(
            offset_map, indices, updates
        )

        return image, heatmap, offset_map

    # Concat heatmap+offset into single target tensor (H, W, 3).
    # Matches the model's single concatenated output.
    def _pack_targets(
        img: tf.Tensor, hm: tf.Tensor, off: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        target = tf.concat([hm, off], axis=-1)  # (H, W, 3): hm[:,:,0], off[:,:,1:3]
        return img, target

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        # Bind board_style_prob into the augmentation function.
        def _augment_fn(img, hm, off):
            return _augment_centernet(img, hm, off, board_style_prob)

        ds = ds.map(
            _augment_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Pack targets for model.fit(): (img, (hm, off)).
    ds = ds.map(_pack_targets, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _preprocess_board_style_luma(
    image_rgb: tf.Tensor,
    target_h: int,
    target_w: int,
) -> tf.Tensor:
    """Simulate firmware preprocessing: luma extraction, nearest resize, zero pad.

    Replicates the STM32N6 firmware pipeline:
      Y channel from YUV422 → nearest-neighbor resize → zero pad to 224x224.
    """
    # Image is already in [0,1] float; convert to [0,255] for luma math.
    image_uint8 = tf.cast(image_rgb * 255.0, tf.float32)
    luma = 0.299 * image_uint8[..., 0] + 0.587 * image_uint8[..., 1] + 0.114 * image_uint8[..., 2]
    luma = tf.clip_by_value(luma, 0.0, 255.0)

    shape = tf.shape(luma)
    crop_h = shape[0]
    crop_w = shape[1]
    scale = tf.minimum(
        tf.cast(target_h, tf.float32) / tf.cast(crop_h, tf.float32),
        tf.cast(target_w, tf.float32) / tf.cast(crop_w, tf.float32),
    )
    scaled_h = tf.maximum(tf.cast(tf.cast(crop_h, tf.float32) * scale, tf.int32), 1)
    scaled_w = tf.maximum(tf.cast(tf.cast(crop_w, tf.float32) * scale, tf.int32), 1)

    luma = tf.expand_dims(luma, axis=-1)
    resized = tf.image.resize(luma, [scaled_h, scaled_w], method="nearest")

    pad_y = (target_h - scaled_h) // 2
    pad_x = (target_w - scaled_w) // 2
    pad_bottom = target_h - scaled_h - pad_y
    pad_right = target_w - scaled_w - pad_x
    padded = tf.pad(resized, [[pad_y, pad_bottom], [pad_x, pad_right], [0, 0]])
    # Replicate luma to 3 channels (model expects 3-channel input).
    rgb = tf.tile(padded, [1, 1, 3])
    return tf.cast(rgb, tf.float32) / 255.0


def _augment_centernet(
    image: tf.Tensor,
    heatmap: tf.Tensor,
    offset: tf.Tensor,
    board_style_prob: float = 0.4,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Light augmentation for CenterNet: brightness, contrast, flip, board-style.

    Args:
        image: Input image (H, W, 3) in [0, 1].
        heatmap: Target heatmap (H_out, W_out, 1).
        offset: Target offset (H_out, W_out, 2).
        board_style_prob: Probability of applying firmware-style luma preprocess.
    """
    # Random brightness adjustment.
    image = tf.image.random_brightness(image, max_delta=0.1)
    # Random contrast adjustment.
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Firmware-style board preprocessing with given probability.
    apply_board = tf.random.uniform([], seed=43) < board_style_prob
    input_h = tf.shape(image)[0]
    input_w = tf.shape(image)[1]

    def _apply_board() -> tf.Tensor:
        return _preprocess_board_style_luma(image, input_h, input_w)

    image = tf.cond(apply_board, _apply_board, lambda: image)

    # Horizontal flip with 50% probability.
    should_flip = tf.random.uniform([], seed=42) > 0.5

    def _flip() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        flipped_img = tf.image.flip_left_right(image)
        flipped_hm = tf.image.flip_left_right(heatmap)
        flipped_off_x = -offset[..., 0:1]
        flipped_off_y = offset[..., 1:2]
        flipped_off = tf.concat([flipped_off_x, flipped_off_y], axis=-1)
        flipped_off = tf.image.flip_left_right(flipped_off)
        return flipped_img, flipped_hm, flipped_off

    def _no_flip() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return image, heatmap, offset

    return tf.cond(should_flip, _flip, _no_flip)
