"""Training sub-module for embedded gauge reading.
"""

from __future__ import annotations

import math
import os

import numpy as np
import tensorflow as tf


def _augment_glare_blobs(image: tf.Tensor) -> tf.Tensor:
    """Stamp 1–3 bright glare blobs via resized gaussian noise. Simulates specular reflections on gauge glass."""
    # Generate blobs at a fixed small resolution then resize up — avoids dynamic meshgrid entirely.
    BLOB_RES = 32
    mask = tf.zeros([BLOB_RES, BLOB_RES, 1], dtype=tf.float32)
    for _ in range(3):
        active = tf.cast(tf.random.uniform([]) < 0.5, tf.float32)
        cx = tf.random.uniform([], 2, BLOB_RES - 2, dtype=tf.float32)
        cy = tf.random.uniform([], 2, BLOB_RES - 2, dtype=tf.float32)
        # Draw a single bright pixel at (cy, cx) with strength, then blur via resize
        brightness = tf.random.uniform([], 0.5, 1.0) * active
        # Use a small dense gaussian kernel: encode as a 1-pixel impulse scaled by brightness
        # We approximate the blob by creating a [BLOB_RES, BLOB_RES] tensor with one hot pixel
        # and relying on bilinear resize to spread it.
        cy_i = tf.cast(tf.round(cy), tf.int32)
        cx_i = tf.cast(tf.round(cx), tf.int32)
        # Scatter a bright spot
        idx = tf.stack([cy_i, cx_i, 0])
        spot = tf.tensor_scatter_nd_update(
            tf.zeros([BLOB_RES, BLOB_RES, 1], dtype=tf.float32),
            [idx],
            [brightness],
        )
        mask = mask + spot

    # Resize to image size with bicubic to spread the spots into soft blobs
    image_shape = tf.shape(image)
    mask = tf.image.resize(mask, [image_shape[0], image_shape[1]], method="bicubic")
    mask = tf.clip_by_value(mask, 0.0, 1.0)  # [H, W, 1] broadcast across channels
    image = image + mask * (1.0 - image)
    return tf.clip_by_value(image, 0.0, 1.0)



def _augment_image(image: tf.Tensor) -> tf.Tensor:
    """Apply photometric augmentation that preserves gauge geometry.

    This augmentation is designed to match the board camera reality:
    - Crop jitter: Simulates slight variations in dial position within crop
    - Brightness/exposure: Simulates camera exposure variations (under/over)
    - Contrast/saturation: Simulates lighting condition variations
    - Gamma: Simulates non-linear exposure response
    - Glare: Simulates specular reflections on gauge glass
    - Noise: Simulates sensor noise
    """
    image_shape: tf.Tensor = tf.shape(image)
    image_h: tf.Tensor = image_shape[0]
    image_w: tf.Tensor = image_shape[1]

    # --- Crop jitter: simulate slight dial position variations within crop ---
    # Random scale from 90% to 100% of crop (was 92-100%)
    scale: tf.Tensor = tf.random.uniform([], minval=0.90, maxval=1.0, dtype=tf.float32)
    crop_h: tf.Tensor = tf.maximum(
        2, tf.cast(tf.cast(image_h, tf.float32) * scale, dtype=tf.int32)
    )
    crop_w: tf.Tensor = tf.maximum(
        2, tf.cast(tf.cast(image_w, tf.float32) * scale, dtype=tf.int32)
    )
    image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
    image = tf.image.resize(image, [image_h, image_w])

    # Additional crop jitter: slight translation within the crop box
    # This simulates the dial not being perfectly centered in the crop
    max_offset: tf.Tensor = tf.cast(
        tf.cast(image_h, tf.float32) * 0.05, tf.int32
    )  # 5% max offset
    if max_offset > 0:
        offset_y = tf.random.uniform([], -max_offset, max_offset + 1, dtype=tf.int32)
        offset_x = tf.random.uniform([], -max_offset, max_offset + 1, dtype=tf.int32)
        # Pad image to allow cropping with offset
        pad_h = image_h + 2 * max_offset
        pad_w = image_w + 2 * max_offset
        image_padded = tf.image.resize_with_pad(image, pad_h, pad_w)
        image = tf.image.crop_to_bounding_box(
            image_padded, max_offset + offset_y, max_offset + offset_x, image_h, image_w
        )

    # AUG_HEAVY=1 → stronger photometric augmentation for board-domain robustness.
    aug_heavy: bool = os.environ.get("AUG_HEAVY", "0") == "1"

    # --- Brightness/exposure augmentation (matches board camera reality) ---
    # Simulates exposure variations: underexposure (dark) and overexposure (bright)
    if aug_heavy:
        # Wider exposure range for heavy augmentation
        exposure_range = 0.35  # +/- 35% brightness
        contrast_range = (0.55, 1.45)
        saturation_range = (0.70, 1.30)
        gamma_range_dark = (1.0, 2.8)
        gamma_range_bright = (0.4, 1.0)
    else:
        # Moderate exposure range for standard augmentation
        exposure_range = 0.20  # +/- 20% brightness
        contrast_range = (0.75, 1.25)
        saturation_range = (0.85, 1.15)
        gamma_range_dark = (1.0, 2.2)
        gamma_range_bright = (0.6, 1.0)

    # Random brightness (linear exposure)
    image = tf.image.random_brightness(image, max_delta=exposure_range)

    # Random contrast (simulates lighting consistency)
    image = tf.image.random_contrast(
        image, lower=contrast_range[0], upper=contrast_range[1]
    )

    # Random saturation (simulates color consistency under different lighting)
    image = tf.image.random_saturation(
        image, lower=saturation_range[0], upper=saturation_range[1]
    )

    # Clip to valid range before gamma operations
    image = tf.clip_by_value(image, 0.0, 1.0)

    # --- Gamma augmentation: simulates non-linear camera exposure response ---
    if aug_heavy:
        # Wider gamma sweep at higher rate: covers under- AND over-exposed captures.
        gamma_dark: tf.Tensor = tf.random.uniform(
            [], minval=gamma_range_dark[0], maxval=gamma_range_dark[1], dtype=tf.float32
        )
        apply_dark: tf.Tensor = tf.random.uniform([]) < 0.35
        image = tf.where(apply_dark, tf.pow(image, gamma_dark), image)
        gamma_bright: tf.Tensor = tf.random.uniform(
            [],
            minval=gamma_range_bright[0],
            maxval=gamma_range_bright[1],
            dtype=tf.float32,
        )
        apply_bright: tf.Tensor = tf.random.uniform([]) < 0.20
        image = tf.where(apply_bright, tf.pow(image, gamma_bright), image)
    else:
        # Moderate gamma adjustment
        gamma: tf.Tensor = tf.random.uniform(
            [], minval=1.0, maxval=2.0, dtype=tf.float32
        )
        apply_dark: tf.Tensor = tf.random.uniform([]) < 0.15
        image = tf.where(apply_dark, tf.pow(image, gamma), image)

    # Simulate specular glare patches on gauge glass (25% chance, was 20%)
    apply_glare: tf.Tensor = tf.random.uniform([]) < 0.25
    image = tf.cond(
        apply_glare,
        lambda: _augment_glare_blobs(image),
        lambda: image,
    )

    # Add sensor noise (slightly higher for board-like conditions)
    noise_std: float = 0.02 if aug_heavy else 0.015
    noise: tf.Tensor = tf.random.normal(
        shape=tf.shape(image),
        mean=0.0,
        stddev=noise_std,
        dtype=tf.float32,
    )
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image



def _augment_full_frame_box_image(image: tf.Tensor) -> tf.Tensor:
    """Apply photometric-only augmentation for source-space crop-box training.

    We avoid crop jitter here because the target is expressed in the original
    source-frame coordinate system and should not be warped without updating
    the labels.
    """
    aug_heavy: bool = os.environ.get("AUG_HEAVY", "0") == "1"
    if aug_heavy:
        exposure_range = 0.30
        contrast_range = (0.60, 1.40)
        saturation_range = (0.75, 1.25)
        noise_std = 0.02
    else:
        exposure_range = 0.15
        contrast_range = (0.82, 1.18)
        saturation_range = (0.90, 1.10)
        noise_std = 0.012

    image = tf.image.random_brightness(image, max_delta=exposure_range)
    image = tf.image.random_contrast(
        image, lower=contrast_range[0], upper=contrast_range[1]
    )
    image = tf.image.random_saturation(
        image, lower=saturation_range[0], upper=saturation_range[1]
    )
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.cond(
        tf.random.uniform([]) < 0.25,
        lambda: _augment_glare_blobs(image),
        lambda: image,
    )
    noise: tf.Tensor = tf.random.normal(
        shape=tf.shape(image),
        mean=0.0,
        stddev=noise_std,
        dtype=tf.float32,
    )
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image



def _augment_rectifier_image_and_box(
    image: tf.Tensor,
    box: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Geometric + photometric augmentation for the rectifier training path.

    Randomly zooms into a sub-region of the 224x224 padded canvas so the model
    sees dial sizes ranging from ~30% to 100% of the frame — covering both the
    far-away phone-photo distribution and close-up board-camera framing.

    The target box (cx, cy, w, h) in [0,1] canvas coords is recomputed to stay
    consistent with the zoomed view.  If the zoom window clips the dial box we
    clamp it so the target always describes the visible portion of the dial.
    """
    image_shape: tf.Tensor = tf.shape(image)
    canvas: tf.Tensor = tf.cast(image_shape[0], tf.float32)  # square 224

    # --- random zoom window in canvas pixels ---
    # zoom_scale in [0.40, 1.0]: 0.40 → 2.5x zoom-in (close-up), 1.0 → full frame
    zoom_scale: tf.Tensor = tf.random.uniform([], minval=0.40, maxval=1.0)
    win_size: tf.Tensor = canvas * zoom_scale

    max_offset: tf.Tensor = tf.maximum(canvas - win_size, 0.0)
    off_x: tf.Tensor = tf.random.uniform([], minval=0.0, maxval=1.0) * max_offset
    off_y: tf.Tensor = tf.random.uniform([], minval=0.0, maxval=1.0) * max_offset

    # crop then resize back to canvas
    off_y_i = tf.cast(tf.math.floor(off_y), tf.int32)
    off_x_i = tf.cast(tf.math.floor(off_x), tf.int32)
    win_i = tf.cast(tf.math.ceil(win_size), tf.int32)
    win_i = tf.maximum(win_i, 2)
    win_i = tf.minimum(win_i, image_shape[0] - off_y_i)
    win_i_w = tf.minimum(win_i, image_shape[1] - off_x_i)
    image = tf.image.crop_to_bounding_box(image, off_y_i, off_x_i, win_i, win_i_w)
    image = tf.image.resize(image, [image_shape[0], image_shape[1]])

    # --- recompute box target in zoomed canvas coords ---
    # input box: [cx, cy, w, h] all in [0,1] relative to original 224x224
    cx = box[0] * canvas
    cy = box[1] * canvas
    bw = box[2] * canvas
    bh = box[3] * canvas

    x_min = cx - bw * 0.5
    y_min = cy - bh * 0.5
    x_max = cx + bw * 0.5
    y_max = cy + bh * 0.5

    # map original canvas coords → zoomed canvas coords
    x_min_z = (x_min - off_x) / win_size * canvas
    y_min_z = (y_min - off_y) / win_size * canvas
    x_max_z = (x_max - off_x) / win_size * canvas
    y_max_z = (y_max - off_y) / win_size * canvas

    # clamp to [0, canvas]
    x_min_z = tf.clip_by_value(x_min_z, 0.0, canvas)
    y_min_z = tf.clip_by_value(y_min_z, 0.0, canvas)
    x_max_z = tf.clip_by_value(x_max_z, 0.0, canvas)
    y_max_z = tf.clip_by_value(y_max_z, 0.0, canvas)

    new_cx = tf.clip_by_value(((x_min_z + x_max_z) * 0.5) / canvas, 0.0, 1.0)
    new_cy = tf.clip_by_value(((y_min_z + y_max_z) * 0.5) / canvas, 0.0, 1.0)
    new_w = tf.clip_by_value((x_max_z - x_min_z) / canvas, 0.0, 1.0)
    new_h = tf.clip_by_value((y_max_z - y_min_z) / canvas, 0.0, 1.0)
    new_box = tf.stack([new_cx, new_cy, new_w, new_h])

    # photometric augmentation (same as scalar path)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    noise: tf.Tensor = tf.random.normal(
        shape=tf.shape(image), mean=0.0, stddev=0.01, dtype=tf.float32
    )
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, new_box

