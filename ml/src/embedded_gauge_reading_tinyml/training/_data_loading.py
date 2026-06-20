"""Training sub-module for embedded gauge reading.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf


import tensorflow as tf
import numpy as np


def _crop_image_with_xyxy(image: tf.Tensor, crop_box_xyxy: tf.Tensor) -> tf.Tensor:
    """Safely crop image using float xyxy box and clip to valid image bounds."""
    shape: tf.Tensor = tf.shape(image)
    img_h: tf.Tensor = shape[0]
    img_w: tf.Tensor = shape[1]

    x_min_f: tf.Tensor = tf.clip_by_value(
        crop_box_xyxy[0], 0.0, tf.cast(img_w - 1, tf.float32)
    )
    y_min_f: tf.Tensor = tf.clip_by_value(
        crop_box_xyxy[1], 0.0, tf.cast(img_h - 1, tf.float32)
    )
    x_max_f: tf.Tensor = tf.clip_by_value(
        crop_box_xyxy[2], x_min_f + 1.0, tf.cast(img_w, tf.float32)
    )
    y_max_f: tf.Tensor = tf.clip_by_value(
        crop_box_xyxy[3], y_min_f + 1.0, tf.cast(img_h, tf.float32)
    )

    x_min: tf.Tensor = tf.cast(tf.math.floor(x_min_f), tf.int32)
    y_min: tf.Tensor = tf.cast(tf.math.floor(y_min_f), tf.int32)
    x_max: tf.Tensor = tf.cast(tf.math.ceil(x_max_f), tf.int32)
    y_max: tf.Tensor = tf.cast(tf.math.ceil(y_max_f), tf.int32)

    crop_w: tf.Tensor = tf.maximum(1, x_max - x_min)
    crop_h: tf.Tensor = tf.maximum(1, y_max - y_min)

    crop_w = tf.minimum(crop_w, img_w - x_min)
    crop_h = tf.minimum(crop_h, img_h - y_min)

    return tf.image.crop_to_bounding_box(image, y_min, x_min, crop_h, crop_w)



def _load_crop_and_preprocess_image(
    image_path: tf.Tensor,
    value: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Read image, crop dial ROI, resize, and normalize to [0, 1]."""
    image_bytes: tf.Tensor = tf.io.read_file(image_path)
    image: tf.Tensor = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = tf.ensure_shape(image, [None, None, 3])

    image = _crop_image_with_xyxy(image, crop_box_xyxy)
    # Preserve dial geometry (needle angle) by avoiding anisotropic warping.
    image = tf.image.resize_with_pad(image, image_height, image_width)

    image = tf.cast(image, tf.float32) / 255.0
    target: tf.Tensor = tf.cast(value, tf.float32)
    return image, target



def _preprocess_board_style(
    cropped_image: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tf.Tensor:
    """Simulate the firmware preprocess: luma extraction, nearest-neighbor resize, zero pad.

    The board firmware extracts Y-channel luma from YUV422, resizes with
    nearest-neighbour to preserve the dial geometry, and zero-pads to 224x224.
    Replicating that path during training removes the bilinear-RGB domain shift.
    """
    # Convert RGB uint8 crop to luma using ITU-R BT.601 coefficients.
    image = tf.cast(cropped_image, tf.float32)
    luma = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    luma = tf.clip_by_value(luma, 0.0, 255.0)
    luma = tf.cast(tf.round(luma), tf.uint8)

    # Compute scale exactly like firmware: scale = min(224/crop_h, 224/crop_w)
    crop_h = tf.shape(luma)[0]
    crop_w = tf.shape(luma)[1]
    scale = tf.minimum(
        tf.cast(image_height, tf.float32) / tf.cast(crop_h, tf.float32),
        tf.cast(image_width, tf.float32) / tf.cast(crop_w, tf.float32),
    )
    scaled_h = tf.cast(tf.cast(crop_h, tf.float32) * scale, tf.int32)
    scaled_w = tf.cast(tf.cast(crop_w, tf.float32) * scale, tf.int32)
    scaled_h = tf.maximum(scaled_h, 1)
    scaled_w = tf.maximum(scaled_w, 1)

    # Nearest-neighbor resize (firmware does integer nearest-neighbour sampling).
    luma = tf.expand_dims(luma, axis=-1)  # [H, W, 1]
    resized = tf.image.resize(luma, [scaled_h, scaled_w], method='nearest')
    resized = tf.cast(tf.round(resized), tf.uint8)

    # Zero-pad to target size using integer truncation division (same as firmware).
    pad_y = (image_height - scaled_h) // 2
    pad_x = (image_width - scaled_w) // 2
    pad_bottom = image_height - scaled_h - pad_y
    pad_right = image_width - scaled_w - pad_x
    padded = tf.pad(resized, [[pad_y, pad_bottom], [pad_x, pad_right], [0, 0]])
    padded = tf.ensure_shape(padded, [image_height, image_width, 1])

    # Replicate luma to 3 channels because the model expects 3-channel input.
    rgb = tf.tile(padded, [1, 1, 3])

    # Normalize to [0, 1] float.
    return tf.cast(rgb, tf.float32) / 255.0


def _load_crop_and_preprocess_image_board_style(
    image_path: tf.Tensor,
    value: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Read image, crop dial ROI, and apply board-style luma+nearest-neighbor preprocess."""
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, [None, None, 3])
    image = _crop_image_with_xyxy(image, crop_box_xyxy)
    image = _preprocess_board_style(image, image_height, image_width)
    target = tf.cast(value, tf.float32)
    return image, target



def _load_rectifier_and_preprocess_image(
    image_path: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Read a full image and emit a padded-canvas crop-box center/size target."""
    image_bytes: tf.Tensor = tf.io.read_file(image_path)
    image: tf.Tensor = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = tf.ensure_shape(image, [None, None, 3])

    orig_h: tf.Tensor = tf.cast(tf.shape(image)[0], tf.float32)
    orig_w: tf.Tensor = tf.cast(tf.shape(image)[1], tf.float32)

    # Keep the full frame intact so the rectifier learns the crop on the same
    # padded 224x224 canvas used at inference time.
    image = tf.image.resize_with_pad(image, image_height, image_width)
    image = tf.cast(image, tf.float32) / 255.0

    x_min: tf.Tensor = tf.cast(crop_box_xyxy[0], tf.float32)
    y_min: tf.Tensor = tf.cast(crop_box_xyxy[1], tf.float32)
    x_max: tf.Tensor = tf.cast(crop_box_xyxy[2], tf.float32)
    y_max: tf.Tensor = tf.cast(crop_box_xyxy[3], tf.float32)

    scale: tf.Tensor = tf.minimum(
        tf.cast(image_height, tf.float32) / tf.maximum(orig_h, 1.0),
        tf.cast(image_width, tf.float32) / tf.maximum(orig_w, 1.0),
    )
    scaled_w: tf.Tensor = orig_w * scale
    scaled_h: tf.Tensor = orig_h * scale
    pad_x: tf.Tensor = 0.5 * (tf.cast(image_width, tf.float32) - scaled_w)
    pad_y: tf.Tensor = 0.5 * (tf.cast(image_height, tf.float32) - scaled_h)

    x_min_canvas: tf.Tensor = x_min * scale + pad_x
    y_min_canvas: tf.Tensor = y_min * scale + pad_y
    x_max_canvas: tf.Tensor = x_max * scale + pad_x
    y_max_canvas: tf.Tensor = y_max * scale + pad_y

    center_x: tf.Tensor = tf.clip_by_value(
        ((x_min_canvas + x_max_canvas) * 0.5)
        / tf.maximum(tf.cast(image_width, tf.float32), 1.0),
        0.0,
        1.0,
    )
    center_y: tf.Tensor = tf.clip_by_value(
        ((y_min_canvas + y_max_canvas) * 0.5)
        / tf.maximum(tf.cast(image_height, tf.float32), 1.0),
        0.0,
        1.0,
    )
    box_w: tf.Tensor = tf.clip_by_value(
        (x_max_canvas - x_min_canvas)
        / tf.maximum(tf.cast(image_width, tf.float32), 1.0),
        0.0,
        1.0,
    )
    box_h: tf.Tensor = tf.clip_by_value(
        (y_max_canvas - y_min_canvas)
        / tf.maximum(tf.cast(image_height, tf.float32), 1.0),
        0.0,
        1.0,
    )
    target: tf.Tensor = tf.stack([center_x, center_y, box_w, box_h], axis=-1)
    return image, target



def _load_source_crop_and_preprocess_image(
    image_path: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Read a full image and emit a normalized source-space xyxy crop target.

    Unlike the older rectifier target, this keeps the crop in the original
    image coordinate system so the network learns the actual board-space box.
    """
    image_bytes: tf.Tensor = tf.io.read_file(image_path)
    image: tf.Tensor = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = tf.ensure_shape(image, [None, None, 3])

    orig_h: tf.Tensor = tf.cast(tf.shape(image)[0], tf.float32)
    orig_w: tf.Tensor = tf.cast(tf.shape(image)[1], tf.float32)
    image = tf.image.resize_with_pad(image, image_height, image_width)
    image = tf.cast(image, tf.float32) / 255.0

    x_min: tf.Tensor = tf.clip_by_value(
        tf.cast(crop_box_xyxy[0], tf.float32) / tf.maximum(orig_w, 1.0),
        0.0,
        1.0,
    )
    y_min: tf.Tensor = tf.clip_by_value(
        tf.cast(crop_box_xyxy[1], tf.float32) / tf.maximum(orig_h, 1.0),
        0.0,
        1.0,
    )
    x_max: tf.Tensor = tf.clip_by_value(
        tf.cast(crop_box_xyxy[2], tf.float32) / tf.maximum(orig_w, 1.0),
        0.0,
        1.0,
    )
    y_max: tf.Tensor = tf.clip_by_value(
        tf.cast(crop_box_xyxy[3], tf.float32) / tf.maximum(orig_h, 1.0),
        0.0,
        1.0,
    )
    target: tf.Tensor = tf.stack([x_min, y_min, x_max, y_max], axis=-1)
    return image, target



def _load_crop_with_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Load/crop one image and attach the requested sample weight."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    return image, target, weight



def _load_crop_with_weight_maybe_board_style(
    image_path: tf.Tensor,
    value: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    board_style_prob: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Load/crop one image, randomly choosing board-style or standard preprocess.

    With probability oard_style_prob we apply the exact firmware luma +
    nearest-neighbour resize + zero-pad path so the model sees the same
    domain it encounters on-device.
    """
    use_board = tf.random.uniform([]) < board_style_prob
    image, target = tf.cond(
        use_board,
        lambda: _load_crop_and_preprocess_image_board_style(
            image_path, value, crop_box_xyxy, image_height, image_width
        ),
        lambda: _load_crop_and_preprocess_image(
            image_path, value, crop_box_xyxy, image_height, image_width
        ),
    )
    return image, target, weight



def _load_rectifier_with_weight(
    image_path: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Load a full-frame rectifier example and attach its sample weight."""
    image, target = _load_rectifier_and_preprocess_image(
        image_path,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    return image, target, weight



def _load_source_crop_corner_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    corner_heatmaps: tf.Tensor,
    corner_coords: tf.Tensor,
    corner_box: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load a full-frame example and attach corner heatmap and box targets."""
    image, _source_target = _load_source_crop_and_preprocess_image(
        image_path,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    _ = value
    targets = {
        "source_crop_canvas_box": tf.cast(corner_box, tf.float32),
        "keypoint_heatmaps": tf.cast(corner_heatmaps, tf.float32),
        "keypoint_coords": tf.cast(corner_coords, tf.float32),
    }
    return image, targets



def _load_source_crop_corner_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    corner_heatmaps: tf.Tensor,
    corner_coords: tf.Tensor,
    corner_box: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    heatmap_weight: tf.Tensor,
    coord_weight: tf.Tensor,
    box_weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load a full-frame corner-localizer example and attach sample weights."""
    image, targets = _load_source_crop_corner_target(
        image_path,
        value,
        corner_heatmaps,
        corner_coords,
        corner_box,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "source_crop_canvas_box": box_weight,
        "keypoint_heatmaps": heatmap_weight,
        "keypoint_coords": coord_weight,
    }
    return image, targets, sample_weights



def _load_source_crop_with_weight(
    image_path: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Load a full-frame source-crop example and attach the requested weight."""
    image, target = _load_source_crop_and_preprocess_image(
        image_path,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    return image, target, weight



def _load_crop_with_obb_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach oriented-box parameters."""
    image, _scalar_target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    obb_target = tf.cast(obb_params, tf.float32)
    return image, {"obb_params": obb_target}



def _load_crop_with_obb_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach OBB parameters plus a sample weight."""
    image, target = _load_crop_with_obb_target(
        image_path,
        value,
        obb_params,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    return image, target, {"obb_params": tf.cast(weight, tf.float32)}



def _load_fullframe_obb_data(
    image_path: tf.Tensor,
    value: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Read full image, apply board-style preprocess, pass OBB targets in canvas space.

    Purpose-built for full-frame OBB training. Unlike _load_crop_with_obb_weight
    which crops to the dial region, this reads the entire image and scales it to
    fit the 224x224 canvas, preserving the gauge's actual position in the frame.
    """
    image_bytes: tf.Tensor = tf.io.read_file(image_path)
    image: tf.Tensor = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = tf.ensure_shape(image, [None, None, 3])
    image = _preprocess_board_style(image, image_height, image_width)
    obb_target = tf.cast(obb_params, tf.float32)
    return image, {"obb_params": obb_target}, {"obb_params": tf.cast(weight, tf.float32)}



def _load_crop_with_interval_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    interval_index: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach scalar, interval, and weight targets."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    interval_target = tf.cast(interval_index, tf.int32)
    targets = {
        "gauge_value": target,
        "interval_logits": interval_target,
    }
    sample_weights = {
        "gauge_value": weight,
        "interval_logits": weight,
    }
    return image, targets, sample_weights



def _load_crop_with_interval_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    interval_index: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach scalar plus interval-class targets."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    interval_target = tf.cast(interval_index, tf.int32)
    targets = {
        "gauge_value": target,
        "interval_logits": interval_target,
    }
    return image, targets



def _load_crop_with_ordinal_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    ordinal_thresholds: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach scalar plus ordinal-threshold targets."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    ordinal_target = tf.cast(ordinal_thresholds, tf.float32)
    targets = {
        "gauge_value": target,
        "ordinal_logits": ordinal_target,
    }
    return image, targets



def _load_crop_with_ordinal_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    ordinal_thresholds: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach ordinal targets plus sample weights."""
    image, targets = _load_crop_with_ordinal_target(
        image_path,
        value,
        ordinal_thresholds,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "ordinal_logits": weight,
    }
    return image, targets, sample_weights



def _load_crop_with_fraction_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    fraction: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach scalar plus sweep-fraction targets."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    fraction_target = tf.cast(fraction, tf.float32)
    targets = {
        "gauge_value": target,
        "sweep_fraction": fraction_target,
    }
    return image, targets



def _load_crop_with_fraction_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    fraction: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach sweep-fraction targets plus sample weights."""
    image, targets = _load_crop_with_fraction_target(
        image_path,
        value,
        fraction,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "sweep_fraction": weight,
    }
    return image, targets, sample_weights



def _load_crop_with_keypoint_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach scalar plus keypoint-heatmap targets."""
    image, target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    keypoint_target = tf.cast(heatmaps, tf.float32)
    targets = {
        "gauge_value": target,
        "keypoint_heatmaps": keypoint_target,
    }
    return image, targets



def _load_crop_with_keypoint_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    heatmap_weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach keypoint targets plus sample weights."""
    image, targets = _load_crop_with_keypoint_target(
        image_path,
        value,
        heatmaps,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "keypoint_heatmaps": heatmap_weight,
    }
    return image, targets, sample_weights



def _load_crop_with_geometry_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach geometry targets for detector training."""
    image, value_target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    keypoint_target = tf.cast(heatmaps, tf.float32)
    coord_target = tf.cast(coords, tf.float32)
    targets = {
        "gauge_value": value_target,
        "keypoint_heatmaps": keypoint_target,
        "keypoint_coords": coord_target,
    }
    return image, targets



def _load_crop_with_geometry_uncertainty_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach geometry plus uncertainty targets."""
    image, targets = _load_crop_with_geometry_target(
        image_path,
        value,
        heatmaps,
        coords,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    targets["gauge_value_lower"] = targets["gauge_value"]
    targets["gauge_value_upper"] = targets["gauge_value"]
    return image, targets



def _load_crop_with_geometry_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    heatmap_weight: tf.Tensor,
    coord_weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach geometry targets plus sample weights."""
    image, targets = _load_crop_with_geometry_target(
        image_path,
        value,
        heatmaps,
        coords,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "keypoint_heatmaps": heatmap_weight,
        "keypoint_coords": coord_weight,
    }
    return image, targets, sample_weights



def _load_crop_with_direction_geometry_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    needle_unit_xy: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach needle-direction and value targets."""
    image, value_target = _load_crop_and_preprocess_image(
        image_path, value, crop_box_xyxy, image_height, image_width
    )
    direction_target = tf.cast(needle_unit_xy, tf.float32)
    targets = {
        "gauge_value": value_target,
        "needle_xy": direction_target,
    }
    return image, targets



def _load_crop_with_direction_geometry_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    needle_unit_xy: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    direction_weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach needle-direction targets plus weights."""
    image, targets = _load_crop_with_direction_geometry_target(
        image_path,
        value,
        needle_unit_xy,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "needle_xy": direction_weight,
    }
    return image, targets, sample_weights



def _load_crop_with_obb_geometry_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach joint OBB + geometry targets."""
    image, targets = _load_crop_with_geometry_target(
        image_path,
        value,
        heatmaps,
        coords,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    targets["obb_params"] = tf.cast(obb_params, tf.float32)
    return image, targets



def _load_crop_with_obb_geometry_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    heatmap_weight: tf.Tensor,
    coord_weight: tf.Tensor,
    obb_weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach joint OBB + geometry targets plus weights."""
    image, targets = _load_crop_with_obb_geometry_target(
        image_path,
        value,
        heatmaps,
        coords,
        obb_params,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "keypoint_heatmaps": heatmap_weight,
        "keypoint_coords": coord_weight,
        "obb_params": obb_weight,
    }
    return image, targets, sample_weights



def _load_crop_with_obb_mask_geometry_target(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    pointer_mask: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Load/crop one image and attach joint OBB + mask geometry targets."""
    image, targets = _load_crop_with_obb_geometry_target(
        image_path,
        value,
        heatmaps,
        coords,
        obb_params,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    targets["pointer_mask"] = tf.cast(pointer_mask, tf.float32)
    return image, targets



def _load_crop_with_obb_mask_geometry_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    pointer_mask: tf.Tensor,
    obb_params: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    heatmap_weight: tf.Tensor,
    coord_weight: tf.Tensor,
    mask_weight: tf.Tensor,
    obb_weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach OBB + mask targets plus sample weights."""
    image, targets = _load_crop_with_obb_mask_geometry_target(
        image_path,
        value,
        heatmaps,
        coords,
        pointer_mask,
        obb_params,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "keypoint_heatmaps": heatmap_weight,
        "keypoint_coords": coord_weight,
        "pointer_mask": mask_weight,
        "obb_params": obb_weight,
    }
    return image, targets, sample_weights



def _load_crop_with_geometry_uncertainty_weight(
    image_path: tf.Tensor,
    value: tf.Tensor,
    heatmaps: tf.Tensor,
    coords: tf.Tensor,
    crop_box_xyxy: tf.Tensor,
    image_height: int,
    image_width: int,
    weight: tf.Tensor,
    heatmap_weight: tf.Tensor,
    coord_weight: tf.Tensor,
    uncertainty_weight: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    """Load/crop one image and attach geometry uncertainty targets plus weights."""
    image, targets = _load_crop_with_geometry_uncertainty_target(
        image_path,
        value,
        heatmaps,
        coords,
        crop_box_xyxy,
        image_height,
        image_width,
    )
    sample_weights = {
        "gauge_value": weight,
        "keypoint_heatmaps": heatmap_weight,
        "keypoint_coords": coord_weight,
        "gauge_value_lower": uncertainty_weight,
        "gauge_value_upper": uncertainty_weight,
    }
    return image, targets, sample_weights


