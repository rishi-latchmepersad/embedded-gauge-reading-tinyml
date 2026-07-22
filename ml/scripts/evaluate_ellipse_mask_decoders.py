"""Compare integer ellipse-mask decoders on the held-out LittleGood split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_gauge_ellipse_mask_v1 import boxes_from_masks, load_split, predict_int8


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gauge_face_ellipse_v1_640_gray"
MODEL = ROOT / "artifacts" / "gauge_ellipse_mask_littlegood_v1" / "gauge_ellipse_mask_v1_int8.tflite"


def moment_boxes(masks: np.ndarray) -> np.ndarray:
    """Recover ellipse center and radii from weighted mask moments."""
    yy, xx = np.mgrid[0:80, 0:80].astype(np.float32)
    boxes = []
    for mask in masks[..., 0]:
        # why: subtracting the low-confidence floor suppresses halo pixels
        # introduced by int8 sigmoid quantization around the face boundary.
        weight = np.maximum(mask - 0.20, 0.0)
        total = float(weight.sum())
        if total <= 1e-6:
            boxes.append(boxes_from_masks(mask[None, ..., None])[0])
            continue
        cx = float((weight * xx).sum() / total + 0.5) / 80.0
        cy = float((weight * yy).sum() / total + 0.5) / 80.0
        var_x = float((weight * (xx - cx * 80.0 + 0.5) ** 2).sum() / total)
        var_y = float((weight * (yy - cy * 80.0 + 0.5) ** 2).sum() / total)
        rx = np.clip(2.0 * np.sqrt(max(var_x, 1e-6)) / 80.0, 0.03, 0.50)
        ry = np.clip(2.0 * np.sqrt(max(var_y, 1e-6)) / 80.0, 0.03, 0.50)
        boxes.append([cx, cy, rx, ry])
    return np.asarray(boxes, dtype=np.float32)


def score(predicted: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    """Score center tolerance, radius error, and box IoU."""
    true_boxes = np.concatenate((targets[:, :2] - targets[:, 2:], targets[:, :2] + targets[:, 2:]), axis=1)
    pred_boxes = np.concatenate((predicted[:, :2] - predicted[:, 2:], predicted[:, :2] + predicted[:, 2:]), axis=1)
    intersection = np.maximum(0.0, np.minimum(true_boxes[:, 2:], pred_boxes[:, 2:]) - np.maximum(true_boxes[:, :2], pred_boxes[:, :2]))
    inter_area = intersection[:, 0] * intersection[:, 1]
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    iou = inter_area / np.maximum(true_area + pred_area - inter_area, 1e-6)
    center_error = np.linalg.norm(predicted[:, :2] - targets[:, :2], axis=1) * 640.0
    return {"box_iou_mean": float(iou.mean()), "box_iou_ge_0_5": float(np.mean(iou >= 0.5)), "center_within_16px": float(np.mean(center_error <= 16.0)), "center_error_px_mean": float(center_error.mean()), "radius_error_px_mean": (np.abs(predicted[:, 2:] - targets[:, 2:]).mean(axis=0) * 640.0).round(4).tolist()}


def main() -> None:
    """Run both mask decoders and write their held-out comparison."""
    paths, targets = load_split(DATA, "test")
    images = np.stack([tf.image.resize(tf.io.decode_png(tf.io.read_file(path), channels=1), [320, 320]).numpy() for path in paths]).astype(np.float32) / 255.0
    masks = predict_int8(MODEL, images)
    report = {"threshold_box": score(boxes_from_masks(masks), targets), "weighted_moments": score(moment_boxes(masks), targets)}
    output = ROOT / "artifacts" / "gauge_ellipse_mask_littlegood_v1" / "decoder_report.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
