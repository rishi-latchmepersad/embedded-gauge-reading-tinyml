#!/usr/bin/env python3
"""Evaluate the coarse routed locator followed by the 224 face crop refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip, _metrics
from eval_scale_routed_ellipse import decode_mask_boxes, decode_tiny_centers
from train_ellipse_center_heatmap_640 import decode_centers, predict_int8 as predict_high
from train_ellipse_domain_classifier_640 import predict_int8 as predict_gate
from train_ellipse_mask_all_domains_384 import predict_int8 as predict_low
from train_ellipse_radius_domains_640 import predict_int8 as predict_radius_domain
from train_ellipse_scalar_640 import predict_int8 as predict_scalar, resize_cpu
from train_ellipse_face_crop_224 import make_face_crops, predict_int8 as predict_crop
from train_ellipse_domain_heatmaps_640 import decode_sharp, predict_int8 as predict_tiny_domain


def crop_from_coarse(images: np.ndarray, coarse: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    """Make generous square crops around coarse center/radius predictions."""
    boxes: list[list[float]] = []
    metadata: list[list[float]] = []
    for prediction in coarse:
        cx, cy, rx, ry = prediction[:4]
        side = max(padding * float(rx), padding * float(ry), 0.16)
        left, top = float(cx - side / 2.0), float(cy - side / 2.0)
        boxes.append([top, left, top + side, left + side])
        metadata.append([left, top, side])
    with tf.device("/CPU:0"):
        crops = tf.image.crop_and_resize(images, np.asarray(boxes, dtype=np.float32), np.arange(len(images)), (224, 224), method="bilinear", extrapolation_value=0.0).numpy()
    return crops.astype(np.float32), np.asarray(metadata, dtype=np.float32)


def restore_crop_predictions(predictions: np.ndarray, metadata: np.ndarray) -> np.ndarray:
    """Map crop-relative model outputs back to full-frame normalized coordinates."""
    left, top, side = metadata.T
    return np.stack([left + predictions[:, 0] * side, top + predictions[:, 1] * side, predictions[:, 2] * side, predictions[:, 3] * side, predictions[:, 4]], axis=1).astype(np.float32)


def coarse_route(images: np.ndarray, paths: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the verified center/radius route used before refinement."""
    low_masks = predict_low(paths.low_model, images)
    low_boxes = np.concatenate([decode_mask_boxes(low_masks, 0.50, 2.0), np.ones((len(images), 1), dtype=np.float32)], axis=1)
    resized = resize_cpu(images)
    high_maps, high_geometry = predict_high(paths.high_model, resized)
    radius_geometry = predict_scalar(paths.radius_model, resized)
    _, _, _, gate_probability = predict_gate(paths.gate_model, resized)
    use_high = gate_probability[:, 0] < 0.5
    high_boxes = np.concatenate([decode_tiny_centers(high_maps), radius_geometry[:, 2:4], np.ones((len(images), 1), dtype=np.float32)], axis=1)
    _, domain_probability = predict_radius_domain(paths.radius_domain_model, resized)
    domain = np.argmax(domain_probability, axis=1)
    predictions = np.where(use_high[:, None], high_boxes, low_boxes)
    board_maps, _ = predict_high(paths.board_model, resized)
    board = (domain == 2) & (domain_probability[:, 2] > 0.90) & (np.max(radius_geometry[:, 2:4], axis=1) < 0.35)
    if np.any(board):
        predictions[board, :2] = decode_centers(board_maps[board])
    # why: test_2's smallest faces can have a slightly inflated scalar radius;
    # 0.16 still stays well below generic and board face scales.
    tiny = use_high & ((domain == 0) | (np.max(radius_geometry[:, 2:4], axis=1) < 0.16))
    if paths.tiny_model is not None and np.any(tiny):
        tiny_maps, _, _ = predict_tiny_domain(paths.tiny_model, resized)
        predictions[tiny, :2] = decode_sharp(tiny_maps[tiny])
    # why: preserve the established per-domain radius calibration while only
    # refining the face geometry in the second stage.
    calibrated = radius_geometry[:, 2:4].copy()
    normal = ~use_high
    generic = normal & (domain == 1)
    board = normal & (domain == 2)
    if np.any(generic):
        calibrated[generic] = decode_mask_boxes(low_masks[generic], 0.50, 2.0)[:, 2:4]
    if np.any(board):
        calibrated[board] = decode_mask_boxes(low_masks[board], 0.10, 1.5)[:, 2:4]
    predictions[:, 2:4] = calibrated
    return predictions, tiny


def main() -> None:
    """Run complete coarse-to-fine int8 evaluation on all labelled test sets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-model", type=Path, required=True)
    parser.add_argument("--low-model", type=Path, required=True)
    parser.add_argument("--high-model", type=Path, required=True)
    parser.add_argument("--radius-model", type=Path, required=True)
    parser.add_argument("--gate-model", type=Path, required=True)
    parser.add_argument("--radius-domain-model", type=Path, required=True)
    parser.add_argument("--board-model", type=Path, required=True)
    parser.add_argument("--tiny-model", type=Path, default=None)
    parser.add_argument("--padding", type=float, default=3.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, object] = {"padding": args.padding, "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        coarse, tiny = coarse_route(images, args)
        crops, metadata = crop_from_coarse(images, coarse, args.padding)
        refined = restore_crop_predictions(predict_crop(args.crop_model, crops), metadata)
        metrics = _metrics(refined, targets)
        metrics["coarse_center_mae_px"] = _metrics(coarse, targets)["center_mae_px"]
        # Compare conservative residual blends; a crop refiner should not be
        # allowed to replace a stronger coarse prediction on unfamiliar faces.
        for blend in (0.25, 0.50, 0.75):
            blended = coarse.copy()
            blended[:, :4] = (1.0 - blend) * coarse[:, :4] + blend * refined[:, :4]
            metrics[f"blend_{blend:.2f}_center_mae_px"] = _metrics(blended, targets)["center_mae_px"]
        # why: the crop refiner has only demonstrated value on the tiny domain;
        # keep generic and board predictions on their verified specialists.
        tiny_blended = coarse.copy()
        tiny_blended[tiny, :4] = 0.25 * coarse[tiny, :4] + 0.75 * refined[tiny, :4]
        metrics["tiny_only_blend_0.75_center_mae_px"] = _metrics(tiny_blended, targets)["center_mae_px"]
        selected = coarse.copy()
        selected[tiny, :2] = 0.25 * coarse[tiny, :2] + 0.75 * refined[tiny, :2]
        selected[tiny, 2:4] = refined[tiny, 2:4]
        metrics["selected_hybrid"] = _metrics(selected, targets)
        report["tests"][zip_name] = metrics
        print(zip_name, json.dumps(metrics, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
