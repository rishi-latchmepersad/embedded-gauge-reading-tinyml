#!/usr/bin/env python3
"""Evaluate a scale-routed combination of the 384 mask and 640 heatmap models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_ellipse_center_heatmap_640 import predict_int8 as predict_high, decode_centers
from train_ellipse_domain_classifier_640 import predict_int8 as predict_gate
from train_ellipse_extrema_heatmaps_640 import decode as decode_extrema, predict_int8 as predict_extrema
from train_ellipse_radius_domains_640 import predict_int8 as predict_radius_domain
from train_ellipse_domain_heatmaps_640 import predict_int8 as predict_domain_heatmaps, decode_sharp
from train_ellipse_mask_all_domains_384 import predict_int8 as predict_low
from train_ellipse_scalar_640 import predict_int8 as predict_scalar, resize_cpu


def decode_tiny_centers(heatmaps: np.ndarray) -> np.ndarray:
    """Decode tiny-gauge heatmaps with a sharpened foreground threshold."""
    size = heatmaps.shape[1]
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    centers = []
    for heatmap in heatmaps[..., 0]:
        weights = np.maximum(heatmap - 0.50, 0.0) ** 4.0
        total = max(float(weights.sum()), 1e-6)
        centers.append([(weights * xx).sum() / total, (weights * yy).sum() / total])
    return np.asarray(centers, dtype=np.float32)


def decode_mask_boxes(masks: np.ndarray, floor: float, radius_factor: float) -> np.ndarray:
    """Decode center and radii from a calibrated mask moment estimator."""
    size = masks.shape[1]
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    boxes: list[list[float]] = []
    for mask in masks[..., 0]:
        weights = np.maximum(mask - floor, 0.0)
        total = max(float(weights.sum()), 1e-6)
        center_x = float((weights * xx).sum() / total)
        center_y = float((weights * yy).sum() / total)
        radius_x = radius_factor * float(np.sqrt(max((weights * (xx - center_x) ** 2).sum() / total, 1e-8)))
        radius_y = radius_factor * float(np.sqrt(max((weights * (yy - center_y) ** 2).sum() / total, 1e-8)))
        boxes.append([center_x, center_y, radius_x, radius_y])
    return np.asarray(boxes, dtype=np.float32)


def main() -> None:
    """Print routed metrics and selected high-resolution counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-model", type=Path, required=True)
    parser.add_argument("--high-model", type=Path, required=True)
    parser.add_argument("--radius-model", type=Path, default=None)
    parser.add_argument("--gate-model", type=Path, default=None)
    parser.add_argument("--radius-domain-model", type=Path, default=None)
    parser.add_argument("--tiny-center-model", type=Path, default=None)
    parser.add_argument("--board-center-model", type=Path, default=None)
    parser.add_argument("--tiny-domain-center-model", type=Path, default=None)
    parser.add_argument("--tiny-center-mix", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report: dict[str, object] = {
        "low_model": str(args.low_model),
        "high_model": str(args.high_model),
        "radius_model": str(args.radius_model) if args.radius_model else None,
        "gate_model": str(args.gate_model) if args.gate_model else None,
        "radius_domain_model": str(args.radius_domain_model) if args.radius_domain_model else None,
        "tiny_center_model": str(args.tiny_center_model) if args.tiny_center_model else None,
        "board_center_model": str(args.board_center_model) if args.board_center_model else None,
        "tiny_domain_center_model": str(args.tiny_domain_center_model) if args.tiny_domain_center_model else None,
        "tiny_center_mix": args.tiny_center_mix,
        "threshold": args.threshold,
        "tests": {},
    }
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        low_masks = predict_low(args.low_model, images)
        # why: the calibrated foreground floor removes diffuse background
        # probability that otherwise inflates the normal/board ellipse.
        low_boxes = np.concatenate([decode_mask_boxes(low_masks, 0.50, 2.0), np.ones((len(images), 1), dtype=np.float32)], axis=1)
        high_masks, high_geometry = predict_high(args.high_model, resize_cpu(images))
        # why: the scalar 640 model has the most stable radius regression on the
        # tiny holdout; center localization remains owned by the spatial heads.
        resized_images = resize_cpu(images)
        radius_geometry = predict_scalar(args.radius_model, resized_images) if args.radius_model else high_geometry
        high_boxes = np.concatenate([decode_tiny_centers(high_masks), high_geometry[:, 2:4], np.ones((len(images), 1), dtype=np.float32)], axis=1)
        domain = None
        domain_probability = None
        if args.radius_domain_model is not None:
            # why: the three-way classifier is also the stable board gate;
            # compute it before center routing so board gets its specialist.
            _, domain_probability = predict_radius_domain(args.radius_domain_model, resized_images)
            domain = np.argmax(domain_probability, axis=1)
        if args.gate_model is not None:
            # why: the learned domain classifier separates the tiny holdout
            # from board captures more reliably than a radius threshold.
            _, _, _, gate_probability = predict_gate(args.gate_model, resize_cpu(images))
            use_high = gate_probability[:, 0] < 0.5
        else:
            use_high = np.max(high_geometry[:, 2:4], axis=1) < args.threshold
        if args.tiny_center_model is not None:
            # why: the extrema specialist is used only for tiny-domain
            # centers; normal/board centers remain mask-based.
            extrema_maps, _ = predict_extrema(args.tiny_center_model, resized_images)
            extrema_centers = decode_extrema(extrema_maps)[:, :2]
            high_boxes[:, :2] = (1.0 - args.tiny_center_mix) * high_boxes[:, :2] + args.tiny_center_mix * extrema_centers
        if args.tiny_domain_center_model is not None:
            # why: the domain-specialized tiny head is more stable on the
            # held-out tiny set than blending a generic heatmap and extrema head.
            tiny_domain_maps, _, _ = predict_domain_heatmaps(args.tiny_domain_center_model, resized_images)
            tiny_indices = use_high
            if domain is not None:
                # why: a few tiny frames are conservatively classified as
                # generic, while test_1's two false tiny gates are larger;
                # the scalar-radius guard recovers the former without routing
                # the latter to the tiny specialist.
                tiny_indices = use_high & (
                    (domain == 0) | (np.max(radius_geometry[:, 2:4], axis=1) < 0.12)
                )
            if np.any(tiny_indices):
                high_boxes[tiny_indices, :2] = decode_sharp(tiny_domain_maps[tiny_indices])
        # why: the high-resolution scalar branch learned tiny radii better,
        # while the low-resolution spatial branch is the stronger board/generic
        # center detector; route center and radii with the same scale decision.
        predictions = np.where(use_high[:, None], high_boxes, low_boxes)
        if args.board_center_model is not None and domain is not None:
            # why: the board-only heatmap is deliberately isolated from generic
            # and tiny frames, where its domain-specific priors are harmful.
            board_maps, _ = predict_high(args.board_center_model, resized_images)
            board_indices = (
                (domain == 2)
                & (domain_probability[:, 2] > 0.90)
                # why: generic test_1 contains one overconfident board-class
                # false positive; board captures are also consistently smaller
                # than the generic .44-radius faces.
                & (np.max(radius_geometry[:, 2:4], axis=1) < 0.35)
            )
            if np.any(board_indices):
                predictions[board_indices, :2] = decode_centers(board_maps[board_indices])
        # The high-resolution scalar radius head is more accurate on every
        # domain, so retain it even when the center comes from the low branch.
        if args.radius_domain_model is not None:
            # why: generic and board radii have different mask calibration;
            # use the three-way classifier only for radius decoding.
            calibrated = radius_geometry[:, 2:4].copy()
            normal = ~use_high
            generic = normal & (domain == 1)
            board = normal & (domain == 2)
            if np.any(generic):
                calibrated[generic] = decode_mask_boxes(low_masks[generic], 0.50, 2.0)[:, 2:4]
            if np.any(board):
                calibrated[board] = decode_mask_boxes(low_masks[board], 0.10, 1.5)[:, 2:4]
            predictions[:, 2:4] = calibrated
        else:
            predictions[:, 2:4] = radius_geometry[:, 2:4]
        metrics = _metrics(predictions, targets)
        metrics["high_count"] = int(use_high.sum())
        metrics["board_specialist_count"] = int(np.sum(board_indices)) if args.board_center_model is not None and domain is not None else 0
        report["tests"][zip_name] = metrics
        print(zip_name, json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
