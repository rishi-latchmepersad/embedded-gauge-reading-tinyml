#!/usr/bin/env python3
"""Evaluate robust image-space ellipse fitting on top of the learned proposal.

This is a deliberately different experiment: the neural model only finds a
generous face proposal, while Canny contours and geometric ellipse fitting
recover the visible rim without forcing a second learned regression head.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from eval_ellipse_all_test_sets import _load_zip, _metrics
from train_coarse_fine_ellipse_224 import crop_image, stage1_decode

STAGE1_MODEL = Path("artifacts/gauge_ellipse_mask_center_scaleconf_384_aux_v1/model_int8.tflite")


def fit_candidates(crop: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Fit and score ellipse candidates from several edge thresholds."""
    gray = np.asarray(np.clip(crop[..., 0] * 255.0, 0, 255), dtype=np.uint8)
    candidates: list[tuple[np.ndarray, float]] = []
    for low, high in ((20, 50), (40, 100), (70, 150), (100, 220)):
        edges = cv2.Canny(gray, low, high, L2gradient=True)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 20:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 0.08 * gray.shape[0]:
                continue
            try:
                (cx, cy), (major, minor), _ = cv2.fitEllipse(contour)
            except cv2.error:
                continue
            axes = np.sort(np.asarray([major, minor], dtype=np.float32))[::-1]
            if axes[1] < 0.12 * gray.shape[0] or axes[0] > 1.8 * gray.shape[0]:
                continue
            # why: gauge faces are close to circular in the image plane; this
            # rejects text strokes and long panel edges before scoring.
            ratio = float(axes[1] / max(axes[0], 1.0))
            if ratio < 0.35:
                continue
            support = min(float(len(contour)) / max(float(perimeter), 1.0), 1.0)
            candidates.append((np.asarray([cx, cy, axes[0] / 2.0, axes[1] / 2.0], dtype=np.float32), support))
    return candidates


def refine_one(image: np.ndarray, proposal: np.ndarray) -> np.ndarray:
    """Fit a rim ellipse inside one proposal and map it to full-frame units."""
    side = float(np.clip(2.2 * max(proposal[2], proposal[3]), 0.18, 1.4))
    box = np.asarray([proposal[0] - side / 2, proposal[1] - side / 2, proposal[0] + side / 2, proposal[1] + side / 2], dtype=np.float32)
    crop, source = crop_image(image, box)
    candidates = fit_candidates(crop)
    if not candidates:
        return proposal.copy()
    best: tuple[float, np.ndarray] | None = None
    proposal_local = np.asarray([(proposal[0] - source[0]) / (source[2] - source[0]), (proposal[1] - source[1]) / (source[3] - source[1])], dtype=np.float32)
    for ellipse, support in candidates:
        local_center = ellipse[:2] / crop.shape[0]
        local_radius = ellipse[2:] / crop.shape[0]
        center_distance = float(np.linalg.norm(local_center - proposal_local))
        radius_distance = float(np.linalg.norm(local_radius - proposal[2:4] / (source[2:] - source[:2])))
        score = 3.0 * center_distance + 1.5 * radius_distance - 0.5 * support
        if best is None or score < best[0]:
            best = (score, ellipse)
    assert best is not None
    ellipse = best[1]
    sx, sy = source[2] - source[0], source[3] - source[1]
    fitted = np.asarray([source[0] + ellipse[0] / crop.shape[0] * sx, source[1] + ellipse[1] / crop.shape[0] * sy, ellipse[2] / crop.shape[0] * sx, ellipse[3] / crop.shape[0] * sy], dtype=np.float32)
    # why: fitting is only allowed to make a conservative residual update;
    # false contours cannot move the learned center by an entire crop width.
    return 0.35 * proposal + 0.65 * fitted


def main() -> None:
    """Evaluate proposal, fitted ellipse, and their center/radius blends."""
    reports: dict[str, object] = {"tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        proposals = stage1_decode(STAGE1_MODEL, images)
        fitted = np.asarray([refine_one(image, proposal) for image, proposal in zip(images, proposals)], dtype=np.float32)
        rows: dict[str, object] = {"proposal": _metrics(np.concatenate([proposals, np.ones((len(proposals), 1), dtype=np.float32)], axis=1), targets)}
        for alpha in (0.25, 0.5, 0.75, 1.0):
            blended = proposals * (1.0 - alpha) + fitted * alpha
            rows[f"classical_blend_{alpha:.2f}"] = _metrics(np.concatenate([blended, np.ones((len(blended), 1), dtype=np.float32)], axis=1), targets)
        reports["tests"][zip_name] = rows
        print(zip_name, json.dumps(rows), flush=True)
    Path("artifacts/classical_ellipse_refinement_v1_report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
