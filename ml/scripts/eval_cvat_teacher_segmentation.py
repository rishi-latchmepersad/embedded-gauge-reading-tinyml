#!/usr/bin/env python3
"""Benchmark the CVAT detector-plus-segmenter teacher on held-out ellipses."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from eval_ellipse_all_test_sets import LABELLED, _extract_ellipse, _metrics


def load_color_zip(zip_name: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Load RGB images and normalized ellipse targets from one CVAT zip."""
    images: list[np.ndarray] = []
    targets: list[tuple[float, float, float, float, float]] = []
    with zipfile.ZipFile(LABELLED / zip_name) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        root = ET.fromstring(archive.read("annotations.xml"))
        for node in root.findall("image"):
            target = _extract_ellipse(node)
            member = members.get(Path(node.get("name", "")).name)
            if target is None or member is None:
                continue
            image = Image.open(io.BytesIO(archive.read(member))).convert("RGB")
            # why: the ellipse targets and all TinyML evaluators use a 640x640 image contract.
            image = image.resize((640, 640), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.uint8))
            targets.append((*target, 1.0))
    return images, np.asarray(targets, dtype=np.float32)


def _ellipse_from_mask(mask: np.ndarray, offset: tuple[float, float], scale: float) -> np.ndarray | None:
    """Fit an ellipse to the largest sufficiently detailed predicted contour."""
    binary = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) >= 20]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20.0:
        return None
    (cx, cy), (major, minor), _ = cv2.fitEllipse(contour)
    xoff, yoff = offset
    return np.asarray([(xoff + cx * scale) / 640.0, (yoff + cy * scale) / 640.0,
                       major * scale / 1280.0, minor * scale / 1280.0, 1.0], dtype=np.float32)


def predict_teacher(detector: YOLO, segmenter: YOLO, images: list[np.ndarray], imgsz: int) -> np.ndarray:
    """Run detector then segmentation and convert the mask ellipse to 640 coordinates."""
    predictions: list[np.ndarray] = []
    for image in images:
        det_results = detector.predict(source=image, imgsz=imgsz, device=0, verbose=False, conf=0.05)
        det = det_results[0]
        if det.boxes is None or len(det.boxes) == 0:
            predictions.append(np.zeros(5, dtype=np.float32))
            continue
        # why: gauge/face/dial classes are safer than choosing a visually unrelated detection.
        class_ids = det.boxes.cls.detach().cpu().numpy().astype(int)
        scores = det.boxes.conf.detach().cpu().numpy()
        allowed = np.where(np.isin(class_ids, [0, 1, 2]))[0]
        index = int(allowed[np.argmax(scores[allowed])]) if len(allowed) else int(np.argmax(scores))
        box = det.boxes.xyxy[index].detach().cpu().numpy()
        x0, y0, x1, y1 = np.maximum(box, 0.0)
        x1 = min(x1, image.shape[1]); y1 = min(y1, image.shape[0])
        crop = image[int(y0):max(int(y1), int(y0) + 1), int(x0):max(int(x1), int(x0) + 1)]
        if crop.size == 0:
            predictions.append(np.zeros(5, dtype=np.float32))
            continue
        seg_results = segmenter.predict(source=crop, imgsz=imgsz, device=0, verbose=False, conf=0.05)
        seg = seg_results[0]
        fitted: np.ndarray | None = None
        if seg.masks is not None and len(seg.masks.data):
            masks = seg.masks.data.detach().cpu().numpy()
            for mask in masks[np.argsort(-seg.boxes.conf.detach().cpu().numpy())]:
                fitted = _ellipse_from_mask(mask, (x0, y0), crop.shape[1] / mask.shape[1])
                if fitted is not None:
                    break
        if fitted is None:
            fitted = np.asarray([(x0 + x1) / 1280.0, (y0 + y1) / 1280.0,
                                 (x1 - x0) / 1280.0, (y1 - y0) / 1280.0, scores[index]], dtype=np.float32)
        predictions.append(fitted)
    return np.asarray(predictions, dtype=np.float32)


def main() -> None:
    """Evaluate and persist teacher geometry metrics for each held-out zip."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--segmenter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    detector, segmenter = YOLO(str(args.detector)), YOLO(str(args.segmenter))
    report: dict[str, object] = {"detector": str(args.detector), "segmenter": str(args.segmenter), "tests": {}}
    for zip_name in ("test_1.zip", "test_2.zip", "test_3.zip"):
        images, targets = load_color_zip(zip_name)
        predictions = predict_teacher(detector, segmenter, images, args.imgsz)
        report["tests"][zip_name] = _metrics(predictions, targets)
        print(zip_name, json.dumps(report["tests"][zip_name], indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
