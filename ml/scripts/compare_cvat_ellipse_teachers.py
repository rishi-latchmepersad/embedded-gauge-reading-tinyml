"""Compare the CVAT YOLO11m ellipse teachers on the LittleGood test split."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# why: the local Poetry torchvision wheel lacks compiled NMS; this pure-torch
# fallback allows offline teacher comparison without changing model weights.
_torchvision_library = torch.library.Library("torchvision", "DEF")
_torchvision_library.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
_torchvision_library.define("qnms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
import torchvision


def _nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Perform ordinary greedy NMS using torch tensor operations."""
    order = scores.argsort(descending=True)
    kept: list[torch.Tensor] = []
    while order.numel():
        index = order[0]
        kept.append(index)
        if order.numel() == 1:
            break
        current = boxes[index]
        rest = boxes[order[1:]]
        xx1 = torch.maximum(current[0], rest[:, 0])
        yy1 = torch.maximum(current[1], rest[:, 1])
        xx2 = torch.minimum(current[2], rest[:, 2])
        yy2 = torch.minimum(current[3], rest[:, 3])
        intersection = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        area = (current[2] - current[0]) * (current[3] - current[1])
        areas = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        overlap = intersection / (area + areas - intersection + 1e-7)
        order = order[1:][overlap <= threshold]
    return torch.stack(kept) if kept else torch.empty((0,), dtype=torch.long)


torchvision.ops.nms = _nms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
CVAT_ROOT = Path("/mnt/d/Projects/cvat/serverless/custom")
TEACHERS = {name: CVAT_ROOT / name / "nuclio" / "gaugeface_best.pt" for name in ("gaugeface-yolo11m", "gaugeface-yolo11m-combined", "gaugeface-yolo11m-v2")}


def detection_box(result: object, width: int, height: int) -> list[int] | None:
    """Select the highest-confidence gauge/face/dial detection."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names
    candidates = [i for i, cls in enumerate(classes) if any(token in str(names[int(cls)]).lower() for token in ("gauge", "face", "dial"))]
    index = max(candidates or range(len(boxes)), key=lambda i: float(scores[i]))
    x1, y1, x2, y2 = boxes[index]
    return [max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))]


def cvat_ellipse(image: np.ndarray, box: list[int]) -> list[float]:
    """Reproduce the CVAT function's edge-fit ellipse fallback."""
    x1, y1, x2, y2 = box
    box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
    left, top = max(0, int(x1 - .35 * box_width)), max(0, int(y1 - .35 * box_height))
    right, bottom = min(image.shape[1], int(x2 + .35 * box_width)), min(image.shape[0], int(y2 + .35 * box_height))
    crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(crop, (5, 5), 0), 35, 110)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    expected = .5 * min(box_width, box_height)
    target_x, target_y = (x1 + x2) / 2 - left, y1 + .42 * box_height - top
    candidates = []
    for contour in contours:
        if len(contour) < 40:
            continue
        (cx, cy), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        radius = .25 * (axis_a + axis_b)
        ratio = max(axis_a, axis_b) / max(1.0, min(axis_a, axis_b))
        if radius < .35 * expected or radius > 1.5 * expected or ratio > 2.0:
            continue
        score = math.hypot(cx - target_x, cy - target_y) / max(1.0, expected) + abs(radius - expected) / max(1.0, expected)
        candidates.append((score, cx, cy, axis_a, axis_b))
    if not candidates:
        return [float(x1), float(y1), float(x2), float(y2)]
    _, cx, cy, axis_a, axis_b = min(candidates)
    return [left + cx - max(axis_a, axis_b) / 2, top + cy - min(axis_a, axis_b) / 2, left + cx + max(axis_a, axis_b) / 2, top + cy + min(axis_a, axis_b) / 2]


def target_box(path: Path, width: int, height: int) -> list[float]:
    """Read the axis-aligned labeled face box."""
    values = np.fromstring(path.read_text(), sep=" ")[1:9].reshape(4, 2)
    low, high = values.min(axis=0), values.max(axis=0)
    return [low[0] * width, low[1] * height, high[0] * width, high[1] * height]


def iou(first: list[float], second: list[float]) -> float:
    """Compute axis-aligned IoU."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    area_b = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / max(1.0, area_a + area_b - intersection)


def main() -> None:
    """Evaluate all CVAT teachers on all labeled LittleGood test images."""
    paths = sorted((DATA / "images" / "test").glob("*.png"))
    for name, path in TEACHERS.items():
        model = YOLO(str(path))
        overlaps, centers, misses = [], [], 0
        for image_path in paths:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            height, width = image.shape[:2]
            result = model.predict(image, conf=.25, device=0, verbose=False)[0]
            box = detection_box(result, width, height)
            if box is None:
                misses += 1
                continue
            predicted = cvat_ellipse(image, box)
            target = target_box(DATA / "labels" / "test" / f"{image_path.stem}.txt", width, height)
            overlaps.append(iou(predicted, target))
            centers.append(math.hypot((predicted[0] + predicted[2] - target[0] - target[2]) / 2, (predicted[1] + predicted[3] - target[1] - target[3]) / 2))
        print(name, {"samples": len(overlaps), "misses": misses, "iou_mean": float(np.mean(overlaps)), "iou_ge_0_5": float(np.mean(np.asarray(overlaps) >= .5)), "center_error_px_mean": float(np.mean(centers))})


if __name__ == "__main__":
    main()
