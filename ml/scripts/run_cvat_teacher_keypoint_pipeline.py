"""Run the real CVAT YOLO11m ellipse teacher into the compact keypoint heads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# why: this local torchvision build lacks compiled NMS; use the same fallback
# as the teacher comparison so inference can run without reinstalling packages.
_torchvision_library = torch.library.Library("torchvision", "DEF")
_torchvision_library.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
_torchvision_library.define("qnms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
import torchvision


def _nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Perform greedy NMS with torch tensors."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        index = order[0]
        keep.append(index)
        if order.numel() == 1:
            break
        current, rest = boxes[index], boxes[order[1:]]
        xx1, yy1 = torch.maximum(current[0], rest[:, 0]), torch.maximum(current[1], rest[:, 1])
        xx2, yy2 = torch.minimum(current[2], rest[:, 2]), torch.minimum(current[3], rest[:, 3])
        intersection = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        area = (current[2] - current[0]) * (current[3] - current[1])
        areas = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        order = order[1:][intersection / (area + areas - intersection + 1e-7) <= threshold]
    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long)


torchvision.ops.nms = _nms
from ultralytics import YOLO

import tensorflow as tf

from train_gauge_center_tip_vector_v1 import predict_int8 as predict_vector
from train_gauge_center_tip_hybrid_v1 import decode_tip, predict_int8 as predict_hybrid


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
FULL_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
TEACHER = Path(os.environ.get("GAUGE_TEACHER", "/mnt/d/Projects/cvat/serverless/custom/gaugeface-yolo11m/nuclio/gaugeface_best.pt"))
VECTOR = ROOT / "artifacts" / "gauge_center_tip_vector_littlegood_v7" / "gauge_center_tip_vector_v1_int8.tflite"
HYBRID = ROOT / "artifacts" / "gauge_center_tip_hybrid_littlegood_v5" / "gauge_center_tip_hybrid_v1_int8.tflite"
OUTPUT = ROOT / "artifacts" / "cvat_teacher_keypoint_pipeline_v1"


def ellipse_from_result(result: object, image: np.ndarray) -> np.ndarray:
    """Apply CVAT's detector-box to ellipse refinement and return cx/cy/rx/ry."""
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    index = int(np.argmax(scores))
    x1, y1, x2, y2 = boxes[index].astype(int)
    width, height = x2 - x1, y2 - y1
    left, top = max(0, int(x1 - .35 * width)), max(0, int(y1 - .35 * height))
    right, bottom = min(image.shape[1], int(x2 + .35 * width)), min(image.shape[0], int(y2 + .35 * height))
    # CVAT's edge-fit path is replicated in the comparison script; this helper
    # uses the detector rectangle as a stable ellipse when a contour is absent.
    gray = np.asarray(Image.fromarray(image[top:bottom, left:right]).convert("L"))
    import cv2
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 110)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    expected = .5 * min(width, height)
    tx, ty = (x1 + x2) / 2 - left, y1 + .42 * height - top
    candidates = []
    for contour in contours:
        if len(contour) < 40:
            continue
        (cx, cy), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        radius = .25 * (axis_a + axis_b)
        ratio = max(axis_a, axis_b) / max(1.0, min(axis_a, axis_b))
        if radius < .35 * expected or radius > 1.5 * expected or ratio > 2.0:
            continue
        score = np.hypot(cx - tx, cy - ty) / max(1.0, expected) + abs(radius - expected) / max(1.0, expected)
        candidates.append((score, cx, cy, axis_a, axis_b))
    if candidates:
        _, cx, cy, axis_a, axis_b = min(candidates)
        return np.asarray([left + cx, top + cy, max(axis_a, axis_b) / 2, min(axis_a, axis_b) / 2], dtype=np.float32)
    return np.asarray([(x1 + x2) / 2, (y1 + y2) / 2, width / 2, height / 2], dtype=np.float32)


def center_input(image: np.ndarray, ellipse: np.ndarray) -> np.ndarray:
    """Build the compact heads' ellipse-conditioned 160x160 input."""
    cx, cy, rx, ry = ellipse
    side = max(2 * rx, 2 * ry) * 1.18
    left, top = cx - side / 2, cy - side / 2
    right, bottom = cx + side / 2, cy + side / 2
    # Crop the full frame before resizing; resizing the whole frame would
    # silently change the coordinate contract learned by the keypoint heads.
    source = Image.fromarray(image).convert("L").crop((left, top, right, bottom)).resize((160, 160), Image.Resampling.BILINEAR)
    gray = np.asarray(source, dtype=np.float32) / 255.0
    axis = (np.arange(160, dtype=np.float32) + .5) / 160 * side - side / 2
    xx, yy = np.meshgrid(axis + cx, axis + cy)
    mask = (((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1).astype(np.float32)
    return np.stack([gray * 2 - 1, mask * 2 - 1], axis=-1)


def main() -> None:
    """Render six random teacher-to-keypoint LittleGood overlays."""
    rows = json.loads((DATA / "metadata.json").read_text())["splits"]["test"]
    selected = np.random.default_rng(42).choice(rows, size=6, replace=False).tolist()
    teacher = YOLO(str(TEACHER))
    # why: the validation harness must remain runnable in CPU-only Poetry
    # environments; training/export still use the configured GPU when present.
    teacher_device = 0 if torch.cuda.is_available() else "cpu"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panels, report = [], []
    for row in selected:
        image_path = FULL_DATA / "images" / "test" / f"{row['stem']}.png"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        ellipse = ellipse_from_result(teacher.predict(image, conf=.25, device=teacher_device, verbose=False)[0], image)
        conditioned = center_input(image, ellipse)
        vector = predict_vector(VECTOR, conditioned[None])[0][:2]
        _, heatmap = predict_hybrid(HYBRID, conditioned[None])
        local_tip = decode_tip(heatmap)[0]
        cx, cy, rx, ry = ellipse
        side = max(2 * rx, 2 * ry) * 1.18
        predicted = np.asarray([[vector[0] * side + cx - side / 2, vector[1] * side + cy - side / 2], [local_tip[0] * side + cx - side / 2, local_tip[1] * side + cy - side / 2]])
        target = np.asarray([row["center_xy_norm"], row["tip_xy_norm"]], dtype=np.float32) * 640
        view = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(view)
        draw.rectangle((cx - rx, cy - ry, cx + rx, cy + ry), outline="red", width=4)
        for point, color in zip(target, ("lime", "cyan")):
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color, outline="white", width=2)
        for point, color in zip(predicted, ("red", "orange")):
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color, outline="white", width=2)
        panels.append(view.resize((320, 320)))
        report.append({"stem": row["stem"], "ellipse": ellipse.tolist(), "center_error_px": float(np.linalg.norm(predicted[0] - target[0])), "tip_error_px": float(np.linalg.norm(predicted[1] - target[1]))})
    sheet = Image.new("RGB", (960, 320 * len(panels)), "black")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 320))
    sheet.save(OUTPUT / "contact_sheet.png")
    (OUTPUT / "predictions.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"contact_sheet": str(OUTPUT / "contact_sheet.png"), "center_mean_px": float(np.mean([x["center_error_px"] for x in report])), "tip_mean_px": float(np.mean([x["tip_error_px"] for x in report]))}, indent=2))


if __name__ == "__main__":
    main()
