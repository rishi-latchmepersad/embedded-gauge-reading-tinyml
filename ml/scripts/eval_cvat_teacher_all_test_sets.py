#!/usr/bin/env python3
"""Benchmark a CVAT Ultralytics gauge-face teacher on all held-out zips."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from eval_ellipse_all_test_sets import LABELLED, _extract_ellipse, _metrics


def load_color_zip(zip_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load RGB frames while retaining the normalized ellipse targets."""
    images: list[np.ndarray] = []
    targets: list[tuple[float, float, float, float, float]] = []
    with zipfile.ZipFile(LABELLED / zip_name) as archive:
        members = {Path(member).name: member for member in archive.namelist()}
        import xml.etree.ElementTree as ET

        root = ET.fromstring(archive.read("annotations.xml"))
        for image_node in root.findall("image"):
            ellipse = _extract_ellipse(image_node)
            if ellipse is None:
                continue
            member = members.get(Path(image_node.get("name", "")).name)
            if member is None:
                continue
            image = Image.open(io.BytesIO(archive.read(member))).convert("RGB")
            image = image.resize((640, 640), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
            targets.append((*ellipse, 1.0))
    return np.asarray(images, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def predict_boxes(model: YOLO, images: np.ndarray, *, image_size: int, batch_size: int) -> np.ndarray:
    """Run the teacher and convert its highest-confidence box to an ellipse."""
    predictions: list[np.ndarray] = []
    rgb_images = (images * 255.0).astype(np.uint8)
    results = model.predict(
        source=list(rgb_images), imgsz=image_size, device=0, batch=batch_size, verbose=False, conf=0.05
    )
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            predictions.append(np.zeros(5, dtype=np.float32))
            continue
        confidence = result.boxes.conf.detach().cpu().numpy()
        index = int(np.argmax(confidence))
        box = result.boxes.xyxy[index].detach().cpu().numpy() / 640.0
        x0, y0, x1, y1 = box
        predictions.append(
            np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0) / 2.0, (y1 - y0) / 2.0, confidence[index]], dtype=np.float32)
        )
    return np.asarray(predictions, dtype=np.float32)


def main() -> None:
    """Evaluate the teacher and write per-domain geometry metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--test-zips", nargs="+", default=("test_1.zip", "test_2.zip", "test_3.zip"))
    args = parser.parse_args()
    model = YOLO(str(args.model))
    report: dict[str, object] = {"model": str(args.model), "tests": {}}
    for zip_name in args.test_zips:
        images, targets = load_color_zip(zip_name)
        predictions = predict_boxes(model, images, image_size=args.imgsz, batch_size=args.batch)
        report["tests"][zip_name] = _metrics(predictions, targets)
        print(zip_name, json.dumps(report["tests"][zip_name], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
