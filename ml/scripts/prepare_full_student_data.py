"""Add board_captures_2 and test_3 images to the student_conditioned dataset.

These images have center/tip labels but weren't included in the original
student_conditioned data.  Uses the multi-head QAT ellipse model for crop
conditioning.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ELLIPSE_MODEL = ROOT / "artifacts" / "gauge_ellipse_qat_linear_v1" / "ellipse_qat_linear_int8.tflite"
LABELLED = ROOT / "data" / "labelled"
STUDENT_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned"
OUTPUT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned_full"
SIZE = 160
HEATMAP = 80
CROP_SCALE = 1.35
ELL_INPUT_SIZE = 224


def load_ellipse_model():
    interp = tf.lite.Interpreter(model_path=str(ELLIPSE_MODEL))
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()


def predict_ellipse(interp, inp_d, out_details, image: np.ndarray) -> np.ndarray:
    """Predict [cx, cy, rx, ry] from a grayscale image array.

    For the linear radius head model:
    - StatefulPartitionedCall:0 = center_xy (sigmoid, coarser quant)
    - StatefulPartitionedCall:2 = radius_xy (linear, finer quant)
    """
    img = Image.fromarray(image.astype(np.uint8)).convert("L")
    img_arr = np.asarray(img.resize((ELL_INPUT_SIZE, ELL_INPUT_SIZE)), np.float32) / 255.0
    s, zp = inp_d["quantization"]
    t = np.clip(np.round(img_arr[None, ..., None] / float(s) + float(zp)), -128, 127).astype(np.int8)
    interp.set_tensor(inp_d["index"], t)
    interp.invoke()

    preds = {}
    for od in out_details:
        raw = interp.get_tensor(od["index"]).astype(np.float32)
        s, zp = od["quantization"]
        preds[od["name"]] = (raw - float(zp)) * float(s)

    # Try named outputs first (linear radius head model)
    if "StatefulPartitionedCall:0" in preds and "StatefulPartitionedCall:2" in preds:
        center = preds["StatefulPartitionedCall:0"][0]
        radius = preds["StatefulPartitionedCall:2"][0]
    else:
        # Fallback: use first two 2-element outputs
        two_elem = [v[0] for v in preds.values() if v.shape[-1] == 2]
        if len(two_elem) >= 2:
            center, radius = two_elem[0], two_elem[1]
        else:
            center, radius = np.array([0.5, 0.5]), np.array([0.2, 0.2])
    return np.clip(np.concatenate([center, radius]), 0.02, 0.98)


def make_sample(image, ellipse_norm, center_norm, tip_norm):
    source_size = max(image.shape)
    ecx, ecy, erx, ery = ellipse_norm * source_size
    side = max(2.0 * erx, 2.0 * ery) * CROP_SCALE
    left, top = ecx - side / 2.0, ecy - side / 2.0

    left_i, top_i = int(max(0, left)), int(max(0, top))
    right_i = int(min(source_size, left + side))
    bottom_i = int(min(source_size, top + side))
    crop = Image.fromarray(image.astype(np.uint8)).crop((left_i, top_i, right_i, bottom_i))
    crop_160 = np.asarray(crop.resize((SIZE, SIZE), Image.Resampling.BILINEAR), np.float32) / 255.0

    xs = (np.arange(SIZE, dtype=np.float32) + 0.5) / float(SIZE) * side + left
    ys = (np.arange(SIZE, dtype=np.float32) + 0.5) / float(SIZE) * side + top
    xx, yy = np.meshgrid(xs, ys)
    mask = (((xx - ecx) / max(erx, 1.0)) ** 2 + ((yy - ecy) / max(ery, 1.0)) ** 2 <= 1.0).astype(np.float32)
    inputs = np.stack([crop_160 * 2.0 - 1.0, mask * 2.0 - 1.0], axis=-1)

    source_points = np.array([center_norm, tip_norm]) * source_size
    local_points = np.clip((source_points - np.array([left, top])) / side, 0.0, 1.0)

    heatmaps = np.zeros((HEATMAP, HEATMAP, 2), dtype=np.float32)
    yyh, xxh = np.mgrid[0:HEATMAP, 0:HEATMAP]
    for ch, pt in enumerate(local_points):
        px, py = pt * HEATMAP - 0.5
        heatmaps[..., ch] = np.exp(-((xxh - px) ** 2 + (yyh - py) ** 2) / (2.0 * 2.2 ** 2))

    return inputs.astype(np.float32), heatmaps, local_points.astype(np.float32)


def process_board_captures_2(interp, inp_d, out_details):
    """Process board_captures_2.zip (22 images with temp_center/temp_tip points)."""
    z = zipfile.ZipFile(LABELLED / "initial_temp_gauge" / "board_captures_2.zip")
    tree = ET.parse(z.open("annotations.xml"))
    root = tree.getroot()

    samples, heatmaps, points = [], [], []
    for img_elem in root.findall("image"):
        name = img_elem.get("name")
        w = int(img_elem.get("width", "224"))
        h = int(img_elem.get("height", "224"))

        center_xy = tip_xy = None
        for elem in img_elem:
            label = elem.get("label", "")
            if label == "temp_center" and elem.tag == "points":
                pts = elem.get("points", "").split(",")
                center_xy = np.array([float(pts[0]) / w, float(pts[1]) / h], dtype=np.float32)
            elif label == "temp_tip" and elem.tag == "points":
                pts = elem.get("points", "").split(",")
                tip_xy = np.array([float(pts[0]) / w, float(pts[1]) / h], dtype=np.float32)

        if center_xy is None or tip_xy is None:
            continue

        data = z.read(f"images/{name}")
        image = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32)
        ellipse_norm = predict_ellipse(interp, inp_d, out_details, image)
        inp, hm, pts = make_sample(image, ellipse_norm, center_xy, tip_xy)
        samples.append(inp)
        heatmaps.append(hm)
        points.append(pts)

    return samples, heatmaps, points


def process_test_3(interp, inp_d, out_details):
    """Process test_3.zip (11 images with Center/Tip ellipses)."""
    z = zipfile.ZipFile(LABELLED / "test_3.zip")
    tree = ET.parse(z.open("annotations.xml"))
    root = tree.getroot()

    samples, heatmaps, points = [], [], []
    for img_elem in root.findall("image"):
        name = img_elem.get("name")
        w = int(img_elem.get("width", "640"))
        h = int(img_elem.get("height", "640"))

        center_xy = tip_xy = None
        for elem in img_elem:
            label = elem.get("label", "")
            if label == "Center" and elem.tag == "ellipse":
                cx, cy = float(elem.get("cx")), float(elem.get("cy"))
                center_xy = np.array([cx / w, cy / h], dtype=np.float32)
            elif label == "Tip" and elem.tag == "ellipse":
                cx, cy = float(elem.get("cx")), float(elem.get("cy"))
                tip_xy = np.array([cx / w, cy / h], dtype=np.float32)

        if center_xy is None or tip_xy is None:
            continue

        data = z.read(f"images/{name}")
        image = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32)
        ellipse_norm = predict_ellipse(interp, inp_d, out_details, image)
        inp, hm, pts = make_sample(image, ellipse_norm, center_xy, tip_xy)
        samples.append(inp)
        heatmaps.append(hm)
        points.append(pts)

    return samples, heatmaps, points


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    interp, inp_d, out_details = load_ellipse_model()

    # Load existing student_conditioned data
    for split in ("train", "val", "test"):
        existing = np.load(STUDENT_DATA / f"{split}.npz")
        ex_inputs = list(existing["inputs"])
        ex_heatmaps = list(existing["heatmaps"])
        ex_points = list(existing["points"])
        print(f"{split} existing: {len(ex_inputs)} samples")

        # Add new samples (only to train split)
        if split == "train":
            bc2_samples, bc2_heatmaps, bc2_points = process_board_captures_2(interp, inp_d, out_details)
            t3_samples, t3_heatmaps, t3_points = process_test_3(interp, inp_d, out_details)

            ex_inputs.extend(bc2_samples)
            ex_heatmaps.extend(bc2_heatmaps)
            ex_points.extend(bc2_points)
            ex_inputs.extend(t3_samples)
            ex_heatmaps.extend(t3_heatmaps)
            ex_points.extend(t3_points)
            print(f"  Added {len(bc2_samples)} board_captures_2 + {len(t3_samples)} test_3")

        np.savez_compressed(
            OUTPUT / f"{split}.npz",
            inputs=np.stack(ex_inputs),
            heatmaps=np.stack(ex_heatmaps),
            points=np.stack(ex_points),
        )
        print(f"{split} final: {len(ex_inputs)} samples")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
