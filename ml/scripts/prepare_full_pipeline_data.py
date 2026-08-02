"""Generate center/tip training data using the full pipeline.

For each image:
  1. Load 640x640 grayscale image
  2. Predict ellipse with linear radius QAT model
  3. Apply source crop (1.18x from 640px) → resize to 160x160
  4. Map ellipse to 160px crop space
  5. Draw mask with 1.35x scale in 160px space
  6. Generate center/tip heatmap targets

This matches the evaluation pipeline exactly, ensuring the model sees the
same crop distribution during training and inference.
"""

from __future__ import annotations

import json
import zipfile
import io
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ELLIPSE_MODEL = ROOT / "artifacts" / "gauge_ellipse_qat_encoder_384g_cvat_v1" / "model_int8.tflite"
LABELLED = ROOT / "data" / "labelled"
ELLIPSE_IMAGES = ROOT / "data" / "gauge_face_ellipse_v1_640_gray"
POINT_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
TEMP_ELLIPSE = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
OUTPUT = ROOT / "data" / "center_tip_full_pipeline_v1"

INPUT_SIZE = 160
HEATMAP_SIZE = 80
ELL_INPUT_SIZE = 384
SOURCE_CROP_SCALE = 1.18  # pre-crop from 640px (matches training data generation)
MASK_CROP_SCALE = 1.35     # mask drawn in 160px space (matches training)


def load_ellipse_model():
    interp = tf.lite.Interpreter(model_path=str(ELLIPSE_MODEL))
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()


def predict_ellipse(interp, inp_d, out_details, image_640: np.ndarray) -> np.ndarray:
    """Predict [cx, cy, rx, ry] in normalized [0,1] coords."""
    img_input = np.asarray(
        Image.fromarray(image_640.astype(np.uint8)).resize((ELL_INPUT_SIZE, ELL_INPUT_SIZE)),
        dtype=np.float32,
    ) / 255.0
    s, zp = inp_d["quantization"]
    t = np.clip(np.round(img_input[None, ..., None] / float(s) + float(zp)), -128, 127).astype(np.int8)
    interp.set_tensor(inp_d["index"], t)
    interp.invoke()

    preds = {}
    for od in out_details:
        raw = interp.get_tensor(od["index"]).astype(np.float32)
        s, zp = od["quantization"]
        preds[od["name"]] = (raw - float(zp)) * float(s)

    # The current 384px candidate exposes one five-value ellipse head. Keep
    # the older multi-head path as a narrow compatibility branch for local
    # experiments, but make the active preparation contract explicit.
    if len(out_details) == 1:
        return np.clip(preds[out_details[0]["name"]][0, :4], 0.02, 0.98)
    center = preds["StatefulPartitionedCall:0"][0]
    radius = preds["StatefulPartitionedCall:2"][0]
    return np.clip(np.concatenate([center, radius]), 0.02, 0.98)


def make_sample(
    image_640: np.ndarray,
    ellipse_norm: np.ndarray,
    center_640: np.ndarray,
    tip_640: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one training sample using the two-stage crop pipeline.

    Args:
        image_640: 640x640 grayscale image
        ellipse_norm: [cx, cy, rx, ry] in normalized [0,1] coords
        center_640: center point in 640px coords
        tip_640: tip point in 640px coords

    Returns:
        inputs (160x160x2), heatmaps (80x80x2), points (2x2) normalized in 160px space
    """
    ecx, ecy = ellipse_norm[0] * 640, ellipse_norm[1] * 640
    erx, ery = ellipse_norm[2] * 640, ellipse_norm[3] * 640

    # Stage 1: source crop from 640px (1.18x scale)
    src_side = max(2 * erx, 2 * ery) * SOURCE_CROP_SCALE
    src_left = max(0.0, ecx - src_side / 2.0)
    src_top = max(0.0, ecy - src_side / 2.0)
    if src_left + src_side > 640:
        src_left = max(0.0, 640.0 - src_side)
    if src_top + src_side > 640:
        src_top = max(0.0, 640.0 - src_side)

    src_li = int(max(0, src_left))
    src_ti = int(max(0, src_top))
    src_ri = int(min(640, src_left + src_side))
    src_bi = int(min(640, src_top + src_side))
    actual_src_side = src_ri - src_li

    if actual_src_side < 10:
        # Degenerate crop — skip
        return None, None, None

    crop = Image.fromarray(image_640.astype(np.uint8)).crop((src_li, src_ti, src_ri, src_bi))
    crop_160 = np.asarray(crop.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR), np.float32) / 255.0

    # Map ellipse to 160px crop space
    scale_to_160 = INPUT_SIZE / actual_src_side
    cx160 = (ecx - src_left) * scale_to_160
    cy160 = (ecy - src_top) * scale_to_160
    rx160 = erx * scale_to_160
    ry160 = ery * scale_to_160

    # Stage 2: mask with 1.35x scale in 160px space
    mask_side = max(2 * rx160, 2 * ry160) * MASK_CROP_SCALE
    xs = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / float(INPUT_SIZE) * mask_side + cx160 - mask_side / 2.0
    ys = (np.arange(INPUT_SIZE, dtype=np.float32) + 0.5) / float(INPUT_SIZE) * mask_side + cy160 - mask_side / 2.0
    xx, yy = np.meshgrid(xs, ys)
    mask = (((xx - cx160) / max(rx160, 1.0)) ** 2 + ((yy - cy160) / max(ry160, 1.0)) ** 2 <= 1.0).astype(np.float32)
    inputs = np.stack([crop_160 * 2.0 - 1.0, mask * 2.0 - 1.0], axis=-1)

    # Map center/tip to 160px crop space, then normalize to [0,1]
    center_160 = (center_640 - np.array([src_left, src_top])) * scale_to_160
    tip_160 = (tip_640 - np.array([src_left, src_top])) * scale_to_160
    center_norm = np.clip(center_160 / float(INPUT_SIZE), 0.0, 1.0)
    tip_norm = np.clip(tip_160 / float(INPUT_SIZE), 0.0, 1.0)

    # Generate heatmaps
    heatmaps = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE, 2), dtype=np.float32)
    yyh, xxh = np.mgrid[0:HEATMAP_SIZE, 0:HEATMAP_SIZE]
    for ch, pt_norm in enumerate([center_norm, tip_norm]):
        px, py = pt_norm * HEATMAP_SIZE - 0.5
        heatmaps[..., ch] = np.exp(-((xxh - px) ** 2 + (yyh - py) ** 2) / (2.0 * 2.2 ** 2))

    local_points = np.stack([center_norm, tip_norm], axis=0).astype(np.float32)
    return inputs.astype(np.float32), heatmaps, local_points


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    interp, inp_d, out_details = load_ellipse_model()

    # Load metadata for center/tip labels
    generic_meta_path = ROOT / "data" / "gauge_center_tip_v1_160_gray" / "metadata.json"
    generic_meta = json.loads(generic_meta_path.read_text()) if generic_meta_path.exists() else None
    point_meta = json.loads((POINT_DATA / "metadata.json").read_text()) if (POINT_DATA / "metadata.json").exists() else None

    # Build suffix lookup for generic images in the ellipse dataset
    def build_suffix_map(split: str) -> dict[str, Path]:
        ell_dir = ELLIPSE_IMAGES / "images" / split
        smap = {}
        if ell_dir.exists():
            for f in ell_dir.glob("*.png"):
                parts = f.stem.split("_", 3)
                if len(parts) >= 4 and parts[0] in ("train", "val", "test"):
                    smap[parts[3]] = f
        return smap

    # Process labelled zips for center/tip labels (test_1 has box annotations)
    def load_zip_labels(zip_name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return {image_name: (center_640, tip_640)} from CVAT zip."""
        zip_path = LABELLED / zip_name
        if not zip_path.exists():
            return {}
        z = zipfile.ZipFile(zip_path)
        root = ET.parse(z.open("annotations.xml")).getroot()
        labels = {}
        for img_elem in root.findall("image"):
            name = img_elem.get("name")
            w = int(img_elem.get("width", "640"))
            h = int(img_elem.get("height", "640"))
            center = tip = None
            for elem in img_elem:
                label = elem.get("label", "")
                if label in ("Center", "temp_center"):
                    if elem.tag == "box":
                        xtl, ytl = float(elem.get("xtl")), float(elem.get("ytl"))
                        xbr, ybr = float(elem.get("xbr")), float(elem.get("ybr"))
                        center = np.array([(xtl + xbr) / 2, (ytl + ybr) / 2]) * (640.0 / w)
                    elif elem.tag == "points":
                        pts = elem.get("points", "").split(",")
                        center = np.array([float(pts[0]), float(pts[1])]) * (640.0 / w)
                    elif elem.tag == "ellipse":
                        center = np.array([float(elem.get("cx")), float(elem.get("cy"))]) * (640.0 / w)
                elif label in ("Tip", "temp_tip"):
                    if elem.tag == "box":
                        xtl, ytl = float(elem.get("xtl")), float(elem.get("ytl"))
                        xbr, ybr = float(elem.get("xbr")), float(elem.get("ybr"))
                        tip = np.array([(xtl + xbr) / 2, (ytl + ybr) / 2]) * (640.0 / w)
                    elif elem.tag == "points":
                        pts = elem.get("points", "").split(",")
                        tip = np.array([float(pts[0]), float(pts[1])]) * (640.0 / w)
                    elif elem.tag == "ellipse":
                        tip = np.array([float(elem.get("cx")), float(elem.get("cy"))]) * (640.0 / w)
            if center is not None and tip is not None:
                # The image might be in a subfolder
                labels[name] = (center, tip)
        return labels

    for split in ("train", "val", "test"):
        samples, targets, points_list = [], [], []
        stems_seen = set()
        suffix_map = build_suffix_map(split)

        # 1. Generic gauge images from ellipse dataset
        if generic_meta and split in generic_meta.get("splits", {}):
            for row in generic_meta["splits"][split]:
                stem = row.get("stem", "")
                if stem in stems_seen or "center_xy_norm" not in row:
                    continue

                # Find matching image
                image_path = suffix_map.get(stem)
                if image_path is None:
                    continue
                stems_seen.add(stem)

                # Load 640px image
                image_640 = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)

                # Use PREDICTED ellipse for crop (matches production pipeline)
                ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_640)
                erx_640, ery_640 = ellipse_norm[2] * 640, ellipse_norm[3] * 640
                ecx_640, ecy_640 = ellipse_norm[0] * 640, ellipse_norm[1] * 640
                src_side = max(2 * erx_640, 2 * ery_640) * SOURCE_CROP_SCALE
                src_left = max(0, ecx_640 - src_side / 2)
                src_top = max(0, ecy_640 - src_side / 2)
                if src_left + src_side > 640:
                    src_left = max(0, 640 - src_side)
                if src_top + src_side > 640:
                    src_top = max(0, 640 - src_side)
                actual_side = min(src_side, 640)

                # Center/tip: map from GT crop normalized coords to 640px, then to predicted crop
                ell_gt = np.asarray(row["ellipse"], dtype=np.float32)
                if row.get("source_width"):
                    ell_gt_640 = ell_gt * (640.0 / float(row["source_width"]))
                else:
                    ell_gt_640 = ell_gt
                gt_side = max(2 * ell_gt_640[2], 2 * ell_gt_640[3]) * SOURCE_CROP_SCALE
                gt_left = max(0, ell_gt_640[0] - gt_side / 2)
                gt_top = max(0, ell_gt_640[1] - gt_side / 2)
                if gt_left + gt_side > 640:
                    gt_left = max(0, 640 - gt_side)
                if gt_top + gt_side > 640:
                    gt_top = max(0, 640 - gt_side)
                gt_actual = min(gt_side, 640)

                center_norm = np.asarray(row["center_xy_norm"], dtype=np.float32)
                tip_norm = np.asarray(row["tip_xy_norm"], dtype=np.float32)
                center_640 = np.array([gt_left, gt_top]) + center_norm * gt_actual
                tip_640 = np.array([gt_left, gt_top]) + tip_norm * gt_actual

                result = make_sample(image_640, ellipse_norm, center_640, tip_640)
                if result[0] is not None:
                    samples.append(result[0])
                    targets.append(result[1])
                    points_list.append(result[2])

        # 2. LittleGood images
        if point_meta and split in point_meta.get("splits", {}):
            for row in point_meta["splits"][split]:
                stem = row.get("stem", "")
                if stem in stems_seen:
                    continue
                stems_seen.add(stem)

                image_path = TEMP_ELLIPSE / "images" / split / f"{stem}.png"
                if not image_path.exists():
                    image_path = POINT_DATA / "images" / split / f"{stem}.png"
                if not image_path.exists():
                    continue

                image_640 = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
                # Handle non-640px images
                if image_640.shape[0] != 640 or image_640.shape[1] != 640:
                    image_640 = np.asarray(
                        Image.fromarray(image_640.astype(np.uint8)).resize((640, 640)),
                        dtype=np.float32,
                    )

                # Use PREDICTED ellipse for crop
                ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_640)
                erx_640, ery_640 = ellipse_norm[2] * 640, ellipse_norm[3] * 640
                ecx_640, ecy_640 = ellipse_norm[0] * 640, ellipse_norm[1] * 640
                src_side = max(2 * erx_640, 2 * ery_640) * SOURCE_CROP_SCALE
                src_left = max(0, ecx_640 - src_side / 2)
                src_top = max(0, ecy_640 - src_side / 2)
                if src_left + src_side > 640:
                    src_left = max(0, 640 - src_side)
                if src_top + src_side > 640:
                    src_top = max(0, 640 - src_side)
                actual_side = min(src_side, 640)

                # Map center/tip from GT crop to predicted crop
                ell_gt = np.asarray(row["ellipse"], dtype=np.float32)
                if row.get("source_width"):
                    ell_gt_640 = ell_gt * (640.0 / float(row["source_width"]))
                else:
                    ell_gt_640 = ell_gt
                gt_side = max(2 * ell_gt_640[2], 2 * ell_gt_640[3]) * SOURCE_CROP_SCALE
                gt_left = max(0, ell_gt_640[0] - gt_side / 2)
                gt_top = max(0, ell_gt_640[1] - gt_side / 2)
                if gt_left + gt_side > 640:
                    gt_left = max(0, 640 - gt_side)
                if gt_top + gt_side > 640:
                    gt_top = max(0, 640 - gt_side)
                gt_actual = min(gt_side, 640)

                center_norm = np.asarray(row["center_xy_norm"], dtype=np.float32)
                tip_norm = np.asarray(row["tip_xy_norm"], dtype=np.float32)
                center_640 = np.array([gt_left, gt_top]) + center_norm * gt_actual
                tip_640 = np.array([gt_left, gt_top]) + tip_norm * gt_actual

                # Keep the points transformed through the labelled ellipse
                # crop.  Replacing them with normalized coordinates from the
                # predicted crop would silently corrupt targets whenever the
                # ellipse detector is offset or has a different radius.
                result = make_sample(image_640, ellipse_norm, center_640, tip_640)
                if result[0] is not None:
                    samples.append(result[0])
                    targets.append(result[1])
                    points_list.append(result[2])

        # 3. test_1 images (from zip with box annotations)
        if split == "train":
            test1_labels = load_zip_labels("test_1.zip")
            z = zipfile.ZipFile(LABELLED / "test_1.zip")
            count = 0
            for img_name, (center_640, tip_640) in test1_labels.items():
                if img_name in stems_seen:
                    continue
                stems_seen.add(img_name)
                try:
                    data = z.read(f"images/{img_name}")
                    image_640 = np.asarray(Image.open(io.BytesIO(data)).convert("L").resize((640, 640)), dtype=np.float32)
                except Exception:
                    continue

                ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_640)
                result = make_sample(image_640, ellipse_norm, center_640, tip_640)
                if result[0] is not None:
                    samples.append(result[0])
                    targets.append(result[1])
                    points_list.append(result[2])
                    count += 1
            if count > 0:
                print(f"  Added {count} test_1 images to train")

        # 4. test_3 images (board captures with ellipse annotations for Center/Tip)
        if split == "train":
            test3_path = LABELLED / "test_3.zip"
            if test3_path.exists():
                z3 = zipfile.ZipFile(test3_path)
                root3 = ET.parse(z3.open("annotations.xml")).getroot()
                count3 = 0
                for img_elem in root3.findall("image"):
                    name = img_elem.get("name")
                    stem = Path(name).stem
                    if stem in stems_seen:
                        continue
                    stems_seen.add(stem)

                    # Extract Center and Tip from ellipse annotations (use center point)
                    center_640 = tip_640 = None
                    for elem in img_elem:
                        label = elem.get("label", "")
                        if label == "Center" and elem.tag == "ellipse":
                            center_640 = np.array([float(elem.get("cx")), float(elem.get("cy"))], dtype=np.float32)
                        elif label == "Tip" and elem.tag == "ellipse":
                            tip_640 = np.array([float(elem.get("cx")), float(elem.get("cy"))], dtype=np.float32)

                    if center_640 is None or tip_640 is None:
                        continue

                    try:
                        data = z3.read(f"images/{name}")
                        image_640 = np.asarray(
                            Image.open(io.BytesIO(data)).convert("L").resize((640, 640)),
                            dtype=np.float32,
                        )
                    except Exception:
                        continue

                    ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_640)
                    result = make_sample(image_640, ellipse_norm, center_640, tip_640)
                    if result[0] is not None:
                        samples.append(result[0])
                        targets.append(result[1])
                        points_list.append(result[2])
                        count3 += 1
                if count3 > 0:
                    print(f"  Added {count3} test_3 images to train")

        if samples:
            np.savez_compressed(
                OUTPUT / f"{split}.npz",
                inputs=np.stack(samples),
                heatmaps=np.stack(targets),
                points=np.stack(points_list),
            )
            print(f"{split}: {len(samples)} samples")
        else:
            print(f"{split}: 0 samples")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
