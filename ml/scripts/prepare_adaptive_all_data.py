"""Generate adaptive-crop training data for ALL images.

Matches generic gauge images from the ellipse dataset (prefixed filenames)
with center/tip labels from the center/tip metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ELLIPSE_MODEL = ROOT / "artifacts" / "gauge_ellipse_qat_linear_v1" / "ellipse_qat_linear_int8.tflite"
ELLIPSE_IMAGES = ROOT / "data" / "gauge_face_ellipse_v1_640_gray"
POINT_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
TEMP_ELLIPSE = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
OUTPUT = ROOT / "data" / "center_tip_adaptive_v2"
SIZE = 160
HEATMAP = 80
ELL_INPUT_SIZE = 224
TARGET_FILL = 0.65


def adaptive_crop_scale(gauge_fill: float) -> float:
    if gauge_fill <= 0:
        return 1.35
    ideal = 1.0 / TARGET_FILL
    if gauge_fill > 0.80:
        return min(ideal, 0.95 / gauge_fill)
    elif gauge_fill > 0.40:
        return ideal
    else:
        return min(ideal, 1.6)


def load_ellipse_model():
    interp = tf.lite.Interpreter(model_path=str(ELLIPSE_MODEL))
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()


def predict_ellipse(interp, inp_d, out_details, image_path):
    img = Image.open(image_path).convert("L")
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
    center = preds["StatefulPartitionedCall:0"][0]
    radius = preds["StatefulPartitionedCall:2"][0]
    return np.clip(np.concatenate([center, radius]), 0.02, 0.98)


def make_sample(image, ellipse_norm, center_norm, tip_norm):
    source_size = max(image.shape)
    ecx, ecy, erx, ery = ellipse_norm * source_size
    gauge_radius = max(erx, ery)
    gauge_fill = 2.0 * gauge_radius / source_size
    crop_scale = adaptive_crop_scale(gauge_fill)
    side = 2.0 * gauge_radius * crop_scale
    max_side = source_size * 0.98
    if side > max_side:
        side = max_side

    left, top = ecx - side / 2.0, ecy - side / 2.0
    if left < 0: left = 0
    if top < 0: top = 0
    if left + side > source_size: left = source_size - side
    if top + side > source_size: top = source_size - side

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

    return inputs.astype(np.float32), heatmaps, local_points.astype(np.float32), crop_scale


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    interp, inp_d, out_details = load_ellipse_model()

    # Build filename mapping: stem_suffix -> full_path
    # Generic images have prefix like "train_1_000000_"
    ell_train = ELLIPSE_IMAGES / "images" / "train"
    ell_val = ELLIPSE_IMAGES / "images" / "val"
    ell_test = ELLIPSE_IMAGES / "images" / "test"

    # Load center/tip metadata
    generic_meta = json.loads((ROOT / "data" / "gauge_center_tip_v1_160_gray" / "metadata.json").read_text())
    point_meta = json.loads((POINT_DATA / "metadata.json").read_text())

    for split in ("train", "val", "test"):
        samples, targets, points_list, scales = [], [], [], []
        stems_seen = set()

        # 1. Generic gauge images — find matching files in ellipse dataset
        ell_dir = ELLIPSE_IMAGES / "images" / split
        if ell_dir.exists():
            # Build suffix lookup: last part of filename without prefix -> full path
            suffix_map = {}
            for f in ell_dir.glob("*.png"):
                name = f.stem
                # Remove prefix like "train_1_000000_" or "val_1_000000_"
                parts = name.split("_", 3)
                if len(parts) >= 4 and parts[0] in ("train", "val", "test"):
                    suffix = parts[3]  # the original stem
                    suffix_map[suffix] = f

            for row in generic_meta["splits"][split]:
                stem = row.get("stem", "")
                if stem in stems_seen or "center_xy_norm" not in row:
                    continue

                # Find matching image
                image_path = suffix_map.get(stem)
                if image_path is None:
                    continue
                stems_seen.add(stem)

                center_norm = np.array(row["center_xy_norm"], dtype=np.float32)
                tip_norm = np.array(row["tip_xy_norm"], dtype=np.float32)
                image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
                ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_path)

                inp, hm, pts, scale = make_sample(image, ellipse_norm, center_norm, tip_norm)
                samples.append(inp); targets.append(hm); points_list.append(pts); scales.append(scale)

        # 2. LittleGood images
        for row in point_meta["splits"][split]:
            stem = row.get("stem", "")
            if stem in stems_seen:
                continue
            stems_seen.add(stem)

            image_path = TEMP_ELLIPSE / "images" / split / f"{stem}.png"
            if not image_path.exists():
                continue

            center_norm = np.array(row["center_xy_norm"], dtype=np.float32)
            tip_norm = np.array(row["tip_xy_norm"], dtype=np.float32)
            image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
            ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_path)

            inp, hm, pts, scale = make_sample(image, ellipse_norm, center_norm, tip_norm)
            samples.append(inp); targets.append(hm); points_list.append(pts); scales.append(scale)

        if samples:
            np.savez_compressed(OUTPUT / f"{split}.npz",
                               inputs=np.stack(samples), heatmaps=np.stack(targets),
                               points=np.stack(points_list))
            scale_arr = np.array(scales)
            print(f"{split}: {len(samples)} samples, scale: min={scale_arr.min():.2f} max={scale_arr.max():.2f} mean={scale_arr.mean():.2f}")
        else:
            print(f"{split}: 0 samples")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
