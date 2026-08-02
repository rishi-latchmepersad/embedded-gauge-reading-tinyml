"""Generate center/tip training data for ALL images using QAT ellipse model.

Processes:
  - Generic gauge images (7,328) from data/gauge_face_ellipse_v1_640_gray/
  - LittleGood images (451) from data/initial_temp_gauge_v1/ellipse/
  - Board captures (223) and test_3 (11) already in the ellipse dataset

Output: combined NPZ files with inputs (160x160x2), heatmaps (80x80x2), points (2x2)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ELLIPSE_MODEL = ROOT / "artifacts" / "gauge_ellipse_qat_multihead_v1" / "ellipse_qat_multihead_int8.tflite"
ELLIPSE_DATA = ROOT / "data" / "gauge_face_ellipse_v1_640_gray"
OUTPUT = ROOT / "data" / "center_tip_all_qat_v1"
SIZE = 160
HEATMAP = 80
CROP_SCALE = 1.35
ELL_INPUT_SIZE = 224


def load_ellipse_model():
    interp = tf.lite.Interpreter(model_path=str(ELLIPSE_MODEL))
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()


def predict_ellipse(interp, inp_d, out_details, image_path: Path) -> np.ndarray:
    """Predict [cx, cy, rx, ry] using multi-head QAT model."""
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

    # Multi-head: center_xy, radius_xy, confidence
    two_elem = [v[0] for v in preds.values() if v.shape[-1] == 2]
    if len(two_elem) >= 2:
        center, radius = two_elem[0], two_elem[1]
    else:
        center = np.array([0.5, 0.5], dtype=np.float32)
        radius = np.array([0.2, 0.2], dtype=np.float32)

    return np.clip(np.concatenate([center, radius]), 0.02, 0.98)


def make_sample(
    image: np.ndarray,
    ellipse_norm: np.ndarray,
    center_norm: np.ndarray,
    tip_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one training sample from image + ellipse + keypoint labels."""
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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    interp, inp_d, out_details = load_ellipse_model()

    # Load generic gauge metadata (has center/tip labels)
    generic_meta = json.loads((ELLIPSE_DATA / "metadata.json").read_text()) if (ELLIPSE_DATA / "metadata.json").exists() else None

    # Also load the center/tip metadata for LittleGood
    ct_meta_path = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip" / "metadata.json"
    ct_meta = json.loads(ct_meta_path.read_text()) if ct_meta_path.exists() else None

    for split in ("train", "val", "test"):
        samples, targets, points_list = [], [], []
        stems_seen = set()

        # 1. Generic gauge images (from ellipse dataset)
        if generic_meta and split in generic_meta.get("splits", {}):
            for row in generic_meta["splits"][split]:
                stem = row.get("stem", Path(row.get("image", "")).stem)
                if stem in stems_seen:
                    continue
                stems_seen.add(stem)

                image_path = ELLIPSE_DATA / "images" / split / f"{stem}.png"
                if not image_path.exists():
                    continue

                # Get center/tip from metadata if available
                if "center_xy_norm" in row and "tip_xy_norm" in row:
                    center_norm = np.array(row["center_xy_norm"], dtype=np.float32)
                    tip_norm = np.array(row["tip_xy_norm"], dtype=np.float32)
                else:
                    # Skip images without center/tip labels
                    continue

                image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
                ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_path)

                inp, hm, pts = make_sample(image, ellipse_norm, center_norm, tip_norm)
                samples.append(inp)
                targets.append(hm)
                points_list.append(pts)

        # 2. LittleGood images (from center_tip metadata)
        if ct_meta and split in ct_meta.get("splits", {}):
            for row in ct_meta["splits"][split]:
                stem = row.get("stem", "")
                if stem in stems_seen:
                    continue
                stems_seen.add(stem)

                image_path = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse" / "images" / split / f"{stem}.png"
                if not image_path.exists():
                    # Try center_tip images
                    image_path = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip" / "images" / split / f"{stem}.png"
                if not image_path.exists():
                    continue

                center_norm = np.array(row["center_xy_norm"], dtype=np.float32)
                tip_norm = np.array(row["tip_xy_norm"], dtype=np.float32)

                image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
                ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_path)

                inp, hm, pts = make_sample(image, ellipse_norm, center_norm, tip_norm)
                samples.append(inp)
                targets.append(hm)
                points_list.append(pts)

        if samples:
            np.savez_compressed(
                OUTPUT / f"{split}.npz",
                inputs=np.stack(samples),
                heatmaps=np.stack(targets),
                points=np.stack(points_list),
            )
            print(f"{split}: {len(samples)} samples")
        else:
            print(f"{split}: 0 samples (no data)")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
