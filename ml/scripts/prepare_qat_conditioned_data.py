"""Generate center/tip training data using multi-head QAT ellipse model.

Uses the QAT multi-head ellipse model for crop conditioning on LittleGood
and board capture images.  Generic gauge images use ground truth ellipses.
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
POINT_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
ELLIPSE_IMAGES = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
OUTPUT = ROOT / "data" / "initial_temp_gauge_v1" / "student_conditioned_qat"
SIZE = 160
HEATMAP = 80
CROP_SCALE = 1.35
ELL_INPUT_SIZE = 224


def load_ellipse_model():
    interp = tf.lite.Interpreter(model_path=str(ELLIPSE_MODEL))
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()


def predict_ellipse(interp, inp_d, out_details, image_path: Path) -> np.ndarray:
    """Predict [cx, cy, rx, ry] using multi-head model."""
    img = Image.open(image_path).convert("L")
    img_arr = np.asarray(img.resize((ELL_INPUT_SIZE, ELL_INPUT_SIZE)), np.float32) / 255.0
    s, zp = inp_d["quantization"]
    t = np.clip(np.round(img_arr[None, ..., None] / float(s) + float(zp)), -128, 127).astype(np.int8)
    interp.set_tensor(inp_d["index"], t)
    interp.invoke()

    # Multi-head outputs: center_xy, radius_xy, confidence
    preds = {}
    for od in out_details:
        raw = interp.get_tensor(od["index"]).astype(np.float32)
        s, zp = od["quantization"]
        preds[od["name"]] = (raw - float(zp)) * float(s)

    # Extract center and radius from the correct output tensors
    center = None
    radius = None
    for name, val in preds.items():
        v = val[0]
        if len(v) == 2:
            if center is None:
                center = v
            else:
                radius = v

    if center is None or radius is None:
        # Fallback: use first two 2-element outputs
        two_elem = [v[0] for v in preds.values() if v.shape[-1] == 2]
        if len(two_elem) >= 2:
            center, radius = two_elem[0], two_elem[1]
        else:
            # Single output model fallback
            center = np.array([0.5, 0.5], dtype=np.float32)
            radius = np.array([0.2, 0.2], dtype=np.float32)

    result = np.concatenate([center, radius])
    return np.clip(result, 0.02, 0.98)


def make_sample(
    image: np.ndarray,
    ellipse_norm: np.ndarray,
    center_norm: np.ndarray,
    tip_norm: np.ndarray,
    source_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Load center/tip metadata
    meta = json.loads((POINT_DATA / "metadata.json").read_text())

    for split, rows in meta["splits"].items():
        samples, targets, points_list = [], [], []
        for row in rows:
            stem = row["stem"]
            # Source image is in the ellipse images directory
            image_path = ELLIPSE_IMAGES / "images" / split / f"{stem}.png"
            if not image_path.exists():
                continue

            image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
            center_norm = np.array(row["center_xy_norm"], dtype=np.float32)
            tip_norm = np.array(row["tip_xy_norm"], dtype=np.float32)

            # Predict ellipse with multi-head QAT model
            ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_path)

            inp, hm, pts = make_sample(image, ellipse_norm, center_norm, tip_norm, source_size=max(image.shape))
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

    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
