"""Generate center/tip training data with adaptive crop scales.

For each image, the crop scale adapts to the gauge size:
  - Large gauges (fill >70%): tight crop (1.0-1.1x)
  - Medium gauges (fill 30-70%): standard crop (1.2-1.4x)
  - Small gauges (fill <30%): wide crop (1.4-1.6x)

This ensures the gauge fills a consistent fraction of the crop regardless
of distance from camera.  Also handles crop clipping when the crop extends
beyond image boundaries.

Uses the linear radius QAT ellipse model for crop conditioning.
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
ELLIPSE_DATA = ROOT / "data" / "gauge_face_ellipse_v1_640_gray"
POINT_DATA = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip"
ELLIPSE_IMAGES = ROOT / "data" / "initial_temp_gauge_v1" / "ellipse"
OUTPUT = ROOT / "data" / "center_tip_adaptive_v1"
SIZE = 160
HEATMAP = 80
ELL_INPUT_SIZE = 224

# Target gauge fill fraction in the crop (what we want the gauge to occupy)
TARGET_FILL = 0.65


def adaptive_crop_scale(gauge_fill: float) -> float:
    """Compute crop scale based on gauge fill fraction.

    The goal is to make the gauge occupy ~65% of the crop regardless of
    its size in the original image.  If the gauge is already large, we use
    a tight crop.  If it's small, we use a wider crop.

    Args:
        gauge_fill: fraction of the image width occupied by the gauge
                    (2 * max(rx, ry) / image_size)

    Returns:
        Crop scale multiplier (side = 2 * max(rx, ry) * crop_scale)
    """
    if gauge_fill <= 0:
        return 1.35  # fallback

    # The crop side will be: 2 * radius * crop_scale
    # After resize to SIZE, the gauge fills: (2 * radius) / (2 * radius * crop_scale) = 1 / crop_scale
    # We want this to equal TARGET_FILL, so: crop_scale = 1 / TARGET_FILL
    # But we also need to ensure the crop fits within the image.

    # Ideal crop scale for target fill
    ideal_scale = 1.0 / TARGET_FILL  # ~1.54

    # If gauge is very large (fill > 80%), use tighter crop to avoid clipping
    if gauge_fill > 0.80:
        # Crop must fit in image: 2*radius*scale <= image_size
        # So: scale <= image_size / (2*radius) = 1 / gauge_fill
        max_scale = 0.95 / gauge_fill  # leave 5% margin
        return min(ideal_scale, max_scale)
    elif gauge_fill > 0.60:
        # Large gauge, use moderate crop
        return min(ideal_scale, 1.2)
    elif gauge_fill > 0.40:
        # Medium gauge
        return ideal_scale
    elif gauge_fill > 0.20:
        # Small gauge, use wider crop
        return ideal_scale
    else:
        # Very small gauge (far away), use widest crop
        return min(ideal_scale, 1.6)


def load_ellipse_model():
    interp = tf.lite.Interpreter(model_path=str(ELLIPSE_MODEL))
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()


def predict_ellipse(interp, inp_d, out_details, image_path: Path) -> np.ndarray:
    """Predict [cx, cy, rx, ry] using linear radius head model."""
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

    if "StatefulPartitionedCall:0" in preds and "StatefulPartitionedCall:2" in preds:
        center = preds["StatefulPartitionedCall:0"][0]
        radius = preds["StatefulPartitionedCall:2"][0]
    else:
        two_elem = [v[0] for v in preds.values() if v.shape[-1] == 2]
        if len(two_elem) >= 2:
            center, radius = two_elem[0], two_elem[1]
        else:
            center, radius = np.array([0.5, 0.5]), np.array([0.2, 0.2])
    return np.clip(np.concatenate([center, radius]), 0.02, 0.98)


def make_sample(
    image: np.ndarray,
    ellipse_norm: np.ndarray,
    center_norm: np.ndarray,
    tip_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Create one training sample with adaptive crop scale.

    Returns:
        inputs, heatmaps, points, crop_scale_used
    """
    source_size = max(image.shape)
    ecx, ecy, erx, ery = ellipse_norm * source_size

    # Compute gauge fill fraction
    gauge_radius = max(erx, ery)
    gauge_fill = 2.0 * gauge_radius / source_size

    # Adaptive crop scale
    crop_scale = adaptive_crop_scale(gauge_fill)
    side = 2.0 * gauge_radius * crop_scale

    # Ensure crop fits within image (clip if needed)
    max_side = source_size * 0.98  # leave 2% margin
    if side > max_side:
        side = max_side
        crop_scale = side / (2.0 * gauge_radius)

    left, top = ecx - side / 2.0, ecy - side / 2.0

    # Clamp crop to image boundaries
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + side > source_size:
        left = source_size - side
    if top + side > source_size:
        top = source_size - side

    left_i, top_i = int(max(0, left)), int(max(0, top))
    right_i = int(min(source_size, left + side))
    bottom_i = int(min(source_size, top + side))
    crop = Image.fromarray(image.astype(np.uint8)).crop((left_i, top_i, right_i, bottom_i))
    crop_160 = np.asarray(crop.resize((SIZE, SIZE), Image.Resampling.BILINEAR), np.float32) / 255.0

    # Ellipse mask
    xs = (np.arange(SIZE, dtype=np.float32) + 0.5) / float(SIZE) * side + left
    ys = (np.arange(SIZE, dtype=np.float32) + 0.5) / float(SIZE) * side + top
    xx, yy = np.meshgrid(xs, ys)
    mask = (((xx - ecx) / max(erx, 1.0)) ** 2 + ((yy - ecy) / max(ery, 1.0)) ** 2 <= 1.0).astype(np.float32)
    inputs = np.stack([crop_160 * 2.0 - 1.0, mask * 2.0 - 1.0], axis=-1)

    # Map center/tip to local crop coordinates
    source_points = np.array([center_norm, tip_norm]) * source_size
    local_points = np.clip((source_points - np.array([left, top])) / side, 0.0, 1.0)

    # Generate heatmaps
    heatmaps = np.zeros((HEATMAP, HEATMAP, 2), dtype=np.float32)
    yyh, xxh = np.mgrid[0:HEATMAP, 0:HEATMAP]
    for ch, pt in enumerate(local_points):
        px, py = pt * HEATMAP - 0.5
        heatmaps[..., ch] = np.exp(-((xxh - px) ** 2 + (yyh - py) ** 2) / (2.0 * 2.2 ** 2))

    return inputs.astype(np.float32), heatmaps, local_points.astype(np.float32), crop_scale


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    interp, inp_d, out_details = load_ellipse_model()

    # Load center/tip metadata
    meta = json.loads((POINT_DATA / "metadata.json").read_text())

    # Also load generic gauge metadata for center/tip labels
    generic_meta_path = ELLIPSE_DATA / "metadata.json"
    generic_meta = json.loads(generic_meta_path.read_text()) if generic_meta_path.exists() else None

    for split in ("train", "val", "test"):
        samples, targets, points_list, scales = [], [], [], []
        stems_seen = set()

        # 1. Generic gauge images (from ellipse dataset with center/tip labels)
        if generic_meta and split in generic_meta.get("splits", {}):
            for row in generic_meta["splits"][split]:
                stem = row.get("stem", Path(row.get("image", "")).stem)
                if stem in stems_seen:
                    continue
                if "center_xy_norm" not in row or "tip_xy_norm" not in row:
                    continue
                stems_seen.add(stem)

                image_path = ELLIPSE_DATA / "images" / split / f"{stem}.png"
                if not image_path.exists():
                    continue

                center_norm = np.array(row["center_xy_norm"], dtype=np.float32)
                tip_norm = np.array(row["tip_xy_norm"], dtype=np.float32)
                image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
                ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_path)

                inp, hm, pts, scale = make_sample(image, ellipse_norm, center_norm, tip_norm)
                samples.append(inp)
                targets.append(hm)
                points_list.append(pts)
                scales.append(scale)

        # 2. LittleGood images (from center_tip metadata)
        if split in meta.get("splits", {}):
            for row in meta["splits"][split]:
                stem = row.get("stem", "")
                if stem in stems_seen:
                    continue
                stems_seen.add(stem)

                image_path = ELLIPSE_IMAGES / "images" / split / f"{stem}.png"
                if not image_path.exists():
                    image_path = ROOT / "data" / "initial_temp_gauge_v1" / "center_tip" / "images" / split / f"{stem}.png"
                if not image_path.exists():
                    continue

                center_norm = np.array(row["center_xy_norm"], dtype=np.float32)
                tip_norm = np.array(row["tip_xy_norm"], dtype=np.float32)
                image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
                ellipse_norm = predict_ellipse(interp, inp_d, out_details, image_path)

                inp, hm, pts, scale = make_sample(image, ellipse_norm, center_norm, tip_norm)
                samples.append(inp)
                targets.append(hm)
                points_list.append(pts)
                scales.append(scale)

        if samples:
            np.savez_compressed(
                OUTPUT / f"{split}.npz",
                inputs=np.stack(samples),
                heatmaps=np.stack(targets),
                points=np.stack(points_list),
            )
            scale_arr = np.array(scales)
            print(f"{split}: {len(samples)} samples, "
                  f"crop_scale: min={scale_arr.min():.2f} max={scale_arr.max():.2f} "
                  f"mean={scale_arr.mean():.2f} median={np.median(scale_arr):.2f}")
        else:
            print(f"{split}: 0 samples")

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
