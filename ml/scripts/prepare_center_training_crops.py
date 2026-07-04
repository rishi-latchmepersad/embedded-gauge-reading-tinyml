"""Prepare CD-crop aligned centre training data from ALL board captures.

Uses every image in center_training_manual/images/ (374 captures), runs the
rim-vote centre estimator for reliable labels, and generates 5 CD-crops per
image with different OBB jitter seeds for data multiplication.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedded_gauge_reading_tinyml.rim_estimator import c_rim_estimator

MANUAL_DIR = PROJECT_ROOT / "data" / "center_training_manual"
OUT_DIR = PROJECT_ROOT / "data" / "center_training_crops"
OUT_IMAGES = OUT_DIR / "images"

INPUT_SIZE = 224
TC_W = 155
TC_H = 123
PAD_COLOR = 128

OBB_JITTER_X_STD = 6.0
OBB_JITTER_Y_STD = 2.0

CROPS_PER_IMAGE = 5
RNG_SEED = 42


def _resize_with_pad_scale(crop_w: int, crop_h: int) -> tuple[float, float, float]:
    scale = min(INPUT_SIZE / crop_w, INPUT_SIZE / crop_h)
    resized_w = crop_w * scale
    resized_h = crop_h * scale
    pad_x = (INPUT_SIZE - resized_w) * 0.5
    pad_y = (INPUT_SIZE - resized_h) * 0.5
    return scale, pad_x, pad_y


def _norm_to_cd_crop(
    cx_full: float, cy_full: float,
    cd_x: int, cd_y: int,
) -> tuple[float, float]:
    scale, pad_x, pad_y = _resize_with_pad_scale(TC_W, TC_H)
    padded_cx = (cx_full - cd_x) * scale + pad_x
    padded_cy = (cy_full - cd_y) * scale + pad_y
    return padded_cx / INPUT_SIZE, padded_cy / INPUT_SIZE


def rgb_to_yuv422(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    assert w % 2 == 0

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0

    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)

    packed = np.zeros((h, w * 2), dtype=np.uint8)
    packed[:, 0::4] = y[:, 0::2]
    packed[:, 1::4] = u[:, 0::2]
    packed[:, 2::4] = y[:, 1::2]
    packed[:, 3::4] = v[:, 0::2]

    return packed


def main() -> None:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    img_dir = MANUAL_DIR / "images"
    png_paths = sorted(img_dir.glob("*.png"))
    print(f"Found {len(png_paths)} images")

    rng = np.random.default_rng(RNG_SEED)

    cd_entries: list[dict] = []
    rim_ok = 0
    rim_fail = 0

    for img_path in png_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            rim_fail += 1
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        yuv422 = rgb_to_yuv422(img_rgb)

        try:
            gt_cx, gt_cy, found = c_rim_estimator.find_rim_center(
                yuv422.reshape(INPUT_SIZE, INPUT_SIZE, 2),
                dial_radius_px=68.9,
            )
        except Exception:
            rim_fail += 1
            continue

        if not found:
            rim_fail += 1
            continue

        rim_ok += 1

        for crop_idx in range(CROPS_PER_IMAGE):
            seed_i = RNG_SEED + crop_idx + rim_ok * 1000
            local_rng = np.random.default_rng(seed_i)

            obb_cx = gt_cx + float(local_rng.normal(0, OBB_JITTER_X_STD))
            obb_cy = gt_cy + float(local_rng.normal(0, OBB_JITTER_Y_STD))

            obb_cx = float(np.clip(obb_cx, TC_W / 2.0, INPUT_SIZE - TC_W / 2.0))
            obb_cy = float(np.clip(obb_cy, TC_H / 2.0, INPUT_SIZE - TC_H / 2.0))

            cd_x = int(round(obb_cx - TC_W / 2.0))
            cd_y = int(round(obb_cy - TC_H / 2.0))
            cd_x = max(0, min(cd_x, INPUT_SIZE - TC_W))
            cd_y = max(0, min(cd_y, INPUT_SIZE - TC_H))

            cd_crop = img_rgb[cd_y:cd_y + TC_H, cd_x:cd_x + TC_W]

            scale, pad_x, pad_y = _resize_with_pad_scale(TC_W, TC_H)
            resized_w = int(round(TC_W * scale))
            resized_h = int(round(TC_H * scale))
            cd_resized = cv2.resize(cd_crop, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_COLOR, dtype=np.uint8)
            x_off = int(round(pad_x))
            y_off = int(round(pad_y))
            canvas[y_off:y_off + resized_h, x_off:x_off + resized_w] = cd_resized

            cx_norm, cy_norm = _norm_to_cd_crop(gt_cx, gt_cy, cd_x, cd_y)

            if not (0.0 <= cx_norm <= 1.0 and 0.0 <= cy_norm <= 1.0):
                continue

            fname = f"cd_{img_path.stem}_c{crop_idx}.png"
            out_path = OUT_IMAGES / fname
            cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            cd_entries.append({
                "image_path": f"images/{fname}",
                "center_x_norm": round(cx_norm, 6),
                "center_y_norm": round(cy_norm, 6),
                "gt_cx_px": round(gt_cx, 2),
                "gt_cy_px": round(gt_cy, 2),
                "obb_cx_px": round(obb_cx, 2),
                "obb_cy_px": round(obb_cy, 2),
                "crop_idx": crop_idx,
                "split": "train",
                "source_image": img_path.name,
                "source": "board_rim",
            })

    print(f"\nRim estimator: {rim_ok} OK, {rim_fail} failed")
    print(f"Generated {len(cd_entries)} CD-crop entries ({rim_ok * CROPS_PER_IMAGE} expected)")

    # Stratified split: keep all crops of the same source image together
    source_images = sorted(set(e["source_image"] for e in cd_entries))
    train_src, val_src = train_test_split(
        source_images, test_size=0.20, random_state=RNG_SEED,
    )
    val_src_set = set(val_src)
    for e in cd_entries:
        if e["source_image"] in val_src_set:
            e["split"] = "val"

    n_train = sum(1 for e in cd_entries if e["split"] == "train")
    n_val = sum(1 for e in cd_entries if e["split"] == "val")
    print(f"{n_train} train, {n_val} val ({len(train_src)}/{len(val_src)} source images)")

    # Clean old metadata fields
    meta_out = OUT_DIR / "metadata.json"
    with open(meta_out, "w") as f:
        json.dump(cd_entries, f, indent=2)
    print(f"Metadata saved: {meta_out}")

    cx_arr = np.array([e["center_x_norm"] for e in cd_entries])
    cy_arr = np.array([e["center_y_norm"] for e in cd_entries])
    print(f"cx_norm: mean={cx_arr.mean():.4f} std={cx_arr.std():.4f} "
          f"range=[{cx_arr.min():.4f},{cx_arr.max():.4f}]")
    print(f"cy_norm: mean={cy_arr.mean():.4f} std={cy_arr.std():.4f} "
          f"range=[{cy_arr.min():.4f},{cy_arr.max():.4f}]")


if __name__ == "__main__":
    main()
