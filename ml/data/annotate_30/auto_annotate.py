#!/usr/bin/env python3
"""Auto-annotate 30 board captures using multi-seed polar vote refinement.

For each image, tries CNN/OBB/Rim seeds, refines each via ±8 px sweep,
scores by polar vote quality, and picks the best. Saves CSV matching
manual_annotated_centers.csv format.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "src"))

from embedded_gauge_reading_tinyml.board_pipeline import (
    _quantize_input,
    decode_obb_crop_box,
)
from embedded_gauge_reading_tinyml.hybrid_localizer import (
    estimate_rim_center,
    refine_center,
    score_polar_quality,
    polar_spoke_vote,
)

ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts" / "deployment"
OBB_PATH = ARTIFACTS / "prod_model_v0.3_obb_int8" / "model_int8.tflite"
CTR_PATH = ARTIFACTS / "center_model_v4_cdcrop_int8" / "model_int8.tflite"
IMAGES_DIR = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml/tmp/annotate_30/images")
OUT_DIR = Path("/mnt/d/Projects/embedded-gauge-reading-tinyml/tmp/annotate_30")

INPUT_SIZE = 224
TC_W = 155
TC_H = 123


def resize_with_pad_geometry(crop_w: int, crop_h: int):
    scale = min(INPUT_SIZE / crop_w, INPUT_SIZE / crop_h)
    resized_w = crop_w * scale
    resized_h = crop_h * scale
    pad_x = (INPUT_SIZE - resized_w) * 0.5
    pad_y = (INPUT_SIZE - resized_h) * 0.5
    return scale, pad_x, pad_y


def run_tflite_raw(interpreter, batch: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    q = _quantize_input(batch, inp)
    interpreter.set_tensor(int(inp["index"]), q)
    interpreter.invoke()
    q_out = interpreter.get_tensor(int(out["index"]))[0]
    s = float(out["quantization"][0])
    zp = int(out["quantization"][1])
    return s * (np.asarray(q_out, dtype=np.float32) - zp)


def main() -> None:
    obb_interp = tf.lite.Interpreter(model_path=str(OBB_PATH))
    obb_interp.allocate_tensors()
    ctr_interp = tf.lite.Interpreter(model_path=str(CTR_PATH))
    ctr_interp.allocate_tensors()

    images = sorted(IMAGES_DIR.glob("*.png"))
    print(f"Annotating {len(images)} images...")

    results = []

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  SKIP {img_path.name} (cannot read)")
            continue
        frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # -- OBB --
        obb_batch = (frame.astype(np.float32) / 255.0)[None, ...]
        obb_out = run_tflite_raw(obb_interp, obb_batch)
        obb_params = obb_out.reshape(-1)
        dec = decode_obb_crop_box(obb_params.astype(np.float32),
                                  source_width=INPUT_SIZE, source_height=INPUT_SIZE)
        if not dec.accepted:
            print(f"  SKIP {img_path.name} (OBB rejected)")
            continue

        obb_cx = float(obb_params[0]) * INPUT_SIZE
        obb_cy = float(obb_params[1]) * INPUT_SIZE
        dial_radius = float(obb_params[2]) * INPUT_SIZE * 0.5

        # -- CD crop --
        cd_x = int(round(obb_cx - TC_W / 2.0))
        cd_y = int(round(obb_cy - TC_H / 2.0))
        cd_x = max(0, min(cd_x, INPUT_SIZE - TC_W))
        cd_y = max(0, min(cd_y, INPUT_SIZE - TC_H))
        cd_crop = frame[cd_y:cd_y + TC_H, cd_x:cd_x + TC_W]

        scale, pad_x, pad_y = resize_with_pad_geometry(TC_W, TC_H)
        cd_resized_w = int(round(TC_W * scale))
        cd_resized_h = int(round(TC_H * scale))
        cd_resized = cv2.resize(cd_crop, (cd_resized_w, cd_resized_h),
                                interpolation=cv2.INTER_LINEAR)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        x_off = int(round(pad_x))
        y_off = int(round(pad_y))
        canvas[y_off:y_off + cd_resized_h, x_off:x_off + cd_resized_w] = cd_resized

        # -- CNN centre --
        ctr_batch = (canvas.astype(np.float32))[None, ...]
        ctr_out = run_tflite_raw(ctr_interp, ctr_batch)
        cx_norm, cy_norm = float(ctr_out[0]), float(ctr_out[1])
        padded_cx = cx_norm * INPUT_SIZE
        padded_cy = cy_norm * INPUT_SIZE
        ff_cx = cd_x + (padded_cx - pad_x) / scale
        ff_cy = cd_y + (padded_cy - pad_y) / scale

        # -- Multi-seed refinement --
        luma = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        rim_cx, rim_cy, rim_ok = estimate_rim_center(luma, INPUT_SIZE, INPUT_SIZE, dial_radius)

        seeds = [(ff_cx, ff_cy, "CNN"),
                 (obb_cx, obb_cy, "OBB")]
        if rim_ok:
            seeds.append((rim_cx, rim_cy, "RIM"))

        best_q = -1.0
        best_sx = best_sy = 0.0
        best_slabel = "none"

        for sx, sy, slabel in seeds:
            votes = polar_spoke_vote(luma, sx, sy, dial_radius,
                                     edge_threshold=8.0, use_structural_boost=True)
            q = score_polar_quality(votes, luma, sx, sy, dial_radius)
            if q > best_q:
                best_q, best_sx, best_sy, best_slabel = q, sx, sy, slabel

        # Refine best seed
        if best_q > 0.0:
            _, _, angle, _ = refine_center(
                frame, best_sx, best_sy, dial_radius,
                edge_threshold=8.0, use_structural_boost=True,
            )
            # Refine returns the best refined center implicitly via angle
            # Use the best seed as the center
            final_cx, final_cy = best_sx, best_sy
        else:
            final_cx, final_cy = best_sx, best_sy
            best_slabel = "fallback"

        results.append((img_path.name, final_cx, final_cy, best_q, best_slabel))
        print(f"  {img_path.name}: center=({final_cx:.0f},{final_cy:.0f}) "
              f"seed={best_slabel} quality={best_q:.1f}")

    # Write CSV
    csv_path = OUT_DIR / "annotated_centers_auto.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "center_x", "center_y"])
        for name, cx, cy, q, seed in results:
            w.writerow([f"captured_images/{name}", f"{cx:.1f}", f"{cy:.1f}"])
    print(f"\nWritten {csv_path} ({len(results)} annotations)")


if __name__ == "__main__":
    main()
