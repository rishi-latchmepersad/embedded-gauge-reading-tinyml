#!/usr/bin/env python3
"""Generate CD-crop training examples that exactly match firmware inference.

The firmware feeds the centre detector a 155×123 rectangle centered on the OBB
output, resized-with-pad to 224×224.  This script pre-computes those crops for
every source image (PXL photos + board captures) so training sees the same
pixel geometry as inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedded_gauge_reading_tinyml.board_pipeline import (
    _quantize_input,
    decode_obb_crop_box,
)

ARTIFACTS = PROJECT_ROOT / "artifacts" / "deployment"
OBB_PATH = ARTIFACTS / "prod_model_v0.3_obb_int8" / "model_int8.tflite"

INPUT_SIZE = 224
TC_W = 155
TC_H = 123


def _resize_with_pad_geometry(crop_w: int, crop_h: int):
    scale = min(INPUT_SIZE / crop_w, INPUT_SIZE / crop_h)
    resized_w = crop_w * scale
    resized_h = crop_h * scale
    pad_x = (INPUT_SIZE - resized_w) * 0.5
    pad_y = (INPUT_SIZE - resized_h) * 0.5
    return scale, pad_x, pad_y


def _norm_to_cd_crop(
    cx_full_norm: float,
    cy_full_norm: float,
    cd_x: int,
    cd_y: int,
) -> tuple[float, float]:
    """Convert full-frame normalized label -> CD-crop normalized label in padded 224x224 space."""
    true_cx = cx_full_norm * INPUT_SIZE
    true_cy = cy_full_norm * INPUT_SIZE
    scale, pad_x, pad_y = _resize_with_pad_geometry(TC_W, TC_H)
    padded_cx = (true_cx - cd_x) * scale + pad_x
    padded_cy = (true_cy - cd_y) * scale + pad_y
    return padded_cx / INPUT_SIZE, padded_cy / INPUT_SIZE


def _compute_cd_crop(
    frame: np.ndarray,
    obb_params: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Extract CD crop (155×123) centered on OBB, resize-with-pad to 224×224."""
    obb_cx = float(obb_params[0]) * INPUT_SIZE
    obb_cy = float(obb_params[1]) * INPUT_SIZE
    cd_x = int(round(obb_cx - TC_W / 2.0))
    cd_y = int(round(obb_cy - TC_H / 2.0))
    cd_x = max(0, min(cd_x, INPUT_SIZE - TC_W))
    cd_y = max(0, min(cd_y, INPUT_SIZE - TC_H))
    cd_crop = frame[cd_y:cd_y + TC_H, cd_x:cd_x + TC_W]

    scale, pad_x, pad_y = _resize_with_pad_geometry(TC_W, TC_H)
    resized_w = int(round(TC_W * scale))
    resized_h = int(round(TC_H * scale))
    cd_resized = cv2.resize(cd_crop, (resized_w, resized_h),
                            interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
    x_off = int(round(pad_x))
    y_off = int(round(pad_y))
    canvas[y_off:y_off + resized_h, x_off:x_off + resized_w] = cd_resized
    return canvas, cd_x, cd_y


def main() -> None:
    print(f"OBB:  {OBB_PATH}")
    obb_interp = tf.lite.Interpreter(model_path=str(OBB_PATH))
    obb_interp.allocate_tensors()

    PXL_DIR = PROJECT_ROOT / "data" / "preprocessed_crops"
    OBB_DIR = PROJECT_ROOT / "data" / "obb_board_crops"
    OUT_DIR = PROJECT_ROOT / "data" / "center_training_crops"
    OUT_IMAGES = OUT_DIR / "images"
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    entries = []
    skipped_obb = 0

    # ---- PXL photos ----
    with open(PXL_DIR / "metadata.json") as f:
        pxl_all = json.load(f)
    pxl_entries = [e for e in pxl_all if e["image_path"].startswith("images/PXL_")]
    print(f"PXL photos: {len(pxl_entries)}")

    for e in pxl_entries:
        src_path = PXL_DIR / e["image_path"]
        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            continue
        frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Run OBB
        batch = (frame.astype(np.float32) / 255.0)[None, ...]
        inp = obb_interp.get_input_details()[0]
        out = obb_interp.get_output_details()[0]
        q_batch = _quantize_input(batch, inp)
        obb_interp.set_tensor(int(inp["index"]), q_batch)
        obb_interp.invoke()
        q_out = obb_interp.get_tensor(int(out["index"]))[0]
        scale = float(out["quantization"][0])
        zp = int(out["quantization"][1])
        obb_params = scale * (np.asarray(q_out, dtype=np.float32) - zp)
        obb_params = obb_params.reshape(-1)

        dec = decode_obb_crop_box(
            obb_params.astype(np.float32),
            source_width=INPUT_SIZE,
            source_height=INPUT_SIZE,
        )
        if not dec.accepted:
            continue

        canvas, cd_x, cd_y = _compute_cd_crop(frame, obb_params)

        # Convert full-frame label -> CD-crop-relative padded label
        cx_norm, cy_norm = _norm_to_cd_crop(
            float(e["center_x_norm"]),
            float(e["center_y_norm"]),
            cd_x, cd_y,
        )

        # Save
        fname = e["image_path"].split("/")[-1]
        out_path = OUT_IMAGES / f"cd_{fname}"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        entries.append({
            "image_path": f"images/cd_{fname}",
            "center_x_norm": float(cx_norm),
            "center_y_norm": float(cy_norm),
            "split": e["split"],
            "source": "pxl",
        })

    # ---- OBB board crops with rim pseudo-labels ----
    # Uses the firmware's C-based rim estimator for accurate labels
    # (see generate_obb_board_crops.py which calls rim_estimator.so).
    with open(OBB_DIR / "metadata.json") as f:
        board_raw = json.load(f)

    board_entries = []
    board_repaired = 0
    for e in board_raw:
        crop_box = e.get("crop_box")
        if not crop_box or len(crop_box) < 4:
            continue
        board_entries.append(e)

    print(f"Board entries: {len(board_entries)}, repaired: {board_repaired}")

    for e in board_entries:
        # The OBB board crops are stored in obb_board_crops/images/
        # The crop_box is the OBB crop region in the full frame
        crop_box = e.get("crop_box")
        if not crop_box or len(crop_box) < 4:
            skipped_obb += 1
            continue

        cx1, cy1, cx2, cy2 = crop_box[:4]
        obb_center_x = (cx1 + cx2) / 2.0
        obb_center_y = (cy1 + cy2) / 2.0

        # True centre from pseudo-label (crop-relative -> full-frame)
        full_cx = cx1 + float(e["center_x_norm"]) * (cx2 - cx1)
        full_cy = cy1 + float(e["center_y_norm"]) * (cy2 - cy1)

        # Load source image (the original capture that produced this OBB crop)
        # The source is from preprocessed_crops for newer captures, or from source_path
        src_path = None
        if "source_path" in e:
            candidate = PROJECT_ROOT / e["source_path"]
            if candidate.exists():
                src_path = candidate
        if src_path is None:
            # Try preprocessed_crops
            candidate = PXL_DIR / e["image_path"]
            if candidate.exists():
                src_path = candidate
        if src_path is None:
            skipped_obb += 1
            continue

        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            skipped_obb += 1
            continue
        frame = cv2.resize(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA,
        )

        # Use OBB centre from crop_box (not re-running OBB)
        cd_x = int(round(obb_center_x - TC_W / 2.0))
        cd_y = int(round(obb_center_y - TC_H / 2.0))
        cd_x = max(0, min(cd_x, INPUT_SIZE - TC_W))
        cd_y = max(0, min(cd_y, INPUT_SIZE - TC_H))

        cd_crop = frame[cd_y:cd_y + TC_H, cd_x:cd_x + TC_W]
        scale, pad_x, pad_y = _resize_with_pad_geometry(TC_W, TC_H)
        resized_w = int(round(TC_W * scale))
        resized_h = int(round(TC_H * scale))
        cd_resized = cv2.resize(cd_crop, (resized_w, resized_h),
                                interpolation=cv2.INTER_LINEAR)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        x_off = int(round(pad_x))
        y_off = int(round(pad_y))
        canvas[y_off:y_off + resized_h, x_off:x_off + resized_w] = cd_resized

        # Convert label
        scale_g, pad_x_g, pad_y_g = _resize_with_pad_geometry(TC_W, TC_H)
        padded_cx = (full_cx - cd_x) * scale_g + pad_x_g
        padded_cy = (full_cy - cd_y) * scale_g + pad_y_g
        cx_norm = padded_cx / INPUT_SIZE
        cy_norm = padded_cy / INPUT_SIZE

        fname = e["image_path"].split("/")[-1]
        out_path = OUT_IMAGES / f"cd_{fname}"
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        entries.append({
            "image_path": f"images/cd_{fname}",
            "center_x_norm": float(cx_norm),
            "center_y_norm": float(cy_norm),
            "split": "train",
            "source": "board",
        })

    # Save metadata
    meta_path = OUT_DIR / "metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nSaved {len(entries)} entries, skipped {skipped_obb} board entries")
    print(f"Meta: {meta_path}")

    # Stats
    np_cx = np.array([e["center_x_norm"] for e in entries])
    np_cy = np.array([e["center_y_norm"] for e in entries])
    print(f"cx_norm: mean={np_cx.mean():.4f} std={np_cx.std():.4f} "
          f"range=[{np_cx.min():.4f},{np_cx.max():.4f}]")
    print(f"cy_norm: mean={np_cy.mean():.4f} std={np_cy.std():.4f} "
          f"range=[{np_cy.min():.4f},{np_cy.max():.4f}]")


if __name__ == "__main__":
    main()
