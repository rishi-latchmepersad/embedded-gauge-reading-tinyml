#!/usr/bin/env python3
"""Prepare training data for GaugeCenterDetectorTransferV1.

Merges manual board-capture annotations with PXL photo data,
generates CD-crops matching firmware inference, and creates Gaussian
heatmap targets (32×32) for heatmap regression training.
"""

from __future__ import annotations

import csv
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
HEATMAP_SIZE = 32
GAUSSIAN_SIGMA = 1.5

OUT_DIR = PROJECT_ROOT / "data" / "heatmap_training"
OUT_IMAGES = OUT_DIR / "images"
OUT_IMAGES.mkdir(parents=True, exist_ok=True)


def _resize_with_pad_geometry(crop_w: int, crop_h: int):
    scale = min(INPUT_SIZE / crop_w, INPUT_SIZE / crop_h)
    resized_w = crop_w * scale
    resized_h = crop_h * scale
    pad_x = (INPUT_SIZE - resized_w) * 0.5
    pad_y = (INPUT_SIZE - resized_h) * 0.5
    return scale, pad_x, pad_y


def _compute_cd_crop(frame: np.ndarray, obb_params: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Extract CD crop (155×123) centered on OBB, resize-with-pad to 224×224."""
    obb_cx = float(obb_params[0]) * INPUT_SIZE
    obb_cy = float(obb_params[1]) * INPUT_SIZE
    cd_x = int(round(obb_cx - TC_W / 2.0))
    cd_y = int(round(obb_cy - TC_H / 2.0))
    cd_x = max(0, min(cd_x, INPUT_SIZE - TC_W))
    cd_y = max(0, min(cd_y, INPUT_SIZE - TC_H))
    cd_crop = frame[cd_y : cd_y + TC_H, cd_x : cd_x + TC_W]

    scale, pad_x, pad_y = _resize_with_pad_geometry(TC_W, TC_H)
    resized_w = int(round(TC_W * scale))
    resized_h = int(round(TC_H * scale))
    cd_resized = cv2.resize(cd_crop, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
    x_off = int(round(pad_x))
    y_off = int(round(pad_y))
    canvas[y_off : y_off + resized_h, x_off : x_off + resized_w] = cd_resized
    return canvas, cd_x, cd_y


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


def generate_gaussian_heatmap(cx_norm: float, cy_norm: float, size: int = HEATMAP_SIZE, sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (cx_norm, cy_norm)."""
    cx = cx_norm * (size - 1)
    cy = cy_norm * (size - 1)
    y = np.arange(size)
    x = np.arange(size)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    return heatmap.astype(np.float32)


def load_manual_annotations(csv_path: Path) -> dict[str, tuple[float, float]]:
    """Load manual annotations as {filename: (cx_norm, cy_norm)}."""
    annotations = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            cx = float(row["cx_norm"])
            cy = float(row["cy_norm"])
            annotations[fname] = (cx, cy)
    return annotations


def main() -> None:
    print(f"OBB: {OBB_PATH}")
    obb_interp = tf.lite.Interpreter(model_path=str(OBB_PATH))
    obb_interp.allocate_tensors()

    entries = []
    
    # ---- PXL photos (already CD-cropped) ----
    PXL_DIR = PROJECT_ROOT / "data" / "center_training_crops"
    if (PXL_DIR / "metadata.json").exists():
        with open(PXL_DIR / "metadata.json") as f:
            pxl_all = json.load(f)
        pxl_entries = [e for e in pxl_all if e.get("source") == "pxl"]
        print(f"PXL CD-crops: {len(pxl_entries)}")

        for e in pxl_entries:
            img_path = PXL_DIR / e["image_path"]
            if not img_path.exists():
                continue

            cx_norm = float(e["center_x_norm"])
            cy_norm = float(e["center_y_norm"])
            heatmap = generate_gaussian_heatmap(cx_norm, cy_norm)

            # Copy image to output
            fname = img_path.name
            out_img_path = OUT_IMAGES / fname
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(out_img_path), img)

            entries.append({
                "image_path": f"images/{fname}",
                "cx_norm": cx_norm,
                "cy_norm": cy_norm,
                "split": e.get("split", "train"),
                "source": "pxl",
            })
    else:
        print("WARNING: No PXL CD-crop data found")

    # ---- Manual board-capture annotations ----
    MANUAL_DIRS = [
        PROJECT_ROOT.parent / "tmp" / "annotate_30",
        PROJECT_ROOT.parent / "tmp" / "annotate_batch2",
    ]
    
    manual_annotations = {}
    for d in MANUAL_DIRS:
        csv_files = list(d.glob("*.csv"))
        for csv_path in csv_files:
            anns = load_manual_annotations(csv_path)
            # Only keep annotations for images that exist in this directory
            for fname, (cx, cy) in anns.items():
                img_path = d / "images" / fname
                if img_path.exists():
                    manual_annotations[fname] = {
                        "cx_norm": cx,
                        "cy_norm": cy,
                        "img_path": img_path,
                    }

    print(f"Manual board annotations: {len(manual_annotations)}")

    # Process each manual annotation
    skipped = 0
    for fname, ann in manual_annotations.items():
        img_path = ann["img_path"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            skipped += 1
            continue

        frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Resize to 224x224 for OBB (same as firmware)
        frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

        # Run OBB
        batch = (frame_resized.astype(np.float32) / 255.0)[None, ...]
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
            skipped += 1
            continue

        # Generate CD-crop
        canvas, cd_x, cd_y = _compute_cd_crop(frame_resized, obb_params)

        # Convert manual annotation (in original image coords) to CD-crop coords
        # The manual annotation is on the original image, but we resized to 224x224
        # So the normalized coords are the same in the resized image
        cx_norm = ann["cx_norm"]
        cy_norm = ann["cy_norm"]
        
        # Convert to CD-crop padded space
        cx_cd, cy_cd = _norm_to_cd_crop(cx_norm, cy_norm, cd_x, cd_y)

        # Save CD-crop
        out_fname = f"cd_manual_{fname}"
        out_img_path = OUT_IMAGES / out_fname
        cv2.imwrite(str(out_img_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        # Generate heatmap
        heatmap = generate_gaussian_heatmap(cx_cd, cy_cd)

        entries.append({
            "image_path": f"images/{out_fname}",
            "cx_norm": cx_cd,
            "cy_norm": cy_cd,
            "split": "train",
            "source": "board_manual",
        })

    print(f"Board manual entries: {len([e for e in entries if e['source'] == 'board_manual'])}, skipped: {skipped}")

    # ---- Assign train/val/test splits ----
    # Keep PXL test set separate, split the rest 80/10/10
    pxl_test = [e for e in entries if e["source"] == "pxl" and e["split"] == "test"]
    pxl_trainval = [e for e in entries if e["source"] == "pxl" and e["split"] != "test"]
    board_entries = [e for e in entries if e["source"] == "board_manual"]

    np.random.seed(42)
    np.random.shuffle(pxl_trainval)
    np.random.shuffle(board_entries)

    # Combine PXL trainval and board entries, then split
    all_trainval = pxl_trainval + board_entries
    n = len(all_trainval)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    for i, e in enumerate(all_trainval):
        if i < n_train:
            e["split"] = "train"
        elif i < n_train + n_val:
            e["split"] = "val"
        else:
            e["split"] = "test"

    # PXL test stays as test, but we need to add them back
    final_entries = all_trainval + pxl_test

    # Save metadata
    meta_path = OUT_DIR / "metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(final_entries, f, indent=2)

    # Stats
    splits = {"train": 0, "val": 0, "test": 0}
    sources = {}
    for e in final_entries:
        splits[e["split"]] += 1
        sources[e["source"]] = sources.get(e["source"], 0) + 1

    print(f"\nSaved {len(final_entries)} entries to {meta_path}")
    print(f"Splits: {splits}")
    print(f"Sources: {sources}")

    np_cx = np.array([e["cx_norm"] for e in final_entries])
    np_cy = np.array([e["cy_norm"] for e in final_entries])
    print(f"cx_norm: mean={np_cx.mean():.4f} std={np_cx.std():.4f} range=[{np_cx.min():.4f},{np_cx.max():.4f}]")
    print(f"cy_norm: mean={np_cy.mean():.4f} std={np_cy.std():.4f} range=[{np_cy.min():.4f},{np_cy.max():.4f}]")


if __name__ == "__main__":
    main()
