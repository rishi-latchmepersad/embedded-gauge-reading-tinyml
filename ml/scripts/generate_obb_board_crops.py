#!/usr/bin/env python3
"""Generate OBB-cropped board capture training data with rim-based pseudo-labels.

For each board capture, runs the OBB model to detect the gauge crop region,
runs the rim-based centre estimator on the full frame, and converts the centre
to crop-relative coordinates.  Saves the OBB crop as a 224x224 PNG with the
normalised centre annotation.

The result is a dataset that matches what the centre-detector CNN actually
sees during live board inference.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedded_gauge_reading_tinyml.board_pipeline import (
    load_model_session,
    decode_obb_crop_box,
    _run_session,
)
from embedded_gauge_reading_tinyml.rim_estimator.c_rim_estimator import (
    find_rim_center,
)

OBB_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "deployment"
    / "prod_model_v0.3_obb_int8"
    / "model_int8.tflite"
)
BOARD_CAPTURES_DIR = PROJECT_ROOT / "data" / "captured_images"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed_crops"
METADATA_PATH = PREPROCESSED_DIR / "metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "obb_board_crops"
INPUT_SIZE = 224


def _load_and_preprocess(path: Path) -> np.ndarray:
    """Load a PNG/JPG, resize to 224x224, return uint8 RGB array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return img


def main() -> None:
    os.makedirs(str(OUTPUT_DIR / "images"), exist_ok=True)

    # Load metadata to find board captures
    with open(METADATA_PATH) as f:
        entries = json.load(f)

    board_entries = [e for e in entries if e["image_path"].startswith("images/capture_")]
    print(f"Board captures in metadata: {len(board_entries)}")

    # Load OBB model
    obb_session = load_model_session(OBB_MODEL_PATH, "tflite")
    print(f"OBB model loaded: {OBB_MODEL_PATH}")

    results: list[dict] = []
    skipped = 0

    for i, entry in enumerate(board_entries):
        # Resolve the source image path
        source_path = PROJECT_ROOT / "data" / "preprocessed_crops" / entry["image_path"]
        if not source_path.exists():
            print(f"  [{i+1}] SKIP (not found): {entry['image_path']}")
            skipped += 1
            continue

        # Load the 224x224 preprocessed image
        img = _load_and_preprocess(source_path)

        # Run OBB model on the full frame
        full_batch = (img.astype(np.float32) / 255.0)[None, ...]
        obb_output, _ = _run_session(
            obb_session, full_batch, preferred_output_keys=("obb_params",)
        )
        obb_params = np.asarray(obb_output, dtype=np.float32).reshape(-1)

        # Decode the crop box
        decision = decode_obb_crop_box(
            obb_params,
            source_width=INPUT_SIZE,
            source_height=INPUT_SIZE,
            input_size=INPUT_SIZE,
        )
        if not decision.accepted:
            print(
                f"  [{i+1}] SKIP (OBB rejected): {entry['image_path']} "
                f"reason={decision.fallback_reason}"
            )
            skipped += 1
            continue

        # Extract crop box in integer pixel coords (full frame)
        x1, y1, x2, y2 = decision.crop_box_xyxy
        x1_i, y1_i = max(0, int(round(x1))), max(0, int(round(y1)))
        x2_i, y2_i = min(INPUT_SIZE, int(round(x2))), min(INPUT_SIZE, int(round(y2)))

        # Crop and resize to 224x224
        crop_img = img[y1_i:y2_i, x1_i:x2_i, :]
        if crop_img.size == 0:
            print(f"  [{i+1}] SKIP (empty crop): {entry['image_path']}")
            skipped += 1
            continue
        crop_resized = cv2.resize(crop_img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

        # Compute dial radius for rim estimator — use a fixed gauge-design
        # ratio instead of the OBB box width.  The OBB width varies with
        # lighting and would shift the rim-search annulus, producing noisy
        # pseudo-labels (std 18-25 px).  The baseline baseline confirms
        # 0.3076 × INPUT_SIZE = 68.9 px is correct for 224×224 frames.
        dial_radius_px = 0.3076 * INPUT_SIZE

        # Run the firmware's rim-centre estimator on the full frame via
        # the C shared library.  The firmware expects YUV422 packed bytes;
        # we extract luma and stuff it into the Y channel of a dummy
        # YUV422 buffer (U/V = 128, neutral) since ReadLuma only reads
        # the Y bytes at position y*width*2 + x*2.
        luma = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        yuv = np.empty((INPUT_SIZE, INPUT_SIZE, 2), dtype=np.uint8)
        yuv[:, :, 0] = luma           # Y channel
        yuv[:, :, 1] = 128            # U / V neutral
        rim_cx, rim_cy, detected = find_rim_center(yuv, dial_radius_px)

        if not detected:
            # Fall back to crop centre
            rim_cx = (x1 + x2) / 2.0
            rim_cy = (y1 + y2) / 2.0
            print(f"  [{i+1}] Rim not detected, using crop centre: ({rim_cx:.1f}, {rim_cy:.1f})")

        # Convert rim centre from full-frame coords to crop-relative normalised coords
        cx_norm = (rim_cx - x1) / (x2 - x1)
        cy_norm = (rim_cy - y1) / (y2 - y1)

        # Clamp to [0, 1]
        cx_norm = float(np.clip(cx_norm, 0.0, 1.0))
        cy_norm = float(np.clip(cy_norm, 0.0, 1.0))

        # Save the crop image
        out_name = Path(entry["image_path"]).name
        out_path = OUTPUT_DIR / "images" / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))

        results.append({
            "image_path": f"images/{out_name}",
            "source_path": entry["source_path"],
            "center_x_norm": cx_norm,
            "center_y_norm": cy_norm,
            "split": entry["split"],
            "temperature_c": entry.get("temperature_c"),
            "quality": entry.get("quality", "clean"),
            "crop_box": [x1, y1, x2, y2],
            "rim_detected": detected,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}] {out_name}: crop=({x1_i},{y1_i})-({x2_i},{y2_i}) "
                f"centre=({cx_norm:.3f}, {cy_norm:.3f}) rim_detected={detected}"
            )

    # Save metadata
    out_metadata_path = OUTPUT_DIR / "metadata.json"
    with open(str(out_metadata_path), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone: {len(results)} crops saved, {skipped} skipped")
    print(f"Metadata: {out_metadata_path}")
    print(f"Images: {OUTPUT_DIR / 'images'}")


if __name__ == "__main__":
    main()
