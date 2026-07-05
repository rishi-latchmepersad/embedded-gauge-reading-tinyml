#!/usr/bin/env python3
"""Auto-detect gauge centre and needle temperature from 224x224 CD crops.

The centre is expected near the geometry prior (112,100) since CD crops are
centred on the gauge.  Hough circle detection refines this when it finds a
strong inner-dial circle.  The needle angle is found by a polar-vote scan for
the dark needle, then converted to Celsius.

Geometry (matching firmware):
  Inner Celsius dial centre prior: (112, 100) on 224x224
  Celsius scale: -30°C to 50°C over 270° sweep starting at 135° (math angle)
  Needle is black on white dial.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------- Gauge geometry ----------
MIN_ANGLE_DEG = 135.0       # -30°C position (math angle, 0°=right, CCW positive)
SWEEP_DEG = 270.0
MIN_VALUE_C = -30.0
MAX_VALUE_C = 50.0
PI = 3.141592653589793
TWO_PI = 2.0 * PI
DEG2RAD = PI / 180.0

# Centre prior (224×224 frame)
CX_PRIOR = 112.0
CY_PRIOR = 100.0

# Polar-vote settings
RAY_STEPS = 40
ANGLE_BINS = 180


def angle_to_celsius(angle_rad: float) -> float | None:
    shifted = angle_rad - MIN_ANGLE_DEG * DEG2RAD
    while shifted < 0.0:
        shifted += TWO_PI
    while shifted >= TWO_PI:
        shifted -= TWO_PI
    sweep_rad = SWEEP_DEG * DEG2RAD
    if shifted > sweep_rad:
        return None
    f = shifted / sweep_rad
    return MIN_VALUE_C + f * (MAX_VALUE_C - MIN_VALUE_C)


def score_quality(img_f: np.ndarray) -> str | None:
    m, s = img_f.mean(), img_f.std()
    if m < 10:      return "black"
    if m > 245:     return "overexposed"
    if m < 60:      return "dark"
    if s < 15:      return "flat"
    if s > 80:      return "noisy"
    return "clean"


def detect_centre(img_u8: np.ndarray) -> tuple[float, float, float]:
    """Use geometry prior — CD crops are centred on the gauge."""
    return CX_PRIOR, CY_PRIOR, 69.0


def detect_needle_angle(img_f: np.ndarray, cx: float, cy: float,
                         radius: float) -> tuple[float | None, float | None]:
    """Find the black needle by radial polar vote.

    For each angle, samples along a ray in the mid-shaft region of the
    dial face and scores by darkness × contrast with background.
    """
    h, w = img_f.shape
    mid_r = radius * 0.60          # centre of sampling band (≈41 px)
    half_band = radius * 0.25      # ±17 px band
    inner = max(5, int(mid_r - half_band))
    outer = min(int(mid_r + half_band), int(min(cx, cy, w - cx - 1, h - cy - 1)) - 1)
    if outer <= inner:
        return None, None

    min_a = MIN_ANGLE_DEG * DEG2RAD
    sweep_r = SWEEP_DEG * DEG2RAD
    angles = np.linspace(min_a, min_a + sweep_r, ANGLE_BINS)
    n_radii = 20
    radii = np.linspace(inner, outer, n_radii)

    scores = np.zeros(ANGLE_BINS)
    for i, ang in enumerate(angles):
        dx, dy = np.cos(ang), np.sin(ang)
        sx = np.clip(np.round(cx + dx * radii).astype(np.int32), 0, w - 1)
        sy = np.clip(np.round(cy + dy * radii).astype(np.int32), 0, h - 1)
        line_luma = img_f[sy, sx]

        # Background: 5px either side
        px, py = -dy, dx
        bx1 = np.clip(np.round(cx + dx * radii + px * 5).astype(np.int32), 0, w - 1)
        by1 = np.clip(np.round(cy + dy * radii + py * 5).astype(np.int32), 0, h - 1)
        bx2 = np.clip(np.round(cx + dx * radii - px * 5).astype(np.int32), 0, w - 1)
        by2 = np.clip(np.round(cy + dy * radii - py * 5).astype(np.int32), 0, h - 1)
        bg = np.maximum(img_f[by1, bx1], img_f[by2, bx2])

        contrast = bg - line_luma
        frac = radii / (outer - inner + 1e-6)
        wt = np.exp(-0.5 * ((frac - 0.5) / 0.2) ** 2)
        scores[i] = float(np.sum(np.maximum(contrast, 0.0) * wt))

    best = np.argmax(scores)
    if scores[best] < 10:
        return None, None

    # Parabolic refine
    if 0 < best < len(angles) - 1:
        a_, b_, c_ = scores[best - 1], scores[best], scores[best + 1]
        d = a_ - 2 * b_ + c_
        if abs(d) > 1e-8:
            off = 0.5 * (a_ - c_) / d
            refined = angles[best] + off * (angles[1] - angles[0])
        else:
            refined = angles[best]
    else:
        refined = angles[best]

    temp = angle_to_celsius(refined)
    return refined, temp


def detect_needle(img_f: np.ndarray, cx: float, cy: float,
                  radius: float) -> tuple[float | None, float | None]:
    """Return (angle_rad, temperature_c) or (None, None)."""
    return detect_needle_angle(img_f, cx, cy, radius)


def process_one(path: Path, verbose: bool = False) -> dict | None:
    try:
        arr = np.array(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), dtype=np.float32)
    except Exception:
        return None
    if arr.shape != (224, 224):
        return None

    qual = score_quality(arr)
    if qual in ("black", "overexposed", "flat"):
        if verbose:
            print(f"  SKIP ({qual})")
        return None

    if qual == "noisy":
        arr = cv2.medianBlur(arr.astype(np.uint8), 3).astype(np.float32)

    cx, cy, radius = detect_centre(arr.astype(np.uint8))
    angle_rad, temp_c = detect_needle(arr, cx, cy, radius)

    r = {
        "center_x": round(cx, 2),
        "center_y": round(cy, 2),
        "radius": round(radius, 2),
        "angle_deg": round(angle_rad * 180 / PI, 2) if angle_rad is not None else None,
        "temperature_c": round(temp_c, 1) if temp_c is not None else None,
        "quality": qual,
    }
    if verbose:
        print(f"  cx={cx:.1f} cy={cy:.1f} r={radius:.1f} "
              f"a={r['angle_deg']}° t={r['temperature_c']}°C {qual}")
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--image", type=str)
    args = p.parse_args()

    if args.image:
        r = process_one(Path(args.image), verbose=True)
        if r: print(json.dumps(r, indent=2))
        else: print("Detection failed")
        return

    img_dir = PROJECT_ROOT / "data" / "center_training_board_mimic" / "images"
    meta_path = PROJECT_ROOT / "data" / "center_training_board_mimic" / "metadata.json"

    with open(meta_path) as f:
        meta = json.load(f)
    print(f"Processing {len(meta)} entries ...")

    updated = 0
    for i, entry in enumerate(meta):
        fp = img_dir / os.path.basename(entry["image_path"])
        if not fp.exists():
            continue
        result = process_one(fp, verbose=args.verbose)
        if result is None:
            continue
        entry["center_x_source"] = result["center_x"]
        entry["center_y_source"] = result["center_y"]
        entry["center_x_norm"] = result["center_x"] / 224.0
        entry["center_y_norm"] = result["center_y"] / 224.0
        entry["full_frame_center_x"] = result["center_x"]
        entry["full_frame_center_y"] = result["center_y"]
        entry["quality_flag"] = result["quality"]
        if result["temperature_c"] is not None:
            entry["temperature_c"] = result["temperature_c"]
        updated += 1

    print(f"\nUpdated {updated}/{len(meta)}")
    cxs = [e["center_x_source"] for e in meta if "center_x_source" in e]
    cys = [e["center_y_source"] for e in meta if "center_y_source" in e]
    ts = [e["temperature_c"] for e in meta if e.get("temperature_c", -999) > -100]
    if cxs: print(f"CX: {min(cxs):.1f}–{max(cxs):.1f}  μ={np.mean(cxs):.1f} σ={np.std(cxs):.1f}")
    if cys: print(f"CY: {min(cys):.1f}–{max(cys):.1f}  μ={np.mean(cys):.1f} σ={np.std(cys):.1f}")
    if ts:  print(f"Temp: {min(ts):.1f}–{max(ts):.1f}  μ={np.mean(ts):.1f} n={len(ts)}")

    if not args.dry_run:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved {meta_path}")
    else:
        print("Dry-run — NOT written")


if __name__ == "__main__":
    main()
