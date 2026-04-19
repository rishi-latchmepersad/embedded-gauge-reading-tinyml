"""Diagnostic: visualise Hough detections with the new inner-annulus mask.

Shows which lines survive after restricting to the inner 15-72% annulus
with the subdial region suppressed.

Usage (from ml/):
    poetry run python -u scripts/debug_classical_cv.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    detect_needle_unit_vector,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry

OUT_DIR = PROJECT_ROOT / "artifacts" / "debug_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ROOT = PROJECT_ROOT.parent
IMAGES = [
    ("capture_0073.png", 46.0),
    ("capture_2026-04-03_08-20-49.png", 45.0),
    ("capture_p5c.png", 5.0),
    ("capture_p20c_preview.png", 20.0),
    ("capture_0007.png", 18.0),
    ("capture_0075.png", 19.0),
    ("capture_2026-04-03_13-48-34.png", 30.0),
    ("capture_2026-04-03_15-46-04.png", 19.0),
    ("capture_p30c.png", 30.0),
    ("capture_m18c.png", -18.0),
    # Good ones as regression check
    ("capture_m30c_preview.png", -30.0),
    ("capture_0c_preview.png", 0.0),
    ("capture_0008.png", 22.0),
    ("capture_p50c_preview.png", 50.0),
]

spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]


def draw_needle_arrow(img, cx, cy, radius, dx, dy, color, label):
    tip_x = int(round(cx + radius * 0.85 * dx))
    tip_y = int(round(cy + radius * 0.85 * dy))
    cv2.arrowedLine(img, (int(cx), int(cy)), (tip_x, tip_y), color, 2, tipLength=0.15)
    cv2.putText(img, label, (tip_x + 3, tip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def draw_sweep_arc(img, cx, cy, radius, spec, color=(0, 200, 0)):
    for frac in np.linspace(0, 1, 540):
        angle = spec.min_angle_rad + frac * spec.sweep_rad
        x = int(round(cx + radius * math.cos(angle)))
        y = int(round(cy + radius * math.sin(angle)))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 1, color, -1)


results = []

for fname, true_val in IMAGES:
    img_path = REPO_ROOT / "captured_images" / fname
    if not img_path.exists():
        print(f"SKIP {fname} (not found)")
        continue

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"SKIP {fname} (unreadable)")
        continue

    h, w = img.shape[:2]
    vis = img.copy()

    estimated = estimate_dial_geometry(img)
    if estimated is not None:
        (cx, cy), radius = estimated
        geom_color = (0, 255, 0)
    else:
        cx, cy = 0.5 * w, 0.5 * h
        radius = 0.45 * min(h, w)
        geom_color = (0, 120, 255)

    cv2.circle(vis, (int(cx), int(cy)), int(radius), geom_color, 1)
    cv2.circle(vis, (int(cx), int(cy)), int(0.75 * radius), (100, 100, 255), 1)
    cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    # Subdial suppression zone rectangle
    sub_x1 = int(round(cx - 0.35 * radius))
    sub_x2 = int(round(cx + 0.35 * radius))
    sub_y1 = int(round(cy + 0.10 * radius))
    sub_y2 = int(round(cy + 0.58 * radius))
    cv2.rectangle(vis, (sub_x1, sub_y1), (sub_x2, sub_y2), (0, 80, 220), 1)

    draw_sweep_arc(vis, cx, cy, radius * 0.90, spec)

    det = detect_needle_unit_vector(img, center_xy=(cx, cy),
                                    dial_radius_px=radius, gauge_spec=spec)
    if det is not None:
        pred_val = needle_vector_to_value(det.unit_dx, det.unit_dy, spec)
        err = abs(pred_val - true_val)
        color = (0, 220, 0) if err < 5.0 else (0, 0, 255)
        draw_needle_arrow(vis, cx, cy, radius, det.unit_dx, det.unit_dy,
                          color, f"{pred_val:.1f}C err={err:.1f}")
        status = f"pred={pred_val:.1f} err={err:.1f}"
    else:
        cv2.putText(vis, "NO DET", (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        status = "NO DET"

    cv2.putText(vis, f"true={true_val}C  {status}", (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    out_path = OUT_DIR / f"v2_{fname}"
    cv2.imwrite(str(out_path), vis)
    results.append((fname, true_val, status))
    print(f"{fname:45s}  true={true_val:6.1f}  {status}")

print(f"\nDone. Open {OUT_DIR}")
