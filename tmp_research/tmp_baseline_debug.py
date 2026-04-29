from __future__ import annotations

from pathlib import Path
import math
import cv2

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    detect_needle_unit_vector,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry, _auto_geometry_candidates

spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
base = Path("../captured_images")
files = [
    "capture_m30c_preview.png",
    "capture_m19c.png",
    "capture_m18c.png",
    "capture_m10c_preview.png",
    "capture_p5c.png",
    "capture_p10c_preview.png",
]

for name in files:
    img = cv2.imread(str(base / name), cv2.IMREAD_COLOR)
    if img is None:
        print(f"{name}: unreadable")
        continue
    est = estimate_dial_geometry(img)
    if est is None:
        cx = img.shape[1] / 2.0
        cy = img.shape[0] / 2.0
        r = 0.45 * min(img.shape[:2])
    else:
        (cx, cy), r = est
    print(f"\n{name}: geom=({cx:.1f},{cy:.1f}) r={r:.1f}")
    for cand in _auto_geometry_candidates(img):
        det = detect_needle_unit_vector(
            img,
            center_xy=cand.center_xy,
            dial_radius_px=cand.dial_radius_px,
            gauge_spec=spec,
        )
        if det is None:
            print(f"  {cand.label:18s} NONE")
            continue
        angle = math.degrees(math.atan2(det.unit_dy, det.unit_dx))
        value = needle_vector_to_value(det.unit_dx, det.unit_dy, spec)
        print(
            f"  {cand.label:18s} val={value:5.1f} ang={angle:6.1f} conf={det.confidence:7.2f} ratio={det.peak_ratio:7.2f}"
        )
