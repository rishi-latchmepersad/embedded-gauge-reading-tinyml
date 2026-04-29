from __future__ import annotations

from pathlib import Path
import math
import cv2

from embedded_gauge_reading_tinyml.baseline_classical_cv import detect_needle_unit_vector, needle_vector_to_value
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry

spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
img = cv2.imread(str(Path("../captured_images/capture_m19c.png")), cv2.IMREAD_COLOR)
true_value = -19.0
est = estimate_dial_geometry(img)
if est is None:
    raise SystemExit("no geometry")
(center_x, center_y), radius = est
print(f"base center=({center_x:.1f},{center_y:.1f}) radius={radius:.1f}")
results = []
for dx in range(-40, 41, 4):
    for dy in range(-40, 41, 4):
        for scale in [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]:
            det = detect_needle_unit_vector(
                img,
                center_xy=(center_x + dx, center_y + dy),
                dial_radius_px=radius * scale,
                gauge_spec=spec,
            )
            if det is None:
                continue
            val = needle_vector_to_value(det.unit_dx, det.unit_dy, spec)
            err = abs(val - true_value)
            results.append((err, val, dx, dy, scale, det.confidence, det.peak_ratio, math.degrees(math.atan2(det.unit_dy, det.unit_dx))))
results.sort(key=lambda item: item[0])
for item in results[:30]:
    err, val, dx, dy, scale, conf, ratio, ang = item
    print(f"err={err:5.2f} val={val:5.1f} ang={ang:6.1f} dx={dx:>3d} dy={dy:>3d} scale={scale:.2f} conf={conf:7.2f} ratio={ratio:8.2f}")
