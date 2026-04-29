from __future__ import annotations

from pathlib import Path
import statistics
import cv2

from embedded_gauge_reading_tinyml.baseline_classical_cv import detect_needle_unit_vector, needle_vector_to_value
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import _auto_geometry_candidates

spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
base = Path("../captured_images")
files = ["capture_m30c_preview.png", "capture_m19c.png", "capture_m10c_preview.png", "capture_p5c.png"]

for name in files:
    img = cv2.imread(str(base / name), cv2.IMREAD_COLOR)
    cands = []
    for cand in _auto_geometry_candidates(img):
        det = detect_needle_unit_vector(img, center_xy=cand.center_xy, dial_radius_px=cand.dial_radius_px, gauge_spec=spec)
        if det is None:
            continue
        val = needle_vector_to_value(det.unit_dx, det.unit_dy, spec)
        quality = det.confidence * max(det.peak_ratio - 1.0, 0.0)
        cands.append((val, quality, cand.label))
    best = max(cands, key=lambda item: item[1])
    mean_all = sum(v for v, _, _ in cands) / len(cands)
    top3 = sorted(cands, key=lambda item: item[1], reverse=True)[:3]
    mean_top3 = sum(v for v, _, _ in top3) / len(top3)
    median_all = statistics.median(v for v, _, _ in cands)
    print(f"{name}: best={best[0]:.1f} mean_all={mean_all:.1f} mean_top3={mean_top3:.1f} median_all={median_all:.1f}")
