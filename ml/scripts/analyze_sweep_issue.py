"""Analyze the sweep restriction issue in the classical baseline.

The diagnostic shows that the sweep restriction (zeroing out bins outside
135°-315°) is causing the detector to find WRONG peaks when the true needle
angle falls outside that range. But wait - the sweep IS 135°-315° for this gauge,
so angles outside that range shouldn't be valid needle positions.

The real question: is the detector finding the wrong feature entirely?
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
    _angle_in_sweep,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry

REPO_ROOT = PROJECT_ROOT.parent
spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]

print("=" * 70)
print("SWEEP ANALYSIS")
print("=" * 70)
print(
    f"min_angle_rad={spec.min_angle_rad:.4f} ({math.degrees(spec.min_angle_rad):.1f}°)"
)
print(f"sweep_rad={spec.sweep_rad:.4f} ({math.degrees(spec.sweep_rad):.1f}°)")
print(
    f"max_angle_rad={spec.min_angle_rad + spec.sweep_rad:.4f} ({math.degrees(spec.min_angle_rad + spec.sweep_rad):.1f}°)"
)
print()

# Test _angle_in_sweep for various angles
print(
    f"{'Angle(deg)':>12s}  {'InSweep(0°)':>12s}  {'InSweep(6°)':>12s}  {'MappedTemp':>12s}"
)
print("-" * 52)
for deg in range(-180, 361, 15):
    rad = math.radians(deg)
    in0 = _angle_in_sweep(rad, spec, margin_rad=0.0)
    in6 = _angle_in_sweep(rad, spec, margin_rad=math.radians(6.0))
    dx = math.cos(rad)
    dy = math.sin(rad)
    temp = needle_vector_to_value(dx, dy, spec)
    print(f"{deg:>12.1f}  {str(in0):>12s}  {str(in6):>12s}  {temp:>12.4f}")

print()
print("=" * 70)
print("DEEP DETECTION ANALYSIS")
print("=" * 70)

# Focus on the worst cases
IMAGES = [
    ("capture_p35c_preview.png", 35.0, "Hot - predicted -9.78°C (44.8°C error)"),
    ("capture_p50c_preview.png", 50.0, "Hot - predicted 10°C (40°C error)"),
    (
        "capture_2026-04-03_08-20-49.png",
        45.0,
        "Hot - predicted -16.89°C (61.9°C error)",
    ),
    ("capture_0c_preview.png", 0.0, "Mid - predicted 18°C (18°C error)"),
    ("capture_m30c_preview.png", -30.0, "Cold - predicted -22°C (8°C error)"),
]

for fname, true_val, note in IMAGES:
    img_path = REPO_ROOT / "data" / "captured" / "images" / fname
    if not img_path.exists():
        print(f"\nSKIP {fname}")
        continue

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"\nSKIP {fname}")
        continue

    h, w = img.shape[:2]
    estimated = estimate_dial_geometry(img)
    if estimated is not None:
        (cx, cy), radius = estimated
    else:
        cx, cy = 0.5 * w, 0.5 * h
        radius = 0.45 * min(h, w)

    print(f"\n--- {fname} (true={true_val:.1f}°C) {note} ---")
    print(f"  Geometry: center=({cx:.1f}, {cy:.1f}), radius={radius:.1f}px")

    # Expected angle for this temperature
    fraction = (true_val - spec.min_value) / (spec.max_value - spec.min_value)
    expected_angle = math.degrees(spec.min_angle_rad + fraction * spec.sweep_rad)
    print(f"  Expected needle angle: {expected_angle:.1f}°")

    # With sweep restriction
    det = detect_needle_unit_vector(
        img, center_xy=(cx, cy), dial_radius_px=radius, gauge_spec=spec
    )
    if det is not None:
        det_angle = math.degrees(math.atan2(det.unit_dy, det.unit_dx))
        pred = needle_vector_to_value(det.unit_dx, det.unit_dy, spec)
        print(
            f"  WITH sweep: angle={det_angle:7.2f}°  pred={pred:7.2f}°C  "
            f"conf={det.confidence:.2f}  ratio={det.peak_ratio:.2f}"
        )
    else:
        print(f"  WITH sweep: No detection")

    # Without sweep restriction
    det2 = detect_needle_unit_vector(
        img, center_xy=(cx, cy), dial_radius_px=radius, gauge_spec=None
    )
    if det2 is not None:
        det_angle2 = math.degrees(math.atan2(det2.unit_dy, det2.unit_dx))
        pred2 = needle_vector_to_value(det2.unit_dx, det2.unit_dy, spec)
        print(
            f"  NO sweep:  angle={det_angle2:7.2f}°  pred={pred2:7.2f}°C  "
            f"conf={det2.confidence:.2f}  ratio={det2.peak_ratio:.2f}"
        )
    else:
        print(f"  NO sweep:  No detection")

    # Check: is the no-sweep angle within the sweep?
    if det2 is not None:
        rad2 = math.atan2(det2.unit_dy, det2.unit_dx)
        in_sweep = _angle_in_sweep(rad2, spec, margin_rad=math.radians(6.0))
        print(f"  No-sweep angle in sweep? {in_sweep}")
        if not in_sweep:
            print(f"  *** No-sweep angle is OUTSIDE the sweep arc! ***")
            print(f"  *** This means the detector found a non-needle feature ***")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()
print("The sweep restriction zeros out histogram bins outside 135°-315°.")
print("When the detector finds a strong peak outside this range (e.g., at 92.5°")
print("for the 50°C image), the sweep restriction forces it to pick a different,")
print("weaker peak WITHIN the sweep range - which is often wrong.")
print()
print("But the no-sweep angle (92.5°) is ALSO wrong for 50°C - the correct angle")
print("should be 315° (4:30 position). So the detector is fundamentally finding")
print("the wrong feature, and the sweep restriction just makes it worse by")
print("forcing it to pick an even wronger feature.")
print()
print("Root cause: The spoke-vote detector is picking up a strong non-needle")
print("feature (probably the subdial needle, a reflection, or a tick mark)")
print("instead of the main temperature needle.")
