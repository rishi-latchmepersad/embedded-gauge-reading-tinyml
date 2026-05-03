"""Diagnose the angle-to-temperature mapping in the classical baseline.

The baseline predictions on hard cases show systematic bias:
- Cold values (-30°C) read ~8°C too warm
- Mid values (0-20°C) read ~18-20°C too warm
- Hot values (35-50°C) sometimes read wildly wrong (negative!)

This script checks whether the issue is in the angle detection or the mapping.
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
    _detect_needle_unit_vector_polar,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry

REPO_ROOT = PROJECT_ROOT.parent
spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]

# Print the gauge spec details
print("=" * 70)
print("GAUGE SPEC: littlegood_home_temp_gauge_c")
print("=" * 70)
print(
    f"  min_angle_rad = {spec.min_angle_rad:.4f} ({math.degrees(spec.min_angle_rad):.1f}°)"
)
print(f"  sweep_rad     = {spec.sweep_rad:.4f} ({math.degrees(spec.sweep_rad):.1f}°)")
print(
    f"  max_angle_rad = {spec.min_angle_rad + spec.sweep_rad:.4f} ({math.degrees(spec.min_angle_rad + spec.sweep_rad):.1f}°)"
)
print(f"  min_value     = {spec.min_value}")
print(f"  max_value     = {spec.max_value}")
print(f"  span          = {spec.max_value - spec.min_value}")
print()

# Test the mapping: what angle corresponds to each temperature?
print("Expected angles for key temperatures:")
print(
    f"  {'Temp (°C)':>10s}  {'Angle (rad)':>12s}  {'Angle (deg)':>12s}  {'Unit DX':>8s}  {'Unit DY':>8s}"
)
print("  " + "-" * 58)
for temp_c in [-30, -20, -10, 0, 10, 20, 30, 40, 50]:
    fraction = (temp_c - spec.min_value) / (spec.max_value - spec.min_value)
    angle_rad = spec.min_angle_rad + fraction * spec.sweep_rad
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    print(
        f"  {temp_c:>10.1f}  {angle_rad:>12.4f}  {math.degrees(angle_rad):>12.1f}  {dx:>8.4f}  {dy:>8.4f}"
    )
print()

# Test the reverse mapping: what temperature does each angle give?
print("Reverse mapping (angle → temperature):")
print(f"  {'Angle (deg)':>12s}  {'Angle (rad)':>12s}  {'Temp (°C)':>10s}")
print("  " + "-" * 40)
for angle_deg in [135, 150, 180, 210, 225, 240, 270, 300, 315]:
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    temp = needle_vector_to_value(dx, dy, spec)
    print(f"  {angle_deg:>12.1f}  {angle_rad:>12.4f}  {temp:>10.4f}")
print()

# Now test on specific images - compare spoke-vote vs polar detection
IMAGES_TO_TEST = [
    ("capture_m30c_preview.png", -30.0),
    ("capture_0c_preview.png", 0.0),
    ("capture_p20c_preview.png", 20.0),
    ("capture_p35c_preview.png", 35.0),
    ("capture_p50c_preview.png", 50.0),
    ("capture_2026-04-03_08-20-49.png", 45.0),
]

print("=" * 70)
print("DETECTION COMPARISON: Spoke-vote vs Polar method")
print("=" * 70)

for fname, true_val in IMAGES_TO_TEST:
    img_path = REPO_ROOT / "data" / "captured" / "images" / fname
    if not img_path.exists():
        print(f"\nSKIP {fname} (not found)")
        continue

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"\nSKIP {fname} (unreadable)")
        continue

    h, w = img.shape[:2]
    estimated = estimate_dial_geometry(img)

    if estimated is not None:
        (cx, cy), radius = estimated
    else:
        cx, cy = 0.5 * w, 0.5 * h
        radius = 0.45 * min(h, w)

    print(f"\n--- {fname} (true={true_val:.1f}°C) ---")
    print(f"  Geometry: center=({cx:.1f}, {cy:.1f}), radius={radius:.1f}px")

    # Method 1: Spoke-vote detector (gradient-polar)
    det1 = detect_needle_unit_vector(
        img, center_xy=(cx, cy), dial_radius_px=radius, gauge_spec=spec
    )
    if det1 is not None:
        angle1 = math.degrees(math.atan2(det1.unit_dy, det1.unit_dx))
        pred1 = needle_vector_to_value(det1.unit_dx, det1.unit_dy, spec)
        print(
            f"  [SPOKE-VOTE] angle={angle1:7.2f}°  pred={pred1:7.2f}°C  "
            f"conf={det1.confidence:.2f}  ratio={det1.peak_ratio:.2f}"
        )
    else:
        print(f"  [SPOKE-VOTE] No detection")

    # Method 2: Polar transform detector
    det2 = _detect_needle_unit_vector_polar(
        img,
        center_xy=(cx, cy),
        dial_radius_px=radius,
        angle_bounds_rad=(spec.min_angle_rad, spec.sweep_rad),
    )
    if det2 is not None:
        angle2 = math.degrees(math.atan2(det2.unit_dy, det2.unit_dx))
        pred2 = needle_vector_to_value(det2.unit_dx, det2.unit_dy, spec)
        print(
            f"  [POLAR-DET]  angle={angle2:7.2f}°  pred={pred2:7.2f}°C  "
            f"conf={det2.confidence:.2f}  ratio={det2.peak_ratio:.2f}"
        )
    else:
        print(f"  [POLAR-DET]  No detection")

    # Method 3: Spoke-vote WITHOUT gauge_spec (no sweep restriction)
    det3 = detect_needle_unit_vector(
        img, center_xy=(cx, cy), dial_radius_px=radius, gauge_spec=None
    )
    if det3 is not None:
        angle3 = math.degrees(math.atan2(det3.unit_dy, det3.unit_dx))
        pred3 = needle_vector_to_value(det3.unit_dx, det3.unit_dy, spec)
        print(
            f"  [SPOKE-NO-SWEEP] angle={angle3:7.2f}°  pred={pred3:7.2f}°C  "
            f"conf={det3.confidence:.2f}  ratio={det3.peak_ratio:.2f}"
        )
    else:
        print(f"  [SPOKE-NO-SWEEP] No detection")

print()
print("=" * 70)
print("ANGLE MAPPING SANITY CHECK")
print("=" * 70)
print()
print("The gauge spec says:")
print(
    f"  -30°C needle points at {math.degrees(spec.min_angle_rad):.1f}° (7:30 position)"
)
print(
    f"  +50°C needle points at {math.degrees(spec.min_angle_rad + spec.sweep_rad):.1f}° (4:30 position)"
)
print(f"  Sweep is {math.degrees(spec.sweep_rad):.1f}° clockwise")
print()
print("In image coordinates (y-down):")
print("  0° = 3 o'clock, 90° = 6 o'clock, 180° = 9 o'clock, 270° = 12 o'clock")
print(f"  -30°C → {math.degrees(spec.min_angle_rad):.1f}° = 7:30 position")
print(
    f"  +50°C → {math.degrees(spec.min_angle_rad + spec.sweep_rad):.1f}° = 4:30 position"
)
print()

# Check: does the sweep arc go the right way?
# At 135° (7:30), the needle points down-left
# At 315° (4:30), the needle points down-right
# That's a 180° clockwise sweep through bottom, which makes sense for a gauge
print("Visual check of the sweep arc:")
print("  135° (7:30) → needle points down-left  → -30°C (cold)")
print("  180° (9:00) → needle points straight left")
print("  225° (7:30 equivalent but through sweep) → needle points up-left")
print("  270° (12:00) → needle points straight up")
print("  315° (4:30) → needle points down-right → +50°C (hot)")
print()
print("This means the needle sweeps clockwise through the BOTTOM of the gauge.")
print("Cold = left side, Hot = right side, going through bottom.")
