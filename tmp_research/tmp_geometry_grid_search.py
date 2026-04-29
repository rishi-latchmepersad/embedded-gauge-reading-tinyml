"""Grid-search geometry for a few hard captures using the current detector."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parent
ML_SRC = REPO_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from embedded_gauge_reading_tinyml.baseline_classical_cv import (  # noqa: E402
    detect_needle_unit_vector,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs  # noqa: E402


def main() -> None:
    """Scan a coarse geometry grid and print the best temperature candidate."""
    spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
    samples: list[tuple[str, float]] = [
        ("captured_images/today_converted/capture_2026-04-09_06-41-57.png", 30.0),
        ("captured_images/capture_p35c_preview.png", 35.0),
        ("captured_images/capture_2026-04-03_08-20-49.png", 45.0),
    ]

    for rel_path, true_value in samples:
        image_path = REPO_ROOT / rel_path
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"missing {rel_path}")
            continue

        height, width = image_bgr.shape[:2]
        image_cx = 0.5 * float(width)
        image_cy = 0.5 * float(height)
        center_offsets = [-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0]
        radius_values = [x / 2.0 for x in range(80, 241, 10)]

        best: list[tuple[float, float, float, float, float, float]] = []
        for dx in center_offsets:
            for dy in center_offsets:
                center_xy = (image_cx + dx, image_cy + dy)
                for radius_px in radius_values:
                    detection = detect_needle_unit_vector(
                        image_bgr,
                        center_xy=center_xy,
                        dial_radius_px=radius_px,
                        gauge_spec=spec,
                    )
                    if detection is None:
                        continue
                    predicted = needle_vector_to_value(
                        detection.unit_dx,
                        detection.unit_dy,
                        spec,
                    )
                    error = abs(predicted - true_value)
                    angle_deg = math.degrees(math.atan2(detection.unit_dy, detection.unit_dx))
                    best.append(
                        (
                            error,
                            predicted,
                            angle_deg,
                            center_xy[0],
                            center_xy[1],
                            radius_px,
                        )
                    )

        best.sort(key=lambda item: item[0])
        print(f"\n{rel_path} true={true_value:.1f}")
        for error, predicted, angle_deg, cx, cy, radius_px in best[:12]:
            print(
                f"  err={error:6.2f} pred={predicted:7.2f} angle={angle_deg:7.2f} "
                f"center=({cx:6.1f},{cy:6.1f}) radius={radius_px:6.1f}"
            )


if __name__ == "__main__":
    main()
