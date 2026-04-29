"""Inspect quality and error rankings on the Hough-centered geometry grid."""

from __future__ import annotations

from pathlib import Path
import math

import cv2
import numpy as np

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    detect_needle_unit_vector,
    needle_detection_quality,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import estimate_dial_geometry


REPO_ROOT: Path = Path(__file__).resolve().parent
SPEC = load_gauge_specs()["littlegood_home_temp_gauge_c"]

IMAGES: list[tuple[str, float]] = [
    ("captured_images/capture_p35c_preview.png", 35.0),
    ("captured_images/capture_2026-04-03_08-20-49.png", 45.0),
]


def main() -> None:
    """Print the top candidates by error and by quality for a few hard cases."""
    center_offsets = list(range(-40, 41, 4))
    radius_values = [x / 2.0 for x in range(80, 241, 10)]
    center_index_lookup = {offset: index for index, offset in enumerate(center_offsets)}
    radius_index_lookup = {radius: index for index, radius in enumerate(radius_values)}

    for rel_path, true_value in IMAGES:
        image_path = REPO_ROOT / rel_path
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"{rel_path}: unreadable")
            continue

        estimated = estimate_dial_geometry(image_bgr)
        if estimated is None:
            print(f"{rel_path}: no geometry estimate")
            continue

        (center_x, center_y), radius_px = estimated
        rows: list[tuple[int, int, int, float, float, float, float, float, float]] = []
        for dx in center_offsets:
            for dy in center_offsets:
                for scale in radius_values:
                    detection = detect_needle_unit_vector(
                        image_bgr,
                        center_xy=(center_x + float(dx), center_y + float(dy)),
                        dial_radius_px=float(scale),
                        gauge_spec=SPEC,
                    )
                    if detection is None:
                        continue
                    predicted_value = float(
                        needle_vector_to_value(detection.unit_dx, detection.unit_dy, SPEC)
                    )
                    quality = float(needle_detection_quality(detection))
                    error = abs(predicted_value - true_value)
                    angle_deg = math.degrees(math.atan2(detection.unit_dy, detection.unit_dx))
                    rows.append(
                        (
                            center_index_lookup[dx],
                            center_index_lookup[dy],
                            radius_index_lookup[scale],
                            error,
                            quality,
                            predicted_value,
                            angle_deg,
                            center_x + float(dx),
                            center_y + float(dy),
                            float(scale),
                        )
                    )

        print(f"\n{rel_path} true={true_value:.1f} geom=({center_x:.1f},{center_y:.1f}) r={radius_px:.1f}")
        print("Top by error:")
        for _, _, _, error, quality, predicted_value, angle_deg, cx, cy, radius in sorted(rows, key=lambda item: item[3])[:12]:
            print(
                f"  err={error:6.2f} pred={predicted_value:7.2f} q={quality:10.2f} "
                f"ang={angle_deg:7.2f} center=({cx:6.1f},{cy:6.1f}) r={radius:6.1f}"
            )
        print("Top by quality:")
        for _, _, _, error, quality, predicted_value, angle_deg, cx, cy, radius in sorted(rows, key=lambda item: item[4], reverse=True)[:12]:
            print(
                f"  err={error:6.2f} pred={predicted_value:7.2f} q={quality:10.2f} "
                f"ang={angle_deg:7.2f} center=({cx:6.1f},{cy:6.1f}) r={radius:6.1f}"
            )
        stability_rows: list[tuple[float, float, float, float, float, float, float]] = []
        for index, item in enumerate(rows):
            dx_i, dy_i, r_i, error, quality, predicted_value, angle_deg, cx, cy, radius = item
            neighbors = [
                other
                for other in rows
                if max(
                    abs(dx_i - other[0]),
                    abs(dy_i - other[1]),
                    abs(r_i - other[2]),
                )
                <= 1
            ]
            if len(neighbors) < 4:
                continue
            neighbor_values = [neighbor[5] for neighbor in neighbors]
            local_std = float(np.std(np.asarray(neighbor_values, dtype=np.float32)))
            local_mean_abs = float(
                np.mean(
                    np.abs(
                        np.asarray(neighbor_values, dtype=np.float32) - predicted_value
                    )
                )
            )
            stability = math.log1p(max(quality, 0.0)) / (1.0 + local_std + 0.5 * local_mean_abs)
            stability_rows.append((stability, error, quality, predicted_value, angle_deg, cx, cy, radius))
        print("Top by stability:")
        for stability, error, quality, predicted_value, angle_deg, cx, cy, radius in sorted(stability_rows, key=lambda item: item[0], reverse=True)[:12]:
            print(
                f"  stab={stability:8.3f} err={error:6.2f} pred={predicted_value:7.2f} "
                f"q={quality:10.2f} ang={angle_deg:7.2f} center=({cx:6.1f},{cy:6.1f}) r={radius:6.1f}"
            )


if __name__ == "__main__":
    main()
