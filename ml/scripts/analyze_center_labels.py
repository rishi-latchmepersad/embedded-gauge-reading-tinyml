"""Analyze the validated manual annotations against the known rim-vote centre (108,108)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = PROJECT_ROOT / "data" / "center_training_manual"
EXPECTED_SIZE = 224

# The rim-vote centre we trust (from live board runs)
RIM_CX = 108
RIM_CY = 108

REJECT_THRESHOLD_PX = 30  # reject labels > 30px from rim centre


def main() -> None:
    meta_path = MANUAL_DIR / "metadata_cleaned.json"
    with open(meta_path) as f:
        entries = json.load(f)

    cx_all = np.array([float(e["center_x"]) * EXPECTED_SIZE for e in entries])
    cy_all = np.array([float(e["center_y"]) * EXPECTED_SIZE for e in entries])

    errors = np.sqrt((cx_all - RIM_CX)**2 + (cy_all - RIM_CY)**2)

    print(f"Total valid entries: {len(entries)}")
    print(f"Rim-vote centre: ({RIM_CX}, {RIM_CY})")
    print(f"\nDistance from rim-vote centre:")
    print(f"  mean: {errors.mean():.1f} px")
    print(f"  std:  {errors.std():.1f} px")
    print(f"  min:  {errors.min():.1f} px")
    print(f"  max:  {errors.max():.1f} px")
    print(f"  median: {np.median(errors):.1f} px")
    print(f"\nWithin 10px: {(errors <= 10).sum()} ({(errors <= 10).mean()*100:.1f}%)")
    print(f"Within 20px: {(errors <= 20).sum()} ({(errors <= 20).mean()*100:.1f}%)")

    # Flag outliers
    outliers = errors > REJECT_THRESHOLD_PX
    n_outliers = outliers.sum()
    if n_outliers > 0:
        print(f"\nOutliers (> {REJECT_THRESHOLD_PX}px from rim centre): {n_outliers}")
        outlier_indices = np.where(outliers)[0]
        for idx in outlier_indices[:20]:
            e = entries[idx]
            cx_px = float(e["center_x"]) * EXPECTED_SIZE
            cy_px = float(e["center_y"]) * EXPECTED_SIZE
            err = errors[idx]
            print(f"  {e['image']}: ({cx_px:.1f}, {cy_px:.1f}) dist={err:.1f}px")

    # Create cleaned set rejecting outliers + the known bad label
    good_mask = errors <= REJECT_THRESHOLD_PX
    good_entries = [e for i, e in enumerate(entries) if good_mask[i]]
    print(f"\nRejected {len(entries) - len(good_entries)} entries with distance > {REJECT_THRESHOLD_PX}px")
    print(f"Kept: {len(good_entries)} entries")

    out_path = MANUAL_DIR / "metadata_good.json"
    with open(out_path, "w") as f:
        json.dump(good_entries, f, indent=2)
    print(f"Saved: {out_path}")

    # Print stats of good set
    gc_x = np.array([float(e["center_x"]) * EXPECTED_SIZE for e in good_entries])
    gc_y = np.array([float(e["center_y"]) * EXPECTED_SIZE for e in good_entries])
    print(f"\nGood set centre stats (px):")
    print(f"  cx: mean={gc_x.mean():.1f} std={gc_x.std():.1f}")
    print(f"  cy: mean={gc_y.mean():.1f} std={gc_y.std():.1f}")


if __name__ == "__main__":
    main()
