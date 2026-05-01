"""Analyze classical baseline and CNN evaluation results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def analyze_classical(csv_path: Path) -> None:
    """Print summary and worst failures for classical baseline."""
    rows = list(csv.DictReader(open(csv_path)))
    errs = [
        (
            abs(float(r["abs_error"])),
            float(r["true_value"]),
            float(r["predicted_value"]),
            float(r["confidence"]),
            r["image_path"],
        )
        for r in rows
    ]
    errs.sort(key=lambda x: -x[0])

    print(f"Total samples: {len(rows)}")
    print(f"MAE: {np.mean([e[0] for e in errs]):.2f}")
    print(f"Max error: {max(errs)[0]:.2f}")
    print(f"Cases > 15C: {sum(1 for e in errs if e[0] > 15)}")
    print(f"Cases > 5C: {sum(1 for e in errs if e[0] > 5)}")
    print()
    print("Worst failures:")
    for e in errs[:15]:
        name = e[4].rsplit("/", 1)[-1]
        print(
            f"  err={e[0]:6.1f}  true={e[1]:6.1f}  pred={e[2]:6.1f}  conf={e[3]:.1f}  {name}"
        )


if __name__ == "__main__":
    analyze_classical(Path("artifacts/hard_cases_predictions_classical.csv"))
