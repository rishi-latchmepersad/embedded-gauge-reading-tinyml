"""Analyze the hard cases manifest and evaluation results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def analyze_manifest():
    with open("data/hard_cases_plus_board30_valid_with_new6.csv") as f:
        rows = list(csv.DictReader(f))

    print(f"Total samples: {len(rows)}")
    vals = [float(r["value"]) for r in rows]
    print(f"Value range: {min(vals):.0f} to {max(vals):.0f}")
    print(f"Unique values: {sorted(set(vals))}")

    apr22 = [r for r in rows if "2026-04-22" in r["image_path"]]
    print(f"\n2026-04-22 captures: {len(apr22)}")
    for r in apr22:
        name = r["image_path"].rsplit("/", 1)[-1]
        print(f"  {name} = {r['value']}C")


def analyze_classical():
    rows = list(csv.DictReader(open("artifacts/hard_cases_predictions_classical.csv")))
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

    print(f"\nClassical Baseline Results:")
    print(f"Total samples: {len(rows)}")
    print(f"MAE: {np.mean([e[0] for e in errs]):.2f}")
    print(f"Max error: {max(errs)[0]:.2f}")
    print(f"Cases > 15C: {sum(1 for e in errs if e[0] > 15)}")
    print(f"Cases > 5C: {sum(1 for e in errs if e[0] > 5)}")
    print("\nWorst failures:")
    for e in errs[:10]:
        name = e[4].rsplit("/", 1)[-1]
        print(
            f"  err={e[0]:6.1f}  true={e[1]:6.1f}  pred={e[2]:6.1f}  conf={e[3]:.1f}  {name}"
        )


def analyze_tflite():
    rows = list(csv.DictReader(open("artifacts/hard_cases_predictions_tflite.csv")))
    errs = [
        (
            abs(float(r["abs_error"])),
            float(r["true_value"]),
            float(r["predicted_value"]),
            r["image_path"],
        )
        for r in rows
    ]
    errs.sort(key=lambda x: -x[0])

    print(f"\nTFLite Model Results:")
    print(f"Total samples: {len(rows)}")
    print(f"MAE: {np.mean([e[0] for e in errs]):.2f}")
    print(f"Max error: {max(errs)[0]:.2f}")
    print(f"Cases > 5C: {sum(1 for e in errs if e[0] > 5)}")
    print("\nWorst failures:")
    for e in errs[:10]:
        name = e[3].rsplit("/", 1)[-1]
        print(f"  err={e[0]:6.1f}  true={e[1]:6.1f}  pred={e[2]:7.2f}  {name}")


if __name__ == "__main__":
    analyze_manifest()
    analyze_classical()
