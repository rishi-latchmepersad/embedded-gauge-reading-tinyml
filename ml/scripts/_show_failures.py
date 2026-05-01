"""Analyze classical baseline failures."""

from __future__ import annotations

import csv

rows = list(csv.DictReader(open("artifacts/hard_cases_predictions_classical.csv")))
for r in rows:
    e = abs(float(r["abs_error"]))
    if e > 15:
        name = r["image_path"].rsplit("/", 1)[-1]
        print(
            f"err={e:6.1f}  true={float(r['true_value']):6.1f}  "
            f"pred={float(r['predicted_value']):6.1f}  "
            f"conf={float(r['confidence']):.1f}  {name}"
        )
