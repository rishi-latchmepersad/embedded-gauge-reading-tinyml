"""Verify that conditioned training tensors exactly match runtime tensors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_student_conditioned_littlegood import (  # noqa: E402
    FULL_DATA,
    OUTPUT,
    POINT_DATA,
    crop_transform,
    full_ellipse,
    make_sample,
    predict_ellipse,
    ellipse_predictor,
)


def main() -> None:
    """Compare all saved test tensors against regenerated runtime tensors."""
    rows = json.loads((POINT_DATA / "metadata.json").read_text())["splits"]["test"]
    saved = np.load(OUTPUT / "test.npz")
    interpreter, inp, out = ellipse_predictor()
    max_input = 0.0
    max_point = 0.0
    failures = []
    for index, row in enumerate(rows):
        image = np.asarray(Image.open(FULL_DATA / "images" / "test" / f"{row['stem']}.png").convert("RGB"))
        predicted = predict_ellipse(interpreter, inp, out, image)
        fixed = np.asarray(row["ellipse"], dtype=np.float32)
        target = full_ellipse("test", row["stem"])
        fixed = np.asarray(row["ellipse"], dtype=np.float32)
        fixed_side = max(2.0 * fixed[2], 2.0 * fixed[3]) * 1.18
        points_source = fixed[:2] - fixed_side / 2.0 + np.asarray((row["center_xy_norm"], row["tip_xy_norm"]), dtype=np.float32) * fixed_side
        points_640 = target[:2] + (points_source - fixed[:2]) * target[2:] / fixed[2:]
        regenerated, _, local = make_sample(image, predicted, points_640)
        input_error = float(np.max(np.abs(regenerated - saved["inputs"][index])))
        point_error = float(np.max(np.abs(local - saved["points"][index])))
        max_input = max(max_input, input_error)
        max_point = max(max_point, point_error)
        if input_error > 1e-5 or point_error > 1e-5:
            failures.append({"index": index, "stem": row["stem"], "input_error": input_error, "point_error": point_error})
    report = {"samples": len(rows), "max_input_abs_error": max_input, "max_point_abs_error": max_point, "failures": failures[:10], "passed": not failures}
    print(json.dumps(report, indent=2))
    (OUTPUT / "parity_report.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
