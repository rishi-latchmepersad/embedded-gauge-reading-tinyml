"""Run the classical gauge baseline on one image and save an annotation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

# Add `ml/src` to sys.path so this script works even before `poetry install`.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.single_image_baseline import (
    SingleImageBaselineConfig,
    run_single_image_baseline,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a single-image baseline run."""
    parser = argparse.ArgumentParser(
        description="Run the classical Canny + Hough baseline on one image."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Image to analyze, for example captured_images/capture_0006.png.",
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default="littlegood_home_temp_gauge_c",
    )
    parser.add_argument(
        "--center-x",
        type=float,
        default=None,
        help="Optional dial center X coordinate in pixels.",
    )
    parser.add_argument(
        "--center-y",
        type=float,
        default=None,
        help="Optional dial center Y coordinate in pixels.",
    )
    parser.add_argument(
        "--dial-radius-px",
        type=float,
        default=None,
        help="Optional dial radius in pixels.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "single_image_baseline",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run folder name. Defaults to timestamp.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the baseline and print a compact result summary."""
    args = parse_args()

    config = SingleImageBaselineConfig(
        image_path=args.image_path,
        gauge_id=args.gauge_id,
        center_x=args.center_x,
        center_y=args.center_y,
        dial_radius_px=args.dial_radius_px,
        artifacts_dir=args.artifacts_dir,
        run_name=args.run_name,
    )

    result = run_single_image_baseline(config)

    print(f"Image: {result.image_path}")
    print(f"Gauge spec: {asdict(result.gauge_spec)}")
    print(
        f"Dial estimate: center=({result.center_xy[0]:.1f}, {result.center_xy[1]:.1f}) "
        f"radius={result.dial_radius_px:.1f}px"
    )
    if result.detection is None:
        print("Detection: none")
    else:
        print(
            "Detection: "
            f"dx={result.detection.unit_dx:.4f}, "
            f"dy={result.detection.unit_dy:.4f}, "
            f"confidence={result.detection.confidence:.2f}"
        )
    if result.predicted_value is None:
        print("Predicted value: none")
    else:
        print(f"Predicted value: {result.predicted_value:.4f}")
    print(f"Annotated preview: {result.annotated_image_path}")


if __name__ == "__main__":
    main()
