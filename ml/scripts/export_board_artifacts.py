"""Export the calibrated scalar CNN to board-ready TFLite artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add `ml/src` to sys.path so this script works from the `ml/` directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.export import (  # noqa: E402
    ExportConfig,
    export_board_tflite_artifacts,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the board export job."""
    parser = argparse.ArgumentParser(
        description="Export a board-ready model to TFLite artifacts."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "training"
        / "scalar_full_finetune_from_best_piecewise_calibrated"
        / "model.keras",
        help="Path to the calibrated scalar Keras model.",
    )
    parser.add_argument(
        "--hard-case-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "hard_cases.csv",
        help="CSV manifest of the labeled hard-case board captures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "deployment"
        / "scalar_full_finetune_from_best_piecewise_calibrated_int8",
        help="Directory where TFLite and metadata should be written.",
    )
    parser.add_argument(
        "--deployment-kind",
        choices=["scalar", "rectifier", "obb"],
        default="scalar",
        help="Deployment flavor to export for the board.",
    )
    parser.add_argument(
        "--representative-count",
        type=int,
        default=32,
        help="How many labeled training images to use for quantization calibration.",
    )
    parser.add_argument(
        "--legacy-mobilenetv2-preprocess",
        action="store_true",
        help="Load the saved model using the legacy MobileNetV2 preprocess Lambda symbol.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the board export and print the generated artifact paths."""
    args = _parse_args()
    config = ExportConfig(
        model_path=args.model,
        output_dir=args.output_dir,
        hard_case_manifest=args.hard_case_manifest,
        deployment_kind=args.deployment_kind,
        representative_count=args.representative_count,
        legacy_mobilenetv2_preprocess=args.legacy_mobilenetv2_preprocess,
    )
    result = export_board_tflite_artifacts(config)
    print(
        "[EXPORT] Board artifacts ready: "
        f"tflite={result.tflite_path} metadata={result.metadata_path} "
        f"input={result.input_shape} output={result.output_shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
