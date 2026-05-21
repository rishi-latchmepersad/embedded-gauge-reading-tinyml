"""Export a trained source-crop-box localizer to int8 TFLite for STM32N6 deployment.

This script takes a Keras checkpoint that emits a normalized source-space xyxy
crop box (the ``source_crop_box`` output head), quantises it to int8 using a
representative dataset drawn from the project's hard-case manifest, and writes the
TFLite flatbuffer plus a JSON metadata file.  When ``--package-dir`` is provided
the artifacts are also staged into the ``st_ai_output/packages/`` tree that
Cube.AI expects.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Resolve project paths so the script can be run from ``ml/`` or ``ml/scripts/``.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.export import (  # noqa: E402
    ExportConfig,
    ExportResult,
    export_board_tflite_artifacts,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the source-crop-box export job."""
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained source-crop-box Keras model to an int8 TFLite artifact "
            "plus calibration metadata, ready for STM32 Cube.AI ingestion."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained source-crop-box .keras checkpoint.",
    )
    parser.add_argument(
        "--hard-case-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "hard_cases_plus_board30_valid_with_new5.csv",
        help="CSV manifest of labeled captures for quantization calibration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "deployment"
        / "source_crop_box_default_int8",
        help="Directory where the TFLite model and metadata JSON should be written.",
    )
    parser.add_argument(
        "--representative-count",
        type=int,
        default=64,
        help="Number of labeled images to use for int8 calibration.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=224,
        help="Input image height expected by the model.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=224,
        help="Input image width expected by the model.",
    )
    parser.add_argument(
        "--legacy-mobilenetv2-preprocess",
        action="store_true",
        help=(
            "Load the saved model using the legacy MobileNetV2 preprocess Lambda symbol."
        ),
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help=(
            "Optional root under firmware/stm32/n657/st_ai_output/packages/ "
            "where the artifact should be copied into a st_ai_output/ subfolder."
        ),
    )
    parser.add_argument(
        "--package-name",
        type=str,
        default=None,
        help="Basename for the package folder (defaults to output_dir name).",
    )
    return parser.parse_args()


def _package_artifacts(
    result: ExportResult,
    package_dir: Path,
    package_name: str,
) -> None:
    """Copy exported artifacts into the STM32N6 ``st_ai_output`` package tree.

    Cube.AI convention places deliverables under::

        packages/<name>/st_ai_output/<model files>
    """
    # Build the target tree.
    target_root: Path = package_dir / package_name
    target_output: Path = target_root / "st_ai_output"
    target_output.mkdir(parents=True, exist_ok=True)

    # Copy the int8 TFLite flatbuffer into the package.
    dest_tflite: Path = target_output / result.tflite_path.name
    shutil.copy2(result.tflite_path, dest_tflite)

    # Copy the calibration / metadata JSON alongside the model.
    dest_meta: Path = target_output / result.metadata_path.name
    shutil.copy2(result.metadata_path, dest_meta)

    print(f"[PACKAGE] Copied artifacts to {target_output}", flush=True)


def main() -> None:
    """Run the source-crop-box export and optional packaging."""
    args = _parse_args()

    # Assemble the export configuration.
    # source_crop_box models run on the full frame like rectifier/obb heads,
    # so we reuse the same representative-dataset path in ``export.py``.
    config = ExportConfig(
        model_path=args.model,
        output_dir=args.output_dir,
        hard_case_manifest=args.hard_case_manifest,
        deployment_kind="source_crop_box",
        representative_count=args.representative_count,
        image_height=args.image_height,
        image_width=args.image_width,
        legacy_mobilenetv2_preprocess=args.legacy_mobilenetv2_preprocess,
    )

    # Execute the int8 quantization pipeline.
    result: ExportResult = export_board_tflite_artifacts(config)
    print(
        "[EXPORT] Source-crop-box artifacts ready: "
        f"tflite={result.tflite_path} metadata={result.metadata_path} "
        f"input={result.input_shape} output={result.output_shape}",
        flush=True,
    )

    # Optionally stage the files into the Cube.AI package directory.
    if args.package_dir is not None:
        package_name: str = args.package_name or args.output_dir.name
        _package_artifacts(result, args.package_dir, package_name)


if __name__ == "__main__":
    main()
