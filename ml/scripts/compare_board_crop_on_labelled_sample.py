"""Compare a labeled training sample against the board-style crop path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make the package importable when this script is run from the ml/ directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_crop_compare import compare_labelled_sample
from embedded_gauge_reading_tinyml.dataset import LABELLED_DIR, RAW_DIR, Sample, load_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the crop comparison job."""
    parser = argparse.ArgumentParser(
        description="Compare a labeled gauge image using training and board crop paths."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Specific labeled image to compare. If omitted, sample-index is used.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index into the loaded labelled dataset when image-path is not given.",
    )
    parser.add_argument(
        "--labelled-dir",
        type=Path,
        default=LABELLED_DIR,
        help="Directory containing CVAT zip exports.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing the raw labeled images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "board_crop_compare",
        help="Directory where comparison artifacts should be written.",
    )
    parser.add_argument(
        "--crop-pad-ratio",
        type=float,
        default=0.10,
        help="Padding ratio applied around the labeled dial ellipse.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square output size for the crop comparison.",
    )
    return parser.parse_args()


def _select_sample(args: argparse.Namespace) -> tuple[Path, Sample]:
    """Select one labeled sample by image path or by dataset index."""
    samples = load_dataset(labelled_dir=args.labelled_dir, raw_dir=args.raw_dir)
    if not samples:
        raise ValueError(
            f"No labeled samples found in {args.labelled_dir} with raw images in {args.raw_dir}."
        )

    if args.image_path is not None:
        image_path = args.image_path.resolve()
        for sample in samples:
            if sample.image_path.resolve() == image_path:
                return image_path, sample
        raise ValueError(f"Image path not found in labeled dataset: {image_path}")

    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(
            f"sample-index {args.sample_index} is out of range for {len(samples)} samples."
        )

    sample = samples[args.sample_index]
    return sample.image_path, sample


def main() -> None:
    """Run the crop comparison and print a compact human-readable summary."""
    args = parse_args()
    image_path, sample = _select_sample(args)

    print(f"[COMPARE] Sample: {image_path}")
    print(
        "[COMPARE] Labels: "
        f"dial=({sample.dial.cx:.1f},{sample.dial.cy:.1f},{sample.dial.rx:.1f},{sample.dial.ry:.1f}) "
        f"center=({sample.center.x:.1f},{sample.center.y:.1f}) "
        f"tip=({sample.tip.x:.1f},{sample.tip.y:.1f})"
    )

    report = compare_labelled_sample(
        sample,
        args.output_dir,
        image_size=args.image_size,
        crop_pad_ratio=args.crop_pad_ratio,
    )

    board = report.board_crop_estimate
    box = board.crop_box
    print(
        "[COMPARE] Training crop: "
        f"x_min={report.training_crop_box[0]:.1f} "
        f"y_min={report.training_crop_box[1]:.1f} "
        f"x_max={report.training_crop_box[2]:.1f} "
        f"y_max={report.training_crop_box[3]:.1f}"
    )
    print(
        "[COMPARE] Board crop: "
        f"x_min={box.x_min} y_min={box.y_min} w={box.width} h={box.height} "
        f"centroid=({box.centroid_x},{box.centroid_y}) "
        f"bright_count={box.bright_count} "
        f"center_luma={board.center_luma} "
        f"mean_luma={board.mean_luma:.2f} "
        f"min_luma={board.min_luma} "
        f"max_luma={board.max_luma}"
    )
    print(f"[COMPARE] Mean abs diff: {report.mean_abs_diff:.2f}")
    print(f"[COMPARE] Training crop image: {report.training_crop_path}")
    print(f"[COMPARE] Board crop image: {report.board_crop_path}")
    print(f"[COMPARE] Comparison figure: {report.comparison_figure_path}")
    print(f"[COMPARE] JSON report: {report.report_json_path}")


if __name__ == "__main__":
    main()
