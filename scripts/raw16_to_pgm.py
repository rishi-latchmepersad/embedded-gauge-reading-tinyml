"""Convert STM32 camera raw captures into a simple viewable grayscale PGM.

The firmware currently stores IMX335 RAW10 data in 16-bit containers as
`capture_XXXX.raw16` or, during diagnostics, in 32-bit containers as
`capture_XXXX.raw32`. This script reads that binary dump, scales the samples to
8-bit, and writes a `.pgm` preview that can be opened by many image viewers.
"""

from __future__ import annotations

import argparse
import array
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the raw16 preview conversion."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a raw capture from the STM32 camera pipeline into "
            "an 8-bit grayscale PGM preview."
        )
    )
    parser.add_argument("input", type=Path, help="Path to the .raw16 or .raw32 file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .pgm path. Defaults beside the input file.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width in pixels. Defaults to the firmware capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height in pixels. Defaults to the firmware capture height.",
    )
    parser.add_argument(
        "--bits-valid",
        type=int,
        default=10,
        help=(
            "Number of valid sensor bits stored in each 16-bit sample. "
            "Defaults to 10 for the current IMX335 RAW10 path."
        ),
    )
    parser.add_argument(
        "--sample-bytes",
        type=int,
        choices=(2, 4),
        default=None,
        help=(
            "Bytes per stored sample. Defaults to auto-detect from file size "
            "for the requested width and height."
        ),
    )
    parser.add_argument(
        "--auto-range",
        action="store_true",
        help="Stretch the preview using the frame min/max instead of fixed bit depth.",
    )
    return parser.parse_args()


def load_raw_samples(
    path: Path, expected_pixels: int, sample_bytes: int
) -> array.array[int]:
    """Load little-endian integer samples and validate the expected frame size."""
    raw_bytes = path.read_bytes()
    expected_size = expected_pixels * sample_bytes
    if len(raw_bytes) != expected_size:
        raise ValueError(
            f"Expected {expected_size} bytes for one frame, found {len(raw_bytes)} bytes."
        )

    if sample_bytes == 2:
        samples = array.array("H")
    elif sample_bytes == 4:
        samples = array.array("I")
    else:
        raise ValueError("sample_bytes must be 2 or 4.")

    samples.frombytes(raw_bytes)
    return samples


def infer_sample_bytes(path: Path, expected_pixels: int) -> int:
    """Infer whether the capture stores 16-bit or 32-bit samples from file size."""
    file_size = path.stat().st_size
    if file_size == expected_pixels * 2:
        return 2
    if file_size == expected_pixels * 4:
        return 4
    raise ValueError(
        "Could not infer sample width from file size. Use --sample-bytes 2 or 4."
    )


def scale_samples_to_u8(
    samples: array.array[int], bits_valid: int, auto_range: bool
) -> bytearray:
    """Scale 16-bit raw samples down to an 8-bit preview image."""
    if bits_valid <= 0 or bits_valid > 16:
        raise ValueError("--bits-valid must be between 1 and 16.")

    if auto_range:
        sample_min = min(samples)
        sample_max = max(samples)
    else:
        sample_min = 0
        sample_max = (1 << bits_valid) - 1

    if sample_max <= sample_min:
        return bytearray(len(samples))

    scaled = bytearray(len(samples))
    scale_denominator = sample_max - sample_min

    for index, sample in enumerate(samples):
        clipped = sample
        if clipped < sample_min:
            clipped = sample_min
        elif clipped > sample_max:
            clipped = sample_max

        scaled[index] = ((clipped - sample_min) * 255) // scale_denominator

    return scaled


def write_pgm(path: Path, width: int, height: int, pixels: bytearray) -> None:
    """Write an 8-bit binary PGM image."""
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + pixels)


def main() -> int:
    """Convert one raw16 capture into a grayscale preview file."""
    args = parse_args()
    output_path = args.output or args.input.with_suffix(".pgm")
    expected_pixels = args.width * args.height
    sample_bytes = args.sample_bytes or infer_sample_bytes(args.input, expected_pixels)

    samples = load_raw_samples(args.input, expected_pixels, sample_bytes)
    preview_pixels = scale_samples_to_u8(
        samples=samples,
        bits_valid=args.bits_valid,
        auto_range=args.auto_range,
    )
    write_pgm(output_path, args.width, args.height, preview_pixels)

    print(f"Wrote preview image to {output_path} (sample_bytes={sample_bytes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
