"""Convert latest YUV422 captures to PNG previews.

Usage:
    python convert_latest_yuv422.py [count]

Converts the most recent .yuv422 files to .png for easy viewing.
Default count is 10 files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_yuv422_yuyv(path: Path, width: int = 224, height: int = 224) -> np.ndarray:
    """Load a packed YUV422 YUYV capture and return grayscale image.

    YUV422 YUYV format: [Y0, U, Y1, V] for 2 pixels
    We extract only the Y (luma) values for grayscale output.
    """
    expected_size = width * height * 2  # 2 bytes per pixel in YUV422
    raw_bytes = path.read_bytes()

    if len(raw_bytes) != expected_size:
        raise ValueError(
            f"Expected {expected_size} bytes for {width}x{height} YUV422, "
            f"got {len(raw_bytes)} bytes"
        )

    # Convert to numpy array
    data = np.frombuffer(raw_bytes, dtype=np.uint8)

    # Extract Y values (every other byte starting at position 0)
    # YUYV layout: Y0, U0, Y1, V0, Y2, U1, Y3, V1, ...
    y_values = data[0::2].reshape(height, width)

    return y_values


def convert_latest_yuv422(captured_dir: Path, count: int = 10) -> None:
    """Convert the most recent YUV422 captures to PNG."""
    # Find all YUV422 files sorted by modification time
    yuv_files = sorted(
        captured_dir.glob("*.yuv422"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not yuv_files:
        print(f"No .yuv422 files found in {captured_dir}")
        return

    print(f"Converting {min(count, len(yuv_files))} most recent YUV422 files...")
    print("-" * 60)

    for i, yuv_path in enumerate(yuv_files[:count], 1):
        try:
            # Load grayscale luma values
            gray = load_yuv422_yuyv(yuv_path, width=224, height=224)

            # Convert to PIL Image
            pil_image = Image.fromarray(gray, mode="L")

            # Save as PNG
            output_path = yuv_path.with_suffix(".png")
            pil_image.save(output_path, "PNG")

            print(f"[{i}/{count}] {yuv_path.name}")
            print(f"    → {output_path.name}")

        except Exception as e:
            print(f"[{i}/{count}] {yuv_path.name} - ERROR: {e}")

    print("-" * 60)
    print("Conversion complete!")


if __name__ == "__main__":
    captured_dir = Path(
        "D:/Projects/embedded-gauge-reading-tinyml/data/captured/images"
    )

    # Check if argument is a file path or count
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.endswith(".yuv422"):
            # Single file conversion
            yuv_path = captured_dir / arg
            if yuv_path.exists():
                try:
                    gray = load_yuv422_yuyv(yuv_path, width=224, height=224)
                    pil_image = Image.fromarray(gray, mode="L")
                    output_path = yuv_path.with_suffix(".png")
                    pil_image.save(output_path, "PNG")
                    print(f"Converted: {yuv_path.name} -> {output_path.name}")
                except Exception as e:
                    print(f"Error converting {yuv_path.name}: {e}")
            else:
                print(f"File not found: {yuv_path}")
        else:
            # Count argument
            try:
                count = int(arg)
                convert_latest_yuv422(captured_dir, count)
            except ValueError:
                print(f"Usage: python convert_latest_yuv422.py [count|filename.yuv422]")
    else:
        # Default: convert 10 most recent
        convert_latest_yuv422(captured_dir, 10)
