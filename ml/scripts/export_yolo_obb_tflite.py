"""Stage the validated YOLO11n-OBB TFLite exports for deployment.

The training job already produces the full-integer-quantized TFLite export
that the STM32N6 board needs. This script copies the validated artifacts into
the canonical deployment filenames used by the packaging flow.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "yolo_obb_320"
SOURCE_DIR = ARTIFACT_ROOT / "train" / "weights" / "best_saved_model"
EXPORT_DIR = ARTIFACT_ROOT / "tflite"
INT8_SOURCE = SOURCE_DIR / "best_full_integer_quant.tflite"
FLOAT_CANDIDATES = (
    SOURCE_DIR / "best_float32.tflite",
    SOURCE_DIR / "best_float16.tflite",
    SOURCE_DIR / "best_integer_quant.tflite",
)
INT8_TARGET = EXPORT_DIR / "yolo11n_obb_int8.tflite"
FLOAT_TARGET = EXPORT_DIR / "yolo11n_obb_f32.tflite"


def _copy_first_existing(candidates: tuple[Path, ...], target: Path) -> Path:
    """Copy the first available source artifact into the canonical filename."""
    for source in candidates:
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return source
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No source artifact found. Tried: {joined}")


def main() -> None:
    """Copy the fully quantized and float TFLite exports into deployment names."""
    if not INT8_SOURCE.is_file():
        raise FileNotFoundError(f"Missing full-integer OBB export: {INT8_SOURCE}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(INT8_SOURCE, INT8_TARGET)
    float_source = _copy_first_existing(FLOAT_CANDIDATES, FLOAT_TARGET)

    print("Staged YOLO11n-OBB TFLite exports:")
    print(f"  int8  : {INT8_TARGET}  <- {INT8_SOURCE}")
    print(f"  float : {FLOAT_TARGET}  <- {float_source}")


if __name__ == "__main__":
    main()
