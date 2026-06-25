#!/usr/bin/env python3
"""Generate an OBB crop manifest for downstream SimCC training.

The script runs the deployed OBB v2 box model on the grouped labeled captures,
decodes each crop back into source-image coordinates, and writes the results to
``tmp/obb_output/obb_crop_manifest.json`` by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_edge_litert.interpreter import Interpreter, OpResolverType  # noqa: E402
from embedded_gauge_reading_tinyml.board_pipeline import (  # noqa: E402
    decode_obb_box_crop_box,
    load_capture_image,
    prepare_full_frame,
)
from embedded_gauge_reading_tinyml.capture_labeling import (  # noqa: E402
    is_original_capture_filename,
)
from embedded_gauge_reading_tinyml.obb_crop_manifest import (  # noqa: E402
    ObbCropRecord,
    dump_obb_crop_manifest,
)

DEFAULT_MANIFEST: Path = PROJECT_ROOT / "data" / "labelled_captured_images.json"
DEFAULT_OBB_MODEL: Path = (
    PROJECT_ROOT / "artifacts" / "training" / "obb_v2_box_20260622_203432" / "model_int8.tflite"
)
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "tmp" / "obb_output"
DEFAULT_PREVIEW_COUNT: int = 12
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_INPUT_SIZE: int = 224


@dataclass(frozen=True, slots=True)
class SourceCaptureSample:
    """One grouped-manifest entry selected for OBB inference."""

    image_path: Path
    source_width: int
    source_height: int
    source_kind: str


def _as_float(value: Any) -> float | None:
    """Parse a JSON scalar into a finite float."""

    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _resolve_source_size(image_path: Path, annotations: list[dict[str, Any]]) -> tuple[int, int]:
    """Resolve the source image dimensions from labels or the image itself."""

    for annotation in annotations:
        row = annotation["source_row"]
        source_width = _as_float(row.get("source_width"))
        source_height = _as_float(row.get("source_height"))
        if source_width is not None and source_height is not None:
            width = int(round(source_width))
            height = int(round(source_height))
            if width > 0 and height > 0:
                return width, height

    absolute_path = (REPO_ROOT / image_path).resolve() if not image_path.is_absolute() else image_path
    suffix = absolute_path.suffix.lower()
    if suffix == ".yuv422":
        file_size = absolute_path.stat().st_size
        inferred_pixels = file_size / 2.0
        inferred_dim = int(round(math.sqrt(inferred_pixels)))
        if inferred_dim > 0 and inferred_dim * inferred_dim * 2 == file_size:
            return inferred_dim, inferred_dim
        raise ValueError(
            f"{image_path} is a raw YUV capture, but no source_width/source_height labels were present."
        )

    with Image.open(absolute_path) as image:
        width, height = image.size
    return int(width), int(height)


def _load_grouped_manifest(manifest_path: Path) -> list[SourceCaptureSample]:
    """Load the grouped labeled-manifest rows that should flow through OBB."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = payload["images"]
    samples: list[SourceCaptureSample] = []
    for entry in images:
        image_path = Path(str(entry["image_path"]))
        if not is_original_capture_filename(image_path.name):
            continue
        annotations = list(entry["annotations"])
        source_width, source_height = _resolve_source_size(image_path, annotations)
        source_kind = str(annotations[0]["source_kind"]) if annotations else "unknown"
        samples.append(
            SourceCaptureSample(
                image_path=image_path,
                source_width=source_width,
                source_height=source_height,
                source_kind=source_kind,
            )
        )
    return samples


def _load_image(sample: SourceCaptureSample) -> np.ndarray:
    """Load one source capture as an RGB array."""

    absolute_path = (REPO_ROOT / sample.image_path).resolve() if not sample.image_path.is_absolute() else sample.image_path
    source_image, _kind = load_capture_image(
        absolute_path,
        image_width=sample.source_width,
        image_height=sample.source_height,
    )
    return np.asarray(source_image, dtype=np.uint8)


def _quantize_input(batch: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    """Quantize a float32 batch into the model's int8 input tensor."""

    scale = float(input_details["quantization"][0])
    zero_point = int(input_details["quantization"][1])
    dtype = input_details["dtype"]
    if scale == 0.0:
        return batch.astype(dtype)
    qmin = np.iinfo(dtype).min
    qmax = np.iinfo(dtype).max
    quantized = np.round(batch / scale + zero_point)
    return np.clip(quantized, qmin, qmax).astype(dtype)


def _dequantize_output(output_tensor: np.ndarray, output_details: dict[str, Any]) -> np.ndarray:
    """Convert a quantized output tensor back to float32 when needed."""
    dtype = np.dtype(output_details["dtype"])
    array = np.asarray(output_tensor)
    if np.issubdtype(dtype, np.integer):
        scale = float(output_details["quantization"][0])
        zero_point = int(output_details["quantization"][1])
        if scale != 0.0:
            return (array.astype(np.float32) - float(zero_point)) * scale
    return array.astype(np.float32)


def _load_interpreter(model_path: Path) -> Interpreter:
    """Load the LiteRT interpreter with the host-safe op resolver."""

    interpreter = Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    return interpreter


def _predict_obb(
    interpreter: Interpreter,
    image: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Run one OBB inference and return confidence plus the 4D box tensor."""

    input_details = interpreter.get_input_details()[0]
    image_batch = np.asarray(image[np.newaxis, ...], dtype=np.float32)
    quantized_batch = _quantize_input(image_batch, input_details)

    output_details = interpreter.get_output_details()
    box_detail = next(detail for detail in output_details if int(detail["shape"][-1]) == 4)
    conf_detail = next(detail for detail in output_details if int(detail["shape"][-1]) == 1)
    interpreter.set_tensor(input_details["index"], quantized_batch)
    interpreter.invoke()
    box = _dequantize_output(
        interpreter.get_tensor(box_detail["index"]),
        box_detail,
    ).reshape(-1)
    confidence = float(
        _dequantize_output(
            interpreter.get_tensor(conf_detail["index"]),
            conf_detail,
        ).reshape(-1)[0]
    )
    return confidence, box


def _make_record(
    sample: SourceCaptureSample,
    *,
    confidence: float,
    box: np.ndarray,
    image_size: int,
    obb_crop_scale: float,
) -> ObbCropRecord:
    """Decode one OBB output into a manifest record."""

    decision = decode_obb_box_crop_box(
        confidence,
        box,
        source_width=sample.source_width,
        source_height=sample.source_height,
        input_size=image_size,
        obb_crop_scale=obb_crop_scale,
    )
    return ObbCropRecord(
        image_path=sample.image_path,
        source_width=sample.source_width,
        source_height=sample.source_height,
        crop_box_xyxy=decision.crop_box_xyxy,
        confidence=confidence,
        accepted=decision.accepted,
        fallback_reason=decision.fallback_reason,
        source_kind=sample.source_kind,
    )


def _draw_preview(sample: SourceCaptureSample, record: ObbCropRecord, output_path: Path) -> None:
    """Render one preview image with the decoded crop box overlaid."""

    source_image = Image.fromarray(_load_image(sample), mode="RGB")
    draw = ImageDraw.Draw(source_image)
    x_min, y_min, x_max, y_max = record.crop_box_xyxy
    color = "green" if record.accepted else "red"
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=4)
    label = f"conf={record.confidence:.3f} accepted={record.accepted}"
    if record.fallback_reason:
        label += f" {record.fallback_reason}"
    draw.text((8, 8), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_image.save(output_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Generate OBB crop overrides for SimCC training.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--obb-model", type=Path, default=DEFAULT_OBB_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=DEFAULT_PREVIEW_COUNT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--obb-crop-scale", type=float, default=1.2)
    return parser.parse_args()


def main() -> None:
    """Generate the crop manifest and a small preview set."""

    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.obb_model.exists():
        raise FileNotFoundError(f"OBB model not found: {args.obb_model}")

    samples = _load_grouped_manifest(args.manifest)
    if args.max_images > 0:
        samples = samples[: args.max_images]
    if not samples:
        raise ValueError("No usable source captures were found in the manifest.")

    interpreter = _load_interpreter(args.obb_model)
    input_details = interpreter.get_input_details()[0]
    print(
        f"[OBB] Loaded {args.obb_model.name} input_shape={tuple(input_details['shape'])} "
        f"quant={input_details['quantization']}"
    )
    for output_detail in interpreter.get_output_details():
        print(
            f"[OBB] output {output_detail['name']} shape={tuple(output_detail['shape'])} "
            f"dtype={output_detail['dtype']}"
        )

    output_dir = args.output_dir
    previews_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[ObbCropRecord] = []
    accepted_count = 0
    for index, sample in enumerate(samples, start=1):
        source_image = _load_image(sample)
        full_frame = prepare_full_frame(source_image, image_size=args.image_size)
        confidence, box = _predict_obb(interpreter, full_frame)
        record = _make_record(
            sample,
            confidence=confidence,
            box=box,
            image_size=args.image_size,
            obb_crop_scale=args.obb_crop_scale,
        )
        records.append(record)
        accepted_count += int(record.accepted)

        if index <= max(0, int(args.preview_count)):
            preview_path = previews_dir / f"{sample.image_path.stem}_obb.png"
            _draw_preview(sample, record, preview_path)

        if index % max(1, int(args.batch_size)) == 0 or index == len(samples):
            print(
                f"[OBB] processed {index}/{len(samples)} "
                f"accepted={accepted_count} rejected={index - accepted_count}",
                flush=True,
            )

    manifest_path = dump_obb_crop_manifest(
        records,
        output_dir / "obb_crop_manifest.json",
        model_path=args.obb_model,
        source_manifest=args.manifest,
    )
    print(f"[OBB] wrote manifest to {manifest_path}")
    print(f"[OBB] preview images in {previews_dir}")


if __name__ == "__main__":
    main()
