#!/usr/bin/env python3
"""Render every OBB training image with its bounding box overlaid.

This utility reads the grouped JSON manifest used by the OBB pipeline, extracts
the requested source kinds, and writes:

* one overlay PNG per labeled image
* paginated contact sheets for quick paging
* a small summary JSON for bookkeeping

The output is meant for fast manual review before retraining a board-localizer.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embedded_gauge_reading_tinyml.board_pipeline import load_capture_image
from embedded_gauge_reading_tinyml.capture_labeling import resolve_absolute_image_path, to_repo_relative_path

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST: Final[Path] = REPO_ROOT / "tmp" / "labelled_captured_images_board_bbox_plus_board_reviews.json"
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "tmp" / "obb_training_box_review"
DEFAULT_SOURCE_KINDS: Final[tuple[str, ...]] = ("pxl_geometry", "reviewed_geometry")
DEFAULT_PAGE_COLUMNS: Final[int] = 4
DEFAULT_PAGE_ROWS: Final[int] = 4
DEFAULT_CARD_WIDTH: Final[int] = 360
DEFAULT_CARD_HEIGHT: Final[int] = 300


@dataclass(frozen=True, slots=True)
class ReviewSample:
    """One manifest image plus the bounding box we want to review."""

    image_path: Path
    source_width: int
    source_height: int
    box_xyxy: tuple[float, float, float, float]
    source_kind: str
    quality_flag: str
    notes: str
    source_manifest: str
    source_row_index: int


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the review renderer."""

    parser = argparse.ArgumentParser(description="Render OBB training images with bbox overlays.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Grouped JSON manifest to review.")
    parser.add_argument(
        "--source-kind",
        dest="source_kinds",
        action="append",
        default=[],
        help="Source kind to include. May be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for render outputs.")
    parser.add_argument("--page-columns", type=int, default=DEFAULT_PAGE_COLUMNS, help="Tiles per page column.")
    parser.add_argument("--page-rows", type=int, default=DEFAULT_PAGE_ROWS, help="Tiles per page row.")
    parser.add_argument("--card-width", type=int, default=DEFAULT_CARD_WIDTH, help="Overlay card width in pixels.")
    parser.add_argument("--card-height", type=int, default=DEFAULT_CARD_HEIGHT, help="Overlay card height in pixels.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images to render.")
    return parser.parse_args()


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate the grouped JSON manifest."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "images" not in payload or not isinstance(payload["images"], list):
        raise ValueError(f"{manifest_path} does not look like a grouped manifest.")
    return payload


def _parse_optional_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    """Return the first finite float found under the requested keys."""

    for key in keys:
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return float(parsed)
    return None


def _parse_optional_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    """Return the first non-empty string found under the requested keys."""

    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_box_xyxy(source_row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Extract a box from either board-bbox or geometry-manifest field names."""

    bbox_fields = (
        ("crop_x_min", "crop_y_min", "crop_x_max", "crop_y_max"),
        ("loose_crop_x1", "loose_crop_y1", "loose_crop_x2", "loose_crop_y2"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        ("x_min", "y_min", "x_max", "y_max"),
    )
    for x1_key, y1_key, x2_key, y2_key in bbox_fields:
        x1 = _parse_optional_float(source_row, (x1_key,))
        y1 = _parse_optional_float(source_row, (y1_key,))
        x2 = _parse_optional_float(source_row, (x2_key,))
        y2 = _parse_optional_float(source_row, (y2_key,))
        if None not in (x1, y1, x2, y2):
            return (float(x1), float(y1), float(x2), float(y2))
    raise ValueError(f"No supported bbox fields were found in row keys: {sorted(source_row.keys())[:12]}")


def _load_samples(payload: dict[str, Any], source_kinds: Sequence[str]) -> list[ReviewSample]:
    """Flatten manifest entries into labeled review samples."""

    allowed = set(source_kinds)
    samples: list[ReviewSample] = []
    for image_index, image_entry in enumerate(payload["images"]):
        annotations = image_entry.get("annotations", [])
        if not isinstance(annotations, list):
            continue

        chosen: dict[str, Any] | None = None
        for annotation in annotations:
            if str(annotation.get("source_kind", "")).strip() not in allowed:
                continue
            chosen = annotation
            break
        if chosen is None:
            continue

        source_row = chosen.get("source_row", {})
        if not isinstance(source_row, dict):
            continue

        image_path_text = _parse_optional_text(chosen, ("image_path", "source_image_path")) or _parse_optional_text(source_row, ("image_path",))
        if not image_path_text:
            continue
        image_path = to_repo_relative_path(image_path_text)
        samples.append(
            ReviewSample(
                image_path=image_path,
                source_width=int(_parse_optional_float(source_row, ("source_width",)) or 0.0),
                source_height=int(_parse_optional_float(source_row, ("source_height",)) or 0.0),
                box_xyxy=_extract_box_xyxy(source_row),
                source_kind=str(chosen.get("source_kind", "")).strip(),
                quality_flag=_parse_optional_text(source_row, ("quality_flag",)) or "review",
                notes=_parse_optional_text(source_row, ("notes",)),
                source_manifest=_parse_optional_text(source_row, ("source_manifest",)),
                source_row_index=image_index,
            )
        )
    return samples


def _load_source_image(sample: ReviewSample) -> np.ndarray:
    """Load the source image as an RGB array."""

    capture_path = resolve_absolute_image_path(sample.image_path)
    rgb, _kind = load_capture_image(
        capture_path,
        image_width=sample.source_width,
        image_height=sample.source_height,
    )
    return np.asarray(rgb, dtype=np.uint8)


def _resize_keep_aspect(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Resize an image while keeping aspect ratio."""

    image = image.copy().convert("RGB")
    image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return image


def _draw_text_panel(draw: ImageDraw.ImageDraw, lines: Sequence[str], *, width: int) -> None:
    """Render a compact metadata panel at the top of the image."""

    font = ImageFont.load_default()
    panel_height = 14 * len(lines) + 12
    draw.rectangle((0, 0, width, panel_height), fill=(0, 0, 0, 190))
    for line_index, line in enumerate(lines):
        draw.text((8, 6 + (line_index * 14)), line, fill="white", font=font)


def _render_overlay_card(sample: ReviewSample, *, card_width: int, card_height: int) -> Image.Image:
    """Render one review card with the bounding box overlaid."""

    source_rgb = _load_source_image(sample)
    display = _resize_keep_aspect(Image.fromarray(source_rgb), card_width, card_height - 44)
    draw = ImageDraw.Draw(display)

    x1, y1, x2, y2 = sample.box_xyxy
    scale_x = display.width / max(1.0, float(source_rgb.shape[1]))
    scale_y = display.height / max(1.0, float(source_rgb.shape[0]))
    scaled_box = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
    border_color = "cyan" if sample.quality_flag.strip().lower() == "clean" else "orange"
    draw.rectangle(scaled_box, outline=border_color, width=3)

    _draw_text_panel(
        draw,
        [
            sample.image_path.name,
            f"{sample.source_kind} | {sample.quality_flag or 'review'}",
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})",
            sample.notes[:48] if sample.notes else "",
        ],
        width=display.width,
    )

    card = Image.new("RGB", (card_width, card_height), (18, 18, 18))
    paste_x = max(0, (card_width - display.width) // 2)
    paste_y = max(44, (card_height - display.height) // 2)
    card.paste(display, (paste_x, paste_y))
    return card


def _write_index_html(output_dir: Path, summary: dict[str, Any]) -> None:
    """Write a tiny HTML index so the review set is easy to browse."""

    pages = summary.get("pages", [])
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>OBB training box review</title>",
        "<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee} img{max-width:100%;height:auto;border:1px solid #333;margin:12px 0}</style>",
        "</head><body>",
        f"<h1>OBB training box review</h1>",
        f"<p>Total samples: {summary.get('sample_count', 0)}</p>",
    ]
    for page in pages:
        lines.append(f"<h2>{page['name']}</h2>")
        lines.append(f"<img src='{page['name']}' alt='{page['name']}'>")
    lines.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Render the selected training images and write review artifacts."""

    args = _parse_args()
    manifest = _load_manifest(args.manifest)
    source_kinds = tuple(args.source_kinds or DEFAULT_SOURCE_KINDS)
    samples = _load_samples(manifest, source_kinds)
    if args.limit is not None:
        samples = samples[: max(0, int(args.limit))]
    if not samples:
        raise SystemExit("No matching training images were found in the manifest.")

    output_dir = args.output_dir
    overlays_dir = output_dir / "overlays"
    pages_dir = output_dir / "pages"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    card_paths: list[Path] = []
    for index, sample in enumerate(samples, start=1):
        card = _render_overlay_card(sample, card_width=args.card_width, card_height=args.card_height)
        card_name = f"{index:04d}_{sample.image_path.stem}.png"
        card_path = overlays_dir / card_name
        card.save(card_path)
        card_paths.append(card_path)

    page_columns = max(1, int(args.page_columns))
    page_rows = max(1, int(args.page_rows))
    cards_per_page = page_columns * page_rows
    page_paths: list[Path] = []
    for page_index in range(int(math.ceil(len(card_paths) / float(cards_per_page)))):
        start = page_index * cards_per_page
        end = min(len(card_paths), start + cards_per_page)
        page_cards = card_paths[start:end]
        page = Image.new("RGB", (page_columns * args.card_width, page_rows * args.card_height), (12, 12, 12))
        for card_index, card_path in enumerate(page_cards):
            with Image.open(card_path) as card_image:
                x = (card_index % page_columns) * args.card_width
                y = (card_index // page_columns) * args.card_height
                page.paste(card_image.convert("RGB"), (x, y))
        page_path = pages_dir / f"page_{page_index + 1:03d}.png"
        page.save(page_path)
        page_paths.append(page_path)

    summary = {
        "manifest": args.manifest.as_posix(),
        "output_dir": output_dir.as_posix(),
        "source_kinds": list(source_kinds),
        "sample_count": len(samples),
        "card_count": len(card_paths),
        "page_count": len(page_paths),
        "pages": [{"name": path.name, "path": path.as_posix()} for path in page_paths],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_index_html(output_dir, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
