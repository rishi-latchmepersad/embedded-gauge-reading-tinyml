#!/usr/bin/env python3
"""Quick OpenCV GUI for drawing gauge bounding boxes on board captures.

The tool keeps the workflow intentionally small:
- load a manifest of board images,
- drag one box per image,
- save the result to a CSV compatible with the grouped-manifest builder.

This version uses OpenCV highgui instead of tkinter so it works in the current
WSL environment without extra system packages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.board_bbox_labeling import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_CSV,
    BoardBBoxCandidate,
    BoardBBoxRecord,
    load_board_bbox_candidates,
    load_board_bbox_records,
    write_board_bbox_records,
)
from embedded_gauge_reading_tinyml.board_pipeline import load_capture_image  # noqa: E402
from embedded_gauge_reading_tinyml.capture_labeling import resolve_absolute_image_path  # noqa: E402

WINDOW_NAME: Final[str] = "Board BBox Labeler"
DISPLAY_WIDTH: Final[int] = 1440
DISPLAY_HEIGHT: Final[int] = 980


@dataclass(slots=True)
class DisplayFrame:
    """One rendered image plus the coordinate mapping used by the GUI."""

    bgr_image: np.ndarray
    scale: float
    offset_x: int
    offset_y: int


@dataclass(slots=True)
class DragState:
    """Track the active box drag in canvas coordinates."""

    start_xy: tuple[int, int] | None = None
    current_xy: tuple[int, int] | None = None
    active: bool = False


class BoardBBoxLabelApp:
    """OpenCV annotation loop for board images."""

    def __init__(
        self,
        candidates: list[BoardBBoxCandidate],
        *,
        output_csv: Path,
        existing_records: dict[str, BoardBBoxRecord],
    ) -> None:
        self.candidates = candidates
        self.output_csv = output_csv
        self.records: dict[str, BoardBBoxRecord] = dict(existing_records)
        self.current_index = self._find_start_index()
        self.drag = DragState()
        self.should_exit = False

    def run(self) -> None:
        """Launch the GUI loop."""

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

        while not self.should_exit:
            candidate = self._current_candidate()
            if candidate is None:
                self._draw_message_frame("No images available.")
                key = cv2.waitKey(0)
                if key in (ord("q"), 27):
                    break
                continue

            record = self._current_record(candidate)
            frame = self._render_candidate(candidate, record)
            cv2.imshow(WINDOW_NAME, frame.bgr_image)

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):
                self.should_exit = True
                break
            if key == ord("s"):
                self._save_current()
            elif key == ord(" "):
                self._save_current()
                self._move_index(1)
            elif key == ord("n"):
                self._move_index(1)
            elif key == ord("p"):
                self._move_index(-1)
            elif key == ord("c"):
                self._clear_current_box()
            elif key == ord("k"):
                self._skip_current()

        self._save_records()
        cv2.destroyAllWindows()

    def _find_start_index(self) -> int:
        """Start on the first unlabeled candidate when possible."""

        for index, candidate in enumerate(self.candidates):
            record = self.records.get(candidate.image_path.as_posix())
            if record is None:
                return index
            if not record.has_box and record.quality_flag.strip().lower() != "exclude":
                return index
        return 0

    def _current_candidate(self) -> BoardBBoxCandidate | None:
        """Return the active candidate."""

        if self.current_index < 0 or self.current_index >= len(self.candidates):
            return None
        return self.candidates[self.current_index]

    def _current_record(self, candidate: BoardBBoxCandidate) -> BoardBBoxRecord:
        """Return or create the current record."""

        key = candidate.image_path.as_posix()
        record = self.records.get(key)
        if record is None:
            return BoardBBoxRecord(
                image_path=candidate.image_path,
                source_width=candidate.source_width,
                source_height=candidate.source_height,
                origin_manifest=candidate.origin_manifest,
            )
        return BoardBBoxRecord(
            image_path=candidate.image_path,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
            crop_x_min=record.crop_x_min,
            crop_y_min=record.crop_y_min,
            crop_x_max=record.crop_x_max,
            crop_y_max=record.crop_y_max,
            quality_flag=record.quality_flag,
            label_source=record.label_source,
            notes=record.notes,
            origin_manifest=record.origin_manifest or candidate.origin_manifest,
        )

    def _render_candidate(
        self,
        candidate: BoardBBoxCandidate,
        record: BoardBBoxRecord,
    ) -> DisplayFrame:
        """Load the image and paint the permanent and temporary overlays."""

        absolute_path = resolve_absolute_image_path(candidate.image_path)
        image_array, _kind = load_capture_image(
            absolute_path,
            image_width=candidate.source_width,
            image_height=candidate.source_height,
        )
        rgb = np.asarray(image_array, dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame = self._fit_frame(bgr)

        if record.has_box:
            x0, y0, x1, y1 = self._box_to_canvas(record, frame)
            cv2.rectangle(frame.bgr_image, (x0, y0), (x1, y1), (80, 217, 107), 3)

        if self.drag.active and self.drag.start_xy and self.drag.current_xy:
            x0, y0 = self.drag.start_xy
            x1, y1 = self.drag.current_xy
            cv2.rectangle(frame.bgr_image, (x0, y0), (x1, y1), (0, 221, 255), 2)

        self._draw_overlay_text(frame.bgr_image, candidate, record)
        return frame

    def _fit_frame(self, bgr_image: np.ndarray) -> DisplayFrame:
        """Fit the source image into the display window while preserving aspect ratio."""

        source_height, source_width = bgr_image.shape[:2]
        scale = min(
            float(DISPLAY_WIDTH) / float(source_width),
            float(DISPLAY_HEIGHT) / float(source_height),
        )
        scaled_width = max(1, int(round(source_width * scale)))
        scaled_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(bgr_image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        canvas[:, :] = (20, 20, 20)
        offset_x = max(0, (DISPLAY_WIDTH - scaled_width) // 2)
        offset_y = max(0, (DISPLAY_HEIGHT - scaled_height) // 2)
        canvas[offset_y : offset_y + scaled_height, offset_x : offset_x + scaled_width] = resized
        return DisplayFrame(canvas, float(scale), int(offset_x), int(offset_y))

    def _box_to_canvas(self, record: BoardBBoxRecord, frame: DisplayFrame) -> tuple[int, int, int, int]:
        """Convert source-space box coordinates into canvas coordinates."""

        assert record.crop_x_min is not None
        assert record.crop_y_min is not None
        assert record.crop_x_max is not None
        assert record.crop_y_max is not None
        x0 = int(round(record.crop_x_min * frame.scale + frame.offset_x))
        y0 = int(round(record.crop_y_min * frame.scale + frame.offset_y))
        x1 = int(round(record.crop_x_max * frame.scale + frame.offset_x))
        y1 = int(round(record.crop_y_max * frame.scale + frame.offset_y))
        return x0, y0, x1, y1

    def _draw_overlay_text(
        self,
        image: np.ndarray,
        candidate: BoardBBoxCandidate,
        record: BoardBBoxRecord,
    ) -> None:
        """Paint useful status text onto the image."""

        box_text = "-"
        if record.has_box:
            box_text = (
                f"[{record.crop_x_min:.1f}, {record.crop_y_min:.1f}] -> "
                f"[{record.crop_x_max:.1f}, {record.crop_y_max:.1f}]"
            )
        lines = [
            f"{self.current_index + 1}/{len(self.candidates)}  {candidate.image_path.name}",
            f"box: {box_text}",
            f"flag: {record.quality_flag or 'review'}",
            "drag=box  s=save  space=save+next  p/n=prev/next  c=clear  k=skip  q=quit",
        ]
        y = 28
        for line in lines:
            cv2.putText(
                image,
                line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            y += 28

    def _canvas_to_source(self, x: int, y: int, frame: DisplayFrame) -> tuple[float, float]:
        """Convert a canvas coordinate back to source image space."""

        source_x = (float(x) - float(frame.offset_x)) / frame.scale
        source_y = (float(y) - float(frame.offset_y)) / frame.scale
        candidate = self._current_candidate()
        if candidate is None:
            return 0.0, 0.0
        return (
            _clamp(source_x, 0.0, float(candidate.source_width)),
            _clamp(source_y, 0.0, float(candidate.source_height)),
        )

    def _canvas_for_source(self, x: float, y: float, frame: DisplayFrame) -> tuple[int, int]:
        """Convert a source coordinate into canvas space."""

        return (
            int(round(x * frame.scale + frame.offset_x)),
            int(round(y * frame.scale + frame.offset_y)),
        )

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object | None) -> None:
        """Track drag-to-box interactions."""

        candidate = self._current_candidate()
        if candidate is None:
            return

        frame = self._render_candidate(candidate, self._current_record(candidate))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = DragState(start_xy=(x, y), current_xy=(x, y), active=True)
            cv2.imshow(WINDOW_NAME, frame.bgr_image)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.drag.active:
            self.drag.current_xy = (x, y)
            cv2.imshow(WINDOW_NAME, self._render_candidate(candidate, self._current_record(candidate)).bgr_image)
            return

        if event != cv2.EVENT_LBUTTONUP or not self.drag.active or self.drag.start_xy is None:
            return

        self.drag.current_xy = (x, y)
        x0, y0 = self.drag.start_xy
        x1, y1 = self.drag.current_xy
        self.drag = DragState()

        source_x0, source_y0 = self._canvas_to_source(min(x0, x1), min(y0, y1), frame)
        source_x1, source_y1 = self._canvas_to_source(max(x0, x1), max(y0, y1), frame)
        if abs(source_x1 - source_x0) < 2.0 or abs(source_y1 - source_y0) < 2.0:
            return

        record = BoardBBoxRecord(
            image_path=candidate.image_path,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
            crop_x_min=source_x0,
            crop_y_min=source_y0,
            crop_x_max=source_x1,
            crop_y_max=source_y1,
            quality_flag="review",
            label_source="manual_gui",
            origin_manifest=candidate.origin_manifest,
        )
        self.records[candidate.image_path.as_posix()] = record
        self._save_records()
        cv2.imshow(WINDOW_NAME, self._render_candidate(candidate, record).bgr_image)

    def _save_records(self) -> None:
        """Write the current records to disk in candidate order."""

        ordered_records: list[BoardBBoxRecord] = []
        for candidate in self.candidates:
            record = self.records.get(candidate.image_path.as_posix())
            if record is not None:
                ordered_records.append(record)
        write_board_bbox_records(self.output_csv, ordered_records)

    def _save_current(self) -> None:
        """Mark the current image as reviewed and save the CSV."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        record = self._current_record(candidate)
        if record.has_box:
            self.records[candidate.image_path.as_posix()] = record.with_quality_flag("review")
        self._save_records()

    def _clear_current_box(self) -> None:
        """Remove any box for the current candidate."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        self.records.pop(candidate.image_path.as_posix(), None)
        self._save_records()

    def _skip_current(self) -> None:
        """Mark the current candidate as excluded."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        self.records[candidate.image_path.as_posix()] = BoardBBoxRecord(
            image_path=candidate.image_path,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
            quality_flag="exclude",
            label_source="manual_gui",
            origin_manifest=candidate.origin_manifest,
        )
        self._save_records()
        self._move_index(1)

    def _move_index(self, delta: int) -> None:
        """Move the active index and keep it in range."""

        if not self.candidates:
            return
        self.current_index = max(0, min(self.current_index + delta, len(self.candidates) - 1))

    def _draw_message_frame(self, message: str) -> None:
        """Show a simple status frame when no candidate is available."""

        frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        frame[:, :] = (20, 20, 20)
        cv2.putText(
            frame,
            message,
            (40, DISPLAY_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, frame)


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp ``value`` into the requested range."""

    return float(max(lower, min(value, upper)))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Label bounding boxes for board captures.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--image-column", type=str, default="image_path")
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """Launch the board bbox labeler."""

    args = parse_args()
    candidates = load_board_bbox_candidates(
        args.manifest,
        image_column=args.image_column,
        limit=args.limit,
    )
    existing_records = load_board_bbox_records(args.output)
    app = BoardBBoxLabelApp(candidates, output_csv=args.output, existing_records=existing_records)
    app.run()


if __name__ == "__main__":
    main()
