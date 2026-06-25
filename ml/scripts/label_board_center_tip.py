#!/usr/bin/env python3
"""Quick OpenCV GUI for labeling board capture center and tip points.

The workflow is intentionally narrow:
- start from the curated board review batch,
- click the gauge center and needle tip,
- save to a manifest-friendly CSV for SimCC training.

This avoids Tkinter so the tool works in the current WSL environment without
extra system packages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import cv2
import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.capture_labeling import (  # noqa: E402
    CaptureCandidate,
    CaptureLabelRecord,
    geometry_angle_degrees,
    load_capture_candidates,
    load_label_records,
    write_label_records,
)
from embedded_gauge_reading_tinyml.capture_labeling import load_image_array  # noqa: E402

WINDOW_NAME: Final[str] = "Board Center/Tip Labeler"
DISPLAY_WIDTH: Final[int] = 1440
DISPLAY_HEIGHT: Final[int] = 980
DEFAULT_INPUT: Path = PROJECT_ROOT.parent / "tmp" / "board_center_tip_review_batch.csv"
DEFAULT_OUTPUT: Path = PROJECT_ROOT.parent / "tmp" / "board_center_tip_labels.csv"
DEFAULT_LIMIT: int = 50


@dataclass(slots=True)
class DisplayFrame:
    """One rendered image plus the coordinate mapping used by the GUI."""

    bgr_image: np.ndarray
    scale: float
    offset_x: int
    offset_y: int


class BoardCenterTipLabelApp:
    """OpenCV annotation loop for center and tip labels."""

    def __init__(
        self,
        candidates: list[CaptureCandidate],
        *,
        output_csv: Path,
        existing_records: dict[str, CaptureLabelRecord],
    ) -> None:
        self.candidates = candidates
        self.output_csv = output_csv
        self.records: dict[str, CaptureLabelRecord] = dict(existing_records)
        self.current_index = self._find_start_index()
        self.point_mode: Literal["center", "tip"] = "center"
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
                self._set_center_mode()
            elif key == ord("t"):
                self._set_tip_mode()
            elif key == ord("x"):
                self._clear_active_point()
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
            if not record.has_center or not record.has_tip:
                if record.quality_flag.strip().lower() != "exclude":
                    return index
        return 0

    def _current_candidate(self) -> CaptureCandidate | None:
        """Return the active candidate."""

        if self.current_index < 0 or self.current_index >= len(self.candidates):
            return None
        return self.candidates[self.current_index]

    def _current_record(self, candidate: CaptureCandidate) -> CaptureLabelRecord:
        """Return or create the current record."""

        key = candidate.image_path.as_posix()
        record = self.records.get(key)
        if record is None:
            return CaptureLabelRecord(
                image_path=candidate.image_path,
                source_width=candidate.source_width,
                source_height=candidate.source_height,
                origin_manifest=candidate.origin_manifest,
            )
        return CaptureLabelRecord(
            image_path=candidate.image_path,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
            center_x_source=record.center_x_source,
            center_y_source=record.center_y_source,
            tip_x_source=record.tip_x_source,
            tip_y_source=record.tip_y_source,
            temperature_c=record.temperature_c,
            label_quality=record.label_quality,
            quality_flag=record.quality_flag,
            notes=record.notes,
            label_source=record.label_source,
            origin_manifest=record.origin_manifest or candidate.origin_manifest,
        )

    def _render_candidate(
        self,
        candidate: CaptureCandidate,
        record: CaptureLabelRecord,
    ) -> DisplayFrame:
        """Load the image and paint the persistent and temporary overlays."""

        rgb = np.asarray(
            load_image_array(
                candidate.image_path,
                source_width=candidate.source_width,
                source_height=candidate.source_height,
            ),
            dtype=np.uint8,
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame = self._fit_frame(bgr)

        center = self._point_to_canvas(record.center_x_source, record.center_y_source, frame)
        tip = self._point_to_canvas(record.tip_x_source, record.tip_y_source, frame)

        if center is not None and tip is not None:
            cv2.line(frame.bgr_image, center, tip, (0, 215, 255), 2, cv2.LINE_AA)
        if center is not None:
            cv2.drawMarker(
                frame.bgr_image,
                center,
                (80, 217, 107),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        if tip is not None:
            cv2.drawMarker(
                frame.bgr_image,
                tip,
                (60, 60, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=18,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

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

    def _canvas_to_source(self, x: int, y: int, frame: DisplayFrame) -> tuple[float, float]:
        """Convert a canvas coordinate back to source image space."""

        candidate = self._current_candidate()
        if candidate is None:
            return 0.0, 0.0
        source_x = (float(x) - float(frame.offset_x)) / frame.scale
        source_y = (float(y) - float(frame.offset_y)) / frame.scale
        return (
            _clamp(source_x, 0.0, float(candidate.source_width)),
            _clamp(source_y, 0.0, float(candidate.source_height)),
        )

    def _point_to_canvas(
        self,
        x: float | None,
        y: float | None,
        frame: DisplayFrame,
    ) -> tuple[int, int] | None:
        """Convert an optional source point into canvas space."""

        if x is None or y is None:
            return None
        return (
            int(round(float(x) * frame.scale + frame.offset_x)),
            int(round(float(y) * frame.scale + frame.offset_y)),
        )

    def _draw_overlay_text(
        self,
        image: np.ndarray,
        candidate: CaptureCandidate,
        record: CaptureLabelRecord,
    ) -> None:
        """Paint useful status text onto the image."""

        center_text = "-" if not record.has_center else f"({record.center_x_source:.1f}, {record.center_y_source:.1f})"
        tip_text = "-" if not record.has_tip else f"({record.tip_x_source:.1f}, {record.tip_y_source:.1f})"
        angle_text = "-"
        angle = geometry_angle_degrees(record)
        if angle is not None:
            angle_text = f"{angle:.1f} deg"

        lines = [
            f"{self.current_index + 1}/{len(self.candidates)}  {candidate.image_path.name}",
            f"mode: {self.point_mode}  center: {center_text}  tip: {tip_text}",
            f"angle: {angle_text}  flag: {record.quality_flag or 'review'}",
            "click=set point  c=center  t=tip  s=save  space=save+next  p/n=prev/next  x=clear  k=skip  q=quit",
        ]
        y = 28
        for line in lines:
            cv2.putText(
                image,
                line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            y += 26

    def _apply_point(self, candidate: CaptureCandidate, source_x: float, source_y: float) -> None:
        """Update the current record with the active point mode."""

        record = self._current_record(candidate)
        if self.point_mode == "center":
            updated = record.with_geometry(
                center_x_source=source_x,
                center_y_source=source_y,
                tip_x_source=record.tip_x_source,
                tip_y_source=record.tip_y_source,
            )
        else:
            updated = record.with_geometry(
                center_x_source=record.center_x_source,
                center_y_source=record.center_y_source,
                tip_x_source=source_x,
                tip_y_source=source_y,
            )
        self.records[candidate.image_path.as_posix()] = updated
        self._save_records()

    def _clear_active_point(self) -> None:
        """Clear the point for the current mode."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        record = self._current_record(candidate)
        if self.point_mode == "center":
            updated = record.with_geometry(
                center_x_source=None,
                center_y_source=None,
                tip_x_source=record.tip_x_source,
                tip_y_source=record.tip_y_source,
            )
        else:
            updated = record.with_geometry(
                center_x_source=record.center_x_source,
                center_y_source=record.center_y_source,
                tip_x_source=None,
                tip_y_source=None,
            )
        self.records[candidate.image_path.as_posix()] = updated
        self._save_records()

    def _set_center_mode(self) -> None:
        """Switch the active point mode to center."""

        self.point_mode = "center"

    def _set_tip_mode(self) -> None:
        """Switch the active point mode to tip."""

        self.point_mode = "tip"

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object | None) -> None:
        """Handle click-to-place point interactions."""

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        candidate = self._current_candidate()
        if candidate is None:
            return

        frame = self._render_candidate(candidate, self._current_record(candidate))
        source_x, source_y = self._canvas_to_source(x, y, frame)
        self._apply_point(candidate, source_x, source_y)
        cv2.imshow(WINDOW_NAME, self._render_candidate(candidate, self._current_record(candidate)).bgr_image)

    def _save_records(self) -> None:
        """Write the current records to disk in candidate order."""

        ordered_records: list[CaptureLabelRecord] = []
        for candidate in self.candidates:
            record = self.records.get(candidate.image_path.as_posix())
            if record is not None:
                ordered_records.append(record)
        write_label_records(self.output_csv, ordered_records)

    def _save_current(self) -> None:
        """Mark the current image as reviewed and save the CSV."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        record = self._current_record(candidate)
        self.records[candidate.image_path.as_posix()] = record.with_quality_flag("review")
        self._save_records()

    def _skip_current(self) -> None:
        """Mark the current candidate as excluded."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        self.records[candidate.image_path.as_posix()] = CaptureLabelRecord(
            image_path=candidate.image_path,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
            center_x_source=None,
            center_y_source=None,
            tip_x_source=None,
            tip_y_source=None,
            temperature_c=None,
            label_quality="manual",
            quality_flag="exclude",
            notes="",
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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Label center and tip points for board captures.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Review batch CSV to label. Defaults to tmp/board_center_tip_review_batch.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV with center/tip labels.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of images to load from the review batch.",
    )
    return parser.parse_args()


def main() -> None:
    """Launch the board center/tip labeler."""

    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Review batch not found: {args.input}. Run prepare_board_center_tip_review_batch.py first."
        )

    candidates = load_capture_candidates(
        args.input,
        include_derivatives=False,
        recursive=False,
    )
    if args.limit > 0:
        candidates = candidates[: int(args.limit)]
    if not candidates:
        raise ValueError(f"No reviewable board captures were found in {args.input}.")

    existing_records = load_label_records(args.output)
    app = BoardCenterTipLabelApp(candidates, output_csv=args.output, existing_records=existing_records)
    app.run()


if __name__ == "__main__":
    main()
