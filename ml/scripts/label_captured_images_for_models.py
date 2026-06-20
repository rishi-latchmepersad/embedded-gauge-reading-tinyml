#!/usr/bin/env python3
"""Interactive labeler for captured gauge images.

This tool lets us review or add center, tip, temperature, and quality labels
for the captured-image training sets used by the combined center + SimCC model.
The saved CSV is compatible with the grouped manifest builder.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as exc:  # pragma: no cover - GUI import failures are runtime only.
    raise RuntimeError(
        "Tkinter is required for the labeling GUI."
    ) from exc

from PIL import Image, ImageDraw, ImageTk

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# The labeler does not need TensorFlow logs while it is just loading helper code.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from embedded_gauge_reading_tinyml.capture_labeling import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_CSV,
    CaptureCandidate,
    CaptureLabelRecord,
    angle_difference_degrees,
    derive_true_angle_degrees,
    geometry_angle_degrees,
    load_capture_candidates,
    load_image_array,
    load_label_records,
    map_source_point_to_canvas_norm,
    write_label_records,
)
from embedded_gauge_reading_tinyml.firmware_preprocessing import (  # noqa: E402
    firmware_training_crop_box,
)
DISPLAY_WIDTH: int = 1100
DISPLAY_HEIGHT: int = 820
DEFAULT_POINT_MODE: Literal["center", "tip"] = "center"
QUALITY_OPTIONS: tuple[str, ...] = ("review", "clean", "partial", "exclude")


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp ``value`` into the requested inclusive range."""

    return float(max(lower, min(value, upper)))


def _format_point(point: tuple[float, float] | None) -> str:
    """Format a point for the status bar."""

    if point is None:
        return "-"
    return f"({point[0]:.1f}, {point[1]:.1f})"


def _format_angle(angle: float | None) -> str:
    """Format an angle for the status bar."""

    if angle is None:
        return "-"
    return f"{angle:.2f} deg"


def _canonical_key(image_path: Path) -> str:
    """Return the path key used by the review CSV."""

    return image_path.as_posix()


class CaptureLabelApp:
    """Tkinter annotation window for captured-image geometry labels."""

    def __init__(
        self,
        root: tk.Tk,
        candidates: list[CaptureCandidate],
        *,
        output_csv: Path,
        existing_records: dict[str, CaptureLabelRecord],
    ) -> None:
        self.root = root
        self.candidates = candidates
        self.output_csv = output_csv
        self.existing_records = existing_records
        self.records: dict[str, CaptureLabelRecord] = dict(existing_records)
        self.current_index = self._find_start_index()
        self.point_mode: Literal["center", "tip"] = DEFAULT_POINT_MODE
        self._photo: ImageTk.PhotoImage | None = None
        self._display_image_size: tuple[int, int] = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        self._display_offset: tuple[int, int] = (0, 0)
        self._display_scale: float = 1.0
        self._current_canvas_image: Image.Image | None = None

        self.temperature_var = tk.StringVar()
        self.quality_var = tk.StringVar(value="review")
        self.label_quality_var = tk.StringVar(value="manual")
        self.status_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.path_var = tk.StringVar()
        self.mode_var = tk.StringVar(value=self.point_mode)
        self.angle_warning_var = tk.StringVar()

        self._build_ui()
        self._bind_keys()
        self._load_current_candidate()

    def _build_ui(self) -> None:
        """Create the window layout."""

        self.root.title("Captured Image Labeler")
        self.root.geometry(f"{DISPLAY_WIDTH + 420}x{DISPLAY_HEIGHT + 40}")
        self.root.minsize(1200, 820)

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(outer, width=390)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self.canvas = tk.Canvas(
            left,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            background="#111111",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        title = ttk.Label(right, text="Captured Image Labels", font=("Segoe UI", 15, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        ttk.Label(right, textvariable=self.path_var, wraplength=360, justify=tk.LEFT).pack(
            anchor="w", pady=(0, 8)
        )

        self.status_label = ttk.Label(right, textvariable=self.status_var, wraplength=360)
        self.status_label.pack(anchor="w", pady=(0, 8))

        self.summary_label = ttk.Label(right, textvariable=self.summary_var, wraplength=360)
        self.summary_label.pack(anchor="w", pady=(0, 8))

        self.warning_label = ttk.Label(
            right,
            textvariable=self.angle_warning_var,
            wraplength=360,
            foreground="#b04040",
        )
        self.warning_label.pack(anchor="w", pady=(0, 8))

        mode_box = ttk.LabelFrame(right, text="Point Mode", padding=8)
        mode_box.pack(fill=tk.X, pady=(4, 8))
        ttk.Label(mode_box, textvariable=self.mode_var).pack(anchor="w")
        ttk.Button(mode_box, text="Center Mode", command=self._set_center_mode).pack(
            fill=tk.X, pady=(6, 2)
        )
        ttk.Button(mode_box, text="Tip Mode", command=self._set_tip_mode).pack(fill=tk.X)

        label_box = ttk.LabelFrame(right, text="Label Fields", padding=8)
        label_box.pack(fill=tk.X, pady=(4, 8))

        self.center_var = tk.StringVar(value="-")
        self.tip_var = tk.StringVar(value="-")
        self.canvas_center_var = tk.StringVar(value="-")
        self.canvas_tip_var = tk.StringVar(value="-")

        self._add_field(label_box, "Center", self.center_var)
        self._add_field(label_box, "Tip", self.tip_var)
        self._add_field(label_box, "Canvas center", self.canvas_center_var)
        self._add_field(label_box, "Canvas tip", self.canvas_tip_var)

        temp_box = ttk.LabelFrame(right, text="Temperature", padding=8)
        temp_box.pack(fill=tk.X, pady=(4, 8))
        temp_row = ttk.Frame(temp_box)
        temp_row.pack(fill=tk.X)
        self.temperature_entry = ttk.Entry(temp_row, textvariable=self.temperature_var, width=12)
        self.temperature_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.temperature_entry.bind("<Return>", lambda _event: self._apply_temperature())
        ttk.Button(temp_row, text="Apply", command=self._apply_temperature).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        quality_row = ttk.Frame(temp_box)
        quality_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(quality_row, text="Flag").pack(side=tk.LEFT)
        self.quality_combo = ttk.Combobox(
            quality_row,
            textvariable=self.quality_var,
            values=QUALITY_OPTIONS,
            state="readonly",
            width=12,
        )
        self.quality_combo.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(quality_row, text="Quality").pack(side=tk.LEFT, padx=(12, 0))
        self.label_quality_entry = ttk.Entry(
            quality_row,
            textvariable=self.label_quality_var,
            width=12,
        )
        self.label_quality_entry.pack(side=tk.LEFT, padx=(8, 0))

        notes_box = ttk.LabelFrame(right, text="Notes", padding=8)
        notes_box.pack(fill=tk.BOTH, expand=False, pady=(4, 8))
        self.notes_text = tk.Text(notes_box, height=6, wrap=tk.WORD)
        self.notes_text.pack(fill=tk.BOTH, expand=True)
        self.notes_text.bind("<FocusOut>", lambda _event: self._sync_notes_from_widget())

        button_box = ttk.Frame(right)
        button_box.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(button_box, text="Save", command=self.save_current).grid(row=0, column=0, sticky="ew")
        ttk.Button(button_box, text="Prev", command=self.previous_image).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(button_box, text="Next", command=self.next_image).grid(
            row=0, column=2, sticky="ew"
        )
        ttk.Button(button_box, text="Clear", command=self.clear_geometry).grid(
            row=1, column=0, sticky="ew", pady=(4, 0)
        )
        ttk.Button(button_box, text="Exclude", command=self.mark_exclude).grid(
            row=1, column=1, sticky="ew", padx=4, pady=(4, 0)
        )
        ttk.Button(button_box, text="Skip", command=self.skip_image).grid(
            row=1, column=2, sticky="ew", pady=(4, 0)
        )
        for column in range(3):
            button_box.columnconfigure(column, weight=1)

        help_box = ttk.LabelFrame(right, text="Shortcuts", padding=8)
        help_box.pack(fill=tk.X, pady=(8, 0))
        help_text = (
            "Click = set active point\n"
            "c/t = center or tip mode\n"
            "space = save and next\n"
            "s = save\n"
            "p/n = previous or next\n"
            "r = clear geometry\n"
            "x = exclude and next"
        )
        ttk.Label(help_box, text=help_text, justify=tk.LEFT).pack(anchor="w")

    def _add_field(self, parent: ttk.Widget, label: str, var: tk.StringVar) -> None:
        """Add one read-only label row to the side panel."""

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _bind_keys(self) -> None:
        """Bind keyboard shortcuts for quick labeling."""

        self.root.bind("c", lambda _event: self._set_center_mode())
        self.root.bind("t", lambda _event: self._set_tip_mode())
        self.root.bind("s", lambda _event: self.save_current())
        self.root.bind("n", lambda _event: self.next_image())
        self.root.bind("p", lambda _event: self.previous_image())
        self.root.bind("r", lambda _event: self.clear_geometry())
        self.root.bind("x", lambda _event: self.mark_exclude())
        self.root.bind("<space>", lambda _event: self.save_current(advance=1))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _find_start_index(self) -> int:
        """Start on the first unlabeled candidate when possible."""

        for index, candidate in enumerate(self.candidates):
            if _canonical_key(candidate.image_path) not in self.records:
                return index
        return 0

    def _blank_record(self, candidate: CaptureCandidate) -> CaptureLabelRecord:
        """Create an empty record for the current candidate image."""

        return CaptureLabelRecord(
            image_path=candidate.image_path,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
            origin_manifest=candidate.origin_manifest,
        )

    def _candidate_for_index(self, index: int) -> CaptureCandidate | None:
        """Return the candidate at ``index`` if it exists."""

        if index < 0 or index >= len(self.candidates):
            return None
        return self.candidates[index]

    def _current_candidate(self) -> CaptureCandidate | None:
        """Return the currently selected candidate."""

        return self._candidate_for_index(self.current_index)

    def _load_current_candidate(self) -> None:
        """Load the current record into the widgets and canvas."""

        candidate = self._current_candidate()
        if candidate is None:
            self.status_var.set("No images available.")
            self.path_var.set(self.output_csv.as_posix())
            return

        current_key = _canonical_key(candidate.image_path)
        record = self.records.get(current_key)
        if record is None:
            record = candidate.seed_record or self._blank_record(candidate)
        record = self._normalize_record(candidate, record)
        self.records[current_key] = record

        self._sync_widgets_from_record(record)
        self._render_candidate(candidate, record)
        self.root.title(
            f"Captured Image Labeler - {candidate.image_path.name} "
            f"({self.current_index + 1}/{len(self.candidates)})"
        )

    def _normalize_record(
        self,
        candidate: CaptureCandidate,
        record: CaptureLabelRecord,
    ) -> CaptureLabelRecord:
        """Make sure a record matches the currently loaded candidate."""

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

    def _sync_widgets_from_record(self, record: CaptureLabelRecord) -> None:
        """Populate all widgets from the active record."""

        self.temperature_var.set("" if record.temperature_c is None else f"{record.temperature_c:.2f}")
        self.quality_var.set(record.quality_flag or "review")
        self.label_quality_var.set(record.label_quality or "manual")
        self.notes_text.delete("1.0", tk.END)
        if record.notes:
            self.notes_text.insert("1.0", record.notes)

    def _sync_notes_from_widget(self) -> None:
        """Copy the notes widget text back into the current record."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        record = self._current_record()
        if record is None:
            record = self._blank_record(candidate)
        notes = self.notes_text.get("1.0", tk.END).strip()
        self.records[_canonical_key(candidate.image_path)] = replace(record, notes=notes)

    def _current_record(self) -> CaptureLabelRecord | None:
        """Return the active record after widget sync."""

        candidate = self._current_candidate()
        if candidate is None:
            return None
        key = _canonical_key(candidate.image_path)
        return self.records.get(key)

    def _apply_temperature(self) -> None:
        """Update the active record from the temperature entry."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        temperature_text = self.temperature_var.get().strip()
        temperature: float | None
        if temperature_text:
            try:
                temperature = float(temperature_text)
            except ValueError:
                self._set_status(f"Invalid temperature: {temperature_text!r}")
                return
        else:
            temperature = None
        record = self._current_record() or self._blank_record(candidate)
        record = replace(record, temperature_c=temperature)
        self.records[_canonical_key(candidate.image_path)] = record
        self._render_candidate(candidate, record)
        self._update_status_and_overlay(record)

    def _set_center_mode(self) -> None:
        """Switch the active click target to the gauge center."""

        self.point_mode = "center"
        self.mode_var.set("center")
        self._set_status("Click on the gauge center.")

    def _set_tip_mode(self) -> None:
        """Switch the active click target to the needle tip."""

        self.point_mode = "tip"
        self.mode_var.set("tip")
        self._set_status("Click on the needle tip.")

    def _set_status(self, text: str) -> None:
        """Update the status bar."""

        self.status_var.set(text)

    def _render_candidate(self, candidate: CaptureCandidate, record: CaptureLabelRecord) -> None:
        """Load the image, paint overlays, and refresh the canvas."""

        image_array = load_image_array(
            candidate.image_path,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
        )
        pil_image = Image.fromarray(image_array, mode="RGB")
        display_image, scale, offset_x, offset_y = self._fit_image(pil_image)
        draw = ImageDraw.Draw(display_image)

        crop_x_min, crop_y_min, crop_x_max, crop_y_max = firmware_training_crop_box(
            candidate.source_width,
            candidate.source_height,
        )
        crop_box = (
            self._source_to_display_x(crop_x_min, scale, offset_x),
            self._source_to_display_y(crop_y_min, scale, offset_y),
            self._source_to_display_x(crop_x_max, scale, offset_x),
            self._source_to_display_y(crop_y_max, scale, offset_y),
        )
        self._draw_rectangle(draw, crop_box, outline="#ffb347", width=3)

        center_xy = self._center_xy(record)
        tip_xy = self._tip_xy(record)
        if center_xy is not None:
            center_canvas = self._source_point_to_display(center_xy, scale, offset_x, offset_y)
            self._draw_crosshair(draw, center_canvas, outline="#49d17d")
        if tip_xy is not None:
            tip_canvas = self._source_point_to_display(tip_xy, scale, offset_x, offset_y)
            self._draw_crosshair(draw, tip_canvas, outline="#ff6666")
        if center_xy is not None and tip_xy is not None:
            self._draw_line(
                draw,
                self._source_point_to_display(center_xy, scale, offset_x, offset_y),
                self._source_point_to_display(tip_xy, scale, offset_x, offset_y),
                outline="#ffd200",
            )

        self._photo = ImageTk.PhotoImage(display_image)
        self._current_canvas_image = display_image
        self._display_scale = scale
        self._display_offset = (offset_x, offset_y)
        self._display_image_size = display_image.size

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._draw_overlay_text(candidate, record)
        self._update_status_and_overlay(record)

    def _draw_overlay_text(self, candidate: CaptureCandidate, record: CaptureLabelRecord) -> None:
        """Draw lightweight overlay text on the canvas."""

        center_xy = self._center_xy(record)
        tip_xy = self._tip_xy(record)
        geometry_angle = geometry_angle_degrees(record)
        true_angle, angle_source = derive_true_angle_degrees(record)
        temp_angle = None
        if record.temperature_c is not None:
            from embedded_gauge_reading_tinyml.capture_labeling import temperature_to_true_angle_degrees

            temp_angle = temperature_to_true_angle_degrees(record.temperature_c)

        overlay_lines = [
            candidate.image_path.name,
            f"center: {_format_point(center_xy)}",
            f"tip: {_format_point(tip_xy)}",
            f"geometry angle: {_format_angle(geometry_angle)}",
            f"true angle: {_format_angle(true_angle)} [{angle_source}]",
        ]
        if temp_angle is not None and geometry_angle is not None:
            overlay_lines.append(
                f"angle delta: {angle_difference_degrees(temp_angle, geometry_angle):.2f} deg"
            )
        x = 14
        y = 14
        for line in overlay_lines:
            self.canvas.create_text(
                x,
                y,
                text=line,
                anchor="nw",
                fill="#f0f0f0",
                font=("Segoe UI", 11, "bold"),
            )
            y += 18

    def _update_status_and_overlay(self, record: CaptureLabelRecord) -> None:
        """Refresh the text fields that summarize the active image."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        self.path_var.set(
            f"{candidate.image_path.as_posix()}\n"
            f"source: {candidate.source_width} x {candidate.source_height}\n"
            f"origin: {candidate.origin_manifest}"
        )
        center_xy = self._center_xy(record)
        tip_xy = self._tip_xy(record)
        center_canvas = self._center_canvas_xy(record)
        tip_canvas = self._tip_canvas_xy(record)
        geometry_angle = geometry_angle_degrees(record)
        true_angle, angle_source = derive_true_angle_degrees(record)
        temp_angle = None
        if record.temperature_c is not None:
            from embedded_gauge_reading_tinyml.capture_labeling import temperature_to_true_angle_degrees

            temp_angle = temperature_to_true_angle_degrees(record.temperature_c)
        self.center_var.set(_format_point(center_xy))
        self.tip_var.set(_format_point(tip_xy))
        self.canvas_center_var.set(_format_point(center_canvas))
        self.canvas_tip_var.set(_format_point(tip_canvas))
        summary = (
            f"Temperature: {record.temperature_c if record.temperature_c is not None else '-'} | "
            f"Flag: {record.quality_flag} | Quality: {record.label_quality} | "
            f"Mode: {self.point_mode}"
        )
        self.summary_var.set(summary)
        status = (
            f"{self.current_index + 1}/{len(self.candidates)} | "
            f"center={_format_point(center_xy)} tip={_format_point(tip_xy)} | "
            f"geom={_format_angle(geometry_angle)} true={_format_angle(true_angle)} ({angle_source})"
        )
        if record.temperature_c is not None:
            status += f" | temp={record.temperature_c:.2f}C"
        if temp_angle is not None and geometry_angle is not None:
            delta = angle_difference_degrees(temp_angle, geometry_angle)
            status += f" | temp-vs-geom={delta:.2f} deg"
            if delta > 10.0:
                self.angle_warning_var.set(
                    "Warning: temperature-derived angle and geometry angle disagree by "
                    f"{delta:.2f} deg."
                )
            else:
                self.angle_warning_var.set("")
        else:
            self.angle_warning_var.set("")
        self.status_var.set(status)

    def _fit_image(self, image: Image.Image) -> tuple[Image.Image, float, int, int]:
        """Fit the image into the drawing area without distorting its aspect ratio."""

        source_width, source_height = image.size
        scale = min(
            DISPLAY_WIDTH / max(source_width, 1),
            DISPLAY_HEIGHT / max(source_height, 1),
        )
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), (17, 17, 17))
        offset_x = max(0, (DISPLAY_WIDTH - resized_width) // 2)
        offset_y = max(0, (DISPLAY_HEIGHT - resized_height) // 2)
        canvas.paste(resized, (offset_x, offset_y))
        return canvas, scale, offset_x, offset_y

    def _source_point_to_display(
        self,
        point_xy: tuple[float, float],
        scale: float,
        offset_x: int,
        offset_y: int,
    ) -> tuple[float, float]:
        """Map a source-space point into display-space pixels."""

        x, y = point_xy
        return (x * scale + offset_x, y * scale + offset_y)

    def _source_to_display_x(self, x: float, scale: float, offset_x: int) -> float:
        """Map a source x coordinate to display space."""

        return x * scale + offset_x

    def _source_to_display_y(self, y: float, scale: float, offset_y: int) -> float:
        """Map a source y coordinate to display space."""

        return y * scale + offset_y

    def _draw_crosshair(
        self,
        draw: ImageDraw.ImageDraw,
        point_xy: tuple[float, float],
        *,
        outline: str,
    ) -> None:
        """Draw a compact crosshair around a label point."""

        x, y = point_xy
        radius = 7
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=outline, width=3)
        draw.line((x - 11, y, x + 11, y), fill=outline, width=2)
        draw.line((x, y - 11, x, y + 11), fill=outline, width=2)

    def _draw_rectangle(
        self,
        draw: ImageDraw.ImageDraw,
        box_xyxy: tuple[float, float, float, float],
        *,
        outline: str,
        width: int,
    ) -> None:
        """Draw the firmware crop box on the preview image."""

        draw.rectangle(box_xyxy, outline=outline, width=width)

    def _draw_line(
        self,
        draw: ImageDraw.ImageDraw,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        outline: str,
    ) -> None:
        """Draw the center-to-tip line used by the SimCC head."""

        draw.line((*start_xy, *end_xy), fill=outline, width=2)

    def _canvas_to_source(
        self,
        canvas_x: float,
        canvas_y: float,
    ) -> tuple[float, float]:
        """Convert a canvas click back into source-space coordinates."""

        offset_x, offset_y = self._display_offset
        source_x = (canvas_x - float(offset_x)) / max(self._display_scale, 1e-6)
        source_y = (canvas_y - float(offset_y)) / max(self._display_scale, 1e-6)
        candidate = self._current_candidate()
        if candidate is None:
            return 0.0, 0.0
        source_x = _clamp(source_x, 0.0, float(candidate.source_width - 1))
        source_y = _clamp(source_y, 0.0, float(candidate.source_height - 1))
        return source_x, source_y

    def _center_xy(self, record: CaptureLabelRecord) -> tuple[float, float] | None:
        """Return the center point as a tuple when it exists."""

        if not record.has_center:
            return None
        return float(record.center_x_source), float(record.center_y_source)

    def _tip_xy(self, record: CaptureLabelRecord) -> tuple[float, float] | None:
        """Return the tip point as a tuple when it exists."""

        if not record.has_tip:
            return None
        return float(record.tip_x_source), float(record.tip_y_source)

    def _center_canvas_xy(self, record: CaptureLabelRecord) -> tuple[float, float] | None:
        """Return the center point in canvas coordinates."""

        center_xy = self._center_xy(record)
        if center_xy is None:
            return None
        crop_box = firmware_training_crop_box(record.source_width, record.source_height)
        center_norm = map_source_point_to_canvas_norm(
            center_xy,
            crop_box_xyxy=crop_box,
            source_width=record.source_width,
            source_height=record.source_height,
        )
        return (center_norm[0] * 224.0, center_norm[1] * 224.0)

    def _tip_canvas_xy(self, record: CaptureLabelRecord) -> tuple[float, float] | None:
        """Return the tip point in canvas coordinates."""

        tip_xy = self._tip_xy(record)
        if tip_xy is None:
            return None
        crop_box = firmware_training_crop_box(record.source_width, record.source_height)
        tip_norm = map_source_point_to_canvas_norm(
            tip_xy,
            crop_box_xyxy=crop_box,
            source_width=record.source_width,
            source_height=record.source_height,
        )
        return (tip_norm[0] * 224.0, tip_norm[1] * 224.0)

    def _on_canvas_click(self, event: tk.Event[tk.Misc]) -> None:
        """Handle mouse clicks on the image canvas."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        source_x, source_y = self._canvas_to_source(float(event.x), float(event.y))
        record = self._current_record() or self._blank_record(candidate)
        if self.point_mode == "center":
            record = record.with_geometry(
                center_x_source=source_x,
                center_y_source=source_y,
                tip_x_source=record.tip_x_source,
                tip_y_source=record.tip_y_source,
            )
            self.point_mode = "tip"
            self.mode_var.set(self.point_mode)
        else:
            record = record.with_geometry(
                center_x_source=record.center_x_source,
                center_y_source=record.center_y_source,
                tip_x_source=source_x,
                tip_y_source=source_y,
            )
            self.point_mode = "center"
            self.mode_var.set(self.point_mode)
        self.records[_canonical_key(candidate.image_path)] = record
        self._sync_widgets_from_record(record)
        self._render_candidate(candidate, record)

    def _update_record_from_widgets(self) -> CaptureLabelRecord | None:
        """Pull the widget state back into the active record."""

        candidate = self._current_candidate()
        if candidate is None:
            return None
        record = self._current_record() or self._blank_record(candidate)
        temperature_text = self.temperature_var.get().strip()
        temperature: float | None = None
        if temperature_text:
            try:
                temperature = float(temperature_text)
            except ValueError:
                self._set_status(f"Invalid temperature: {temperature_text!r}")
                return None
        notes = self.notes_text.get("1.0", tk.END).strip()
        record = CaptureLabelRecord(
            image_path=candidate.image_path,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
            center_x_source=record.center_x_source,
            center_y_source=record.center_y_source,
            tip_x_source=record.tip_x_source,
            tip_y_source=record.tip_y_source,
            temperature_c=temperature,
            label_quality=self.label_quality_var.get().strip() or "manual",
            quality_flag=self.quality_var.get().strip() or "review",
            notes=notes,
            label_source=record.label_source,
            origin_manifest=record.origin_manifest or candidate.origin_manifest,
        )
        self.records[_canonical_key(candidate.image_path)] = record
        return record

    def save_current(self, advance: int = 0) -> None:
        """Save the current record to disk and optionally move the selection."""

        record = self._update_record_from_widgets()
        if record is None:
            return
        candidate = self._current_candidate()
        if candidate is not None:
            self._render_candidate(candidate, record)
        current_status = self.status_var.get()
        write_label_records(self.output_csv, list(self.records.values()))
        if advance == 0:
            self._set_status(f"Saved {self.output_csv.name} | {current_status}")
        if advance != 0:
            self._move_index(advance)

    def _move_index(self, delta: int) -> None:
        """Move the cursor by ``delta`` images, saving before the move."""

        if not self.candidates:
            return
        self.current_index = max(0, min(self.current_index + delta, len(self.candidates) - 1))
        self._load_current_candidate()

    def next_image(self) -> None:
        """Save the current record and move to the next image."""

        self.save_current(advance=1)

    def previous_image(self) -> None:
        """Save the current record and move to the previous image."""

        self.save_current(advance=-1)

    def clear_geometry(self) -> None:
        """Clear the center and tip points for the active image."""

        candidate = self._current_candidate()
        if candidate is None:
            return
        record = self._current_record() or self._blank_record(candidate)
        record = record.with_geometry(
            center_x_source=None,
            center_y_source=None,
            tip_x_source=None,
            tip_y_source=None,
        )
        self.records[_canonical_key(candidate.image_path)] = record
        self._sync_widgets_from_record(record)
        self._render_candidate(candidate, record)
        self._set_status("Cleared geometry labels.")

    def mark_exclude(self) -> None:
        """Mark the current image as excluded and advance."""

        self.quality_var.set("exclude")
        self.save_current(advance=1)

    def skip_image(self) -> None:
        """Advance without modifying the current record."""

        self._move_index(1)

    def _on_close(self) -> None:
        """Save the current state before exiting."""

        self.save_current()
        self.root.destroy()


def _default_input_path() -> Path:
    """Choose the most useful default input source for the labeler."""

    if DEFAULT_MANIFEST_PATH.exists():
        return DEFAULT_MANIFEST_PATH
    return PROJECT_ROOT / "data" / "captured_images"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Label captured gauge images with center, tip, and temperature values."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input manifest or directory. Defaults to labelled_captured_images.json when present.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV file where the reviewed labels should be written.",
    )
    parser.add_argument(
        "--include-derivatives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep preview and derived variants instead of filtering them out.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scan directories recursively when the input is a directory.",
    )
    parser.add_argument(
        "--raw-width",
        type=int,
        default=None,
        help="Fallback width for raw captures when the size cannot be inferred.",
    )
    parser.add_argument(
        "--raw-height",
        type=int,
        default=None,
        help="Fallback height for raw captures when the size cannot be inferred.",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild labelled_captured_images.json after saving the review CSV.",
    )
    return parser.parse_args()


def _rebuild_grouped_manifest(review_csv: Path) -> None:
    """Rebuild the grouped JSON manifest using the reviewed labels as an extra source."""

    import subprocess

    builder = PROJECT_ROOT / "scripts" / "build_labelled_captured_images_manifest.py"
    manifest = PROJECT_ROOT / "data" / "labelled_captured_images.json"
    command = [
        sys.executable,
        str(builder),
        "--output",
        str(manifest),
        "--extra-source",
        str(review_csv),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """Launch the label review GUI."""

    args = parse_args()
    input_path = args.input or _default_input_path()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    candidates = load_capture_candidates(
        input_path,
        include_derivatives=bool(args.include_derivatives),
        recursive=bool(args.recursive),
        raw_width_hint=args.raw_width,
        raw_height_hint=args.raw_height,
    )
    if not candidates:
        raise RuntimeError("No candidate images were found for labeling.")

    existing_records = load_label_records(args.output)
    root = tk.Tk()
    app = CaptureLabelApp(
        root,
        candidates,
        output_csv=args.output,
        existing_records=existing_records,
    )
    try:
        root.mainloop()
    finally:
        if args.rebuild_manifest and args.output.exists():
            _rebuild_grouped_manifest(args.output)


if __name__ == "__main__":
    main()
