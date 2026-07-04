#!/usr/bin/env python3
"""Generate synthetic gauge CD-crops with known pivot centers for centre-detector training.

Renders analog gauge dials on a large canvas, then applies the same
crop→resize→pad CD-crop pipeline the firmware uses.  This makes the synthetic
images visually match real CD-crops: the dial fills the full width and gray
bars appear on the top/bottom from the resize-with-pad step.

Each image has an exact, noise-free (cx_norm, cy_norm) label computed from
the known pivot position and the crop transform.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "heatmap_training"
OUT_IMAGES = DATA_DIR / "images"
OUT_IMAGES.mkdir(parents=True, exist_ok=True)

# CD-crop geometry (mirrors prepare_heatmap_training_data.py)
INPUT_SIZE: int = 224
TC_W: int = 155
TC_H: int = 123
PAD_COLOR: int = 128  # gray, matching firmware

NUM_SAMPLES: int = 800

# Large canvas: dial rendered here, then a 155x123 CD-crop is extracted
CANVAS: int = 400

# Gauge centre jitter on the large canvas (± px around canvas centre)
CTR_JITTER: float = 40.0

# OBB detection noise (px — wider = more diverse centre positions in CD-crop).
# cx_norm ≈ 0.5 - (obb_cx - gauge_cx) / 155, so ±25 px → range [0.34, 0.66].
OBB_NOISE: float = 25.0


def _resize_with_pad_scale() -> tuple[float, float, float]:
    """Return (scale, pad_x, pad_y) for 155×123 → 224×224 resize-with-pad."""
    scale = min(INPUT_SIZE / TC_W, INPUT_SIZE / TC_H)  # 224/155
    resized_w = TC_W * scale
    resized_h = TC_H * scale
    pad_x = (INPUT_SIZE - resized_w) * 0.5
    pad_y = (INPUT_SIZE - resized_h) * 0.5
    return scale, pad_x, pad_y


SCALE, PAD_X, PAD_Y = _resize_with_pad_scale()


def _gauge_to_crop_label(
    gauge_cx: float,
    gauge_cy: float,
    cd_x: int,
    cd_y: int,
) -> tuple[float, float]:
    """Map a gauge-centre pixel on the large canvas to CD-crop normalised label."""
    padded_cx = (gauge_cx - cd_x) * SCALE + PAD_X
    padded_cy = (gauge_cy - cd_y) * SCALE + PAD_Y
    return padded_cx / INPUT_SIZE, padded_cy / INPUT_SIZE


def _render_dial(
    canvas_size: int,
    cx: float,
    cy: float,
    radius: float,
    rng: np.random.Generator,
) -> Image.Image:
    """Draw a single analog gauge dial on a square canvas.

    The dial centre IS the needle pivot — (cx, cy) in canvas pixels.
    """
    img = Image.new("RGB", (canvas_size, canvas_size), (PAD_COLOR, PAD_COLOR, PAD_COLOR))
    draw = ImageDraw.Draw(img)
    r = radius

    # ---------- dial body ----------
    outer_fill = (235, 232, 226)
    inner_fill = (248, 246, 242)
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=outer_fill, outline=(35, 35, 35), width=4,
    )
    draw.ellipse(
        [cx - r * 0.87, cy - r * 0.87, cx + r * 0.87, cy + r * 0.87],
        fill=inner_fill, outline=(80, 80, 80), width=1,
    )

    # ---------- tick marks (lower arc 225°→315°, like real gauge) ----------
    start_deg = 225.0
    end_deg = 315.0
    for i, deg in enumerate(np.linspace(start_deg, end_deg, 19)):
        angle = math.radians(float(deg))
        is_major = i % 3 == 0
        t_inner = r * (0.72 if is_major else 0.78)
        t_outer = r * (0.92 if is_major else 0.87)
        x1 = cx + math.cos(angle) * t_inner
        y1 = cy - math.sin(angle) * t_inner
        x2 = cx + math.cos(angle) * t_outer
        y2 = cy - math.sin(angle) * t_outer
        draw.line((x1, y1, x2, y2), fill=(25, 25, 25), width=3 if is_major else 1)

    # ---------- labels ----------
    font = ImageFont.load_default()
    for deg, text in [(225.0, "-30"), (255.0, "0"), (285.0, "30"), (315.0, "50")]:
        angle = math.radians(deg)
        tx = cx + math.cos(angle) * r * 0.60 - 6
        ty = cy - math.sin(angle) * r * 0.60 - 5
        draw.text((tx, ty), text, fill=(30, 30, 30), font=font)

    # ---------- needle ----------
    needle_deg = float(rng.uniform(225, 315))
    needle_angle = math.radians(needle_deg)
    needle_len = r * 0.83
    nx = cx + math.cos(needle_angle) * needle_len
    ny = cy - math.sin(needle_angle) * needle_len
    draw.line((cx, cy, nx, ny), fill=(190, 30, 30), width=4)
    draw.line((cx, cy, nx, ny), fill=(220, 55, 55), width=2)

    # ---------- counterweight ----------
    tail_angle = needle_angle + math.pi
    tail_len = r * 0.16
    tx = cx + math.cos(tail_angle) * tail_len
    ty = cy - math.sin(tail_angle) * tail_len
    draw.line((cx, cy, tx, ty), fill=(60, 60, 60), width=4)

    # ---------- hub ----------
    hub_r = max(r * 0.05, 4.0)
    draw.ellipse(
        [cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r],
        fill=(30, 30, 30), outline=(10, 10, 10),
    )

    # ---------- glare ----------
    glare_angle = math.radians(float(rng.uniform(0, 360)))
    gx = cx + math.cos(glare_angle) * r * float(rng.uniform(0.35, 0.50))
    gy = cy - math.sin(glare_angle) * r * float(rng.uniform(0.35, 0.50))
    draw.ellipse([gx - 15, gy - 8, gx + 15, gy + 8], fill=(255, 255, 255))

    # ---------- vignette ----------
    draw.ellipse(
        [cx - r * 0.94, cy - r * 0.94, cx + r * 0.94, cy + r * 0.94],
        outline=(10, 10, 10), width=1,
    )

    # ---------- degrade ----------
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.array(img, dtype=np.int16)
    arr += rng.integers(-6, 7, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.75, 1.15)))
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.80, 1.10)))

    return img


def main() -> None:
    rng = np.random.default_rng(42)
    canvas_ctr = CANVAS / 2.0

    new_entries: list[dict] = []

    for idx in range(NUM_SAMPLES):
        # Gauge centre on the large canvas (slightly jittered)
        gauge_cx = float(rng.normal(canvas_ctr, CTR_JITTER / 3.0))
        gauge_cy = float(rng.normal(canvas_ctr, CTR_JITTER / 3.0))
        gauge_cx = float(np.clip(gauge_cx, canvas_ctr - CTR_JITTER, canvas_ctr + CTR_JITTER))
        gauge_cy = float(np.clip(gauge_cy, canvas_ctr - CTR_JITTER, canvas_ctr + CTR_JITTER))

        # Dial radius — fills most of the canvas so the CD-crop shows a large face
        radius = float(rng.uniform(CANVAS * 0.35, CANVAS * 0.42))

        # OBB centre (crop centre): gauge centre + detection noise
        obb_cx = gauge_cx + float(rng.normal(0, OBB_NOISE / 3.0))
        obb_cy = gauge_cy + float(rng.normal(0, OBB_NOISE / 3.0))

        # Clamp OBB to keep the crop rect inside the canvas
        half_w, half_h = TC_W / 2.0, TC_H / 2.0
        obb_cx = float(np.clip(obb_cx, half_w, CANVAS - half_w))
        obb_cy = float(np.clip(obb_cy, half_h, CANVAS - half_h))

        # Render the dial
        dial = _render_dial(CANVAS, gauge_cx, gauge_cy, radius, rng)

        # CD-crop top-left
        cd_x = int(round(obb_cx - half_w))
        cd_y = int(round(obb_cy - half_h))
        cd_x = max(0, min(cd_x, CANVAS - TC_W))
        cd_y = max(0, min(cd_y, CANVAS - TC_H))
        crop = dial.crop((cd_x, cd_y, cd_x + TC_W, cd_y + TC_H))  # 155×123

        # Resize + pad → 224×224
        resized_w = int(round(TC_W * SCALE))
        resized_h = int(round(TC_H * SCALE))
        crop_resized = crop.resize((resized_w, resized_h), Image.BILINEAR)
        canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (PAD_COLOR, PAD_COLOR, PAD_COLOR))
        x_off = int(round(PAD_X))
        y_off = int(round(PAD_Y))
        canvas.paste(crop_resized, (x_off, y_off))

        # Compute label in CD-crop space
        cx_norm, cy_norm = _gauge_to_crop_label(gauge_cx, gauge_cy, cd_x, cd_y)

        fname = f"cd_synthetic_{idx:05d}.png"
        out_path = OUT_IMAGES / fname
        canvas.save(str(out_path), format="PNG")

        new_entries.append({
            "image_path": f"images/{fname}",
            "cx_norm": round(cx_norm, 6),
            "cy_norm": round(cy_norm, 6),
            "split": "train",
            "source": "synthetic",
        })

    # ---- Merge with existing metadata (real entries only) ----
    meta_path = DATA_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    else:
        existing = []

    # Keep only real entries (pxl + board_manual), drop old synthetic
    real = [e for e in existing if e.get("source") != "synthetic"]

    combined = real + new_entries

    rng_shuffle = np.random.default_rng(42)
    fixed_test = [e for e in combined if e["split"] == "test"]
    rest = [e for e in combined if e["split"] != "test"]
    rng_shuffle.shuffle(rest)

    n_train = int(len(rest) * 0.80)
    n_val = int(len(rest) * 0.10)

    for i, entry in enumerate(rest):
        if i < n_train:
            entry["split"] = "train"
        elif i < n_train + n_val:
            entry["split"] = "val"
        else:
            entry["split"] = "test"

    final = rest + fixed_test

    with open(meta_path, "w") as f:
        json.dump(final, f, indent=2)

    # Report
    splits: dict[str, int] = {}
    sources: dict[str, int] = {}
    for e in final:
        splits[e["split"]] = splits.get(e["split"], 0) + 1
        sources[e["source"]] = sources.get(e["source"], 0) + 1
    print(f"Generated {NUM_SAMPLES} synthetic samples (CD-crop pipeline)")
    print(f"Total: {len(final)}  Splits: {splits}  Sources: {sources}")

    syn = [e for e in final if e["source"] == "synthetic"]
    cx_v = [e["cx_norm"] for e in syn]
    cy_v = [e["cy_norm"] for e in syn]
    print(f"Syn cx: mean={np.mean(cx_v):.4f} std={np.std(cx_v):.4f} range=[{min(cx_v):.4f},{max(cx_v):.4f}]")
    print(f"Syn cy: mean={np.mean(cy_v):.4f} std={np.std(cy_v):.4f} range=[{min(cy_v):.4f},{max(cy_v):.4f}]")


if __name__ == "__main__":
    main()
