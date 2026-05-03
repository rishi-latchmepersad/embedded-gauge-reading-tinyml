"""
Generate six step-illustration images for the classical CV baseline process diagram.
Each image is based on capture_p30c.png and annotates exactly what the embedded C
code does at that pipeline stage.

Steps:
  1  Raw YUV422 luma (greyscale render)
  2  Training crop region highlighted
  3  Bright centroid scan — qualifying pixels and centroid marked
  4  Ray sweep — all 360 rays coloured by score
  5  Best angle selected with confidence annotation
  6  Temperature output annotation

Run from WSL:
  python3 ml/scripts/generate_baseline_step_images.py
"""

import math
import pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path("/mnt/d/Projects/embedded-gauge-reading-tinyml")
YUV_SOURCE = ROOT / "data" / "captured" / "images" / "capture_p35c.yuv422"
OUT_DIR = ROOT / "docs" / "baseline_step_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE = 3  # upsample for readability in draw.io
W, H = 224, 224  # native capture resolution

# ---------------------------------------------------------------------------
# Parameters mirroring app_baseline_runtime.c defines
# ---------------------------------------------------------------------------
CROP_X_MIN, CROP_Y_MIN = 23, 57
CROP_W, CROP_H = 155, 123
CROP_X_MAX = CROP_X_MIN + CROP_W  # 178
CROP_Y_MAX = CROP_Y_MIN + CROP_H  # 180

BRIGHT_THRESHOLD = 150
SATURATION_THRESHOLD = 220
MIN_BRIGHT_PIXELS = 1024

ANGLE_BINS = 360
RAY_SAMPLES = 32
RAY_START_FRAC = 0.20
RAY_END_FRAC = 0.78
MIN_ANGLE_DEG = 135.0
SWEEP_DEG = 270.0
MIN_VALUE_C = -30.0
MAX_VALUE_C = 50.0

SUBDIAL_X_FRAC = 0.35
SUBDIAL_Y_MIN_FRAC = 0.10
SUBDIAL_Y_MAX_FRAC = 0.58


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_luma(path):
    """Load image as float32 luma array [H, W]."""
    img = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def upscale(img):
    return img.resize((W * SCALE, H * SCALE), Image.NEAREST)


def draw_label(draw, text, xy, color=(255, 255, 0), bg=(0, 0, 0, 180), font=None):
    """Draw text with a semi-transparent black backing rectangle."""
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=bg,
    )
    draw.text((x, y), text, fill=color, font=font)


def s(v):
    """Scale a pixel coordinate or size to the upscaled canvas."""
    return int(v * SCALE)


def rect_outline(draw, x0, y0, x1, y1, color, width=2):
    draw.rectangle([s(x0), s(y0), s(x1), s(y1)], outline=color, width=width)


def is_in_subdial(cx, cy, px, py, radius):
    dx = abs(px - cx)
    dy_signed = py - cy
    if dx < SUBDIAL_X_FRAC * radius:
        if SUBDIAL_Y_MIN_FRAC * radius < dy_signed < SUBDIAL_Y_MAX_FRAC * radius:
            return True
    return False


def score_angle(luma, cx, cy, angle_rad):
    unit_dx = math.cos(angle_rad)
    unit_dy = math.sin(angle_rad)
    perp_dx = -unit_dy
    perp_dy = unit_dx
    max_r = min(
        cx,
        W - 1 - cx,
        cy,
        H - 1 - cy,
    )
    start_r = max_r * RAY_START_FRAC
    end_r = max_r * RAY_END_FRAC
    step = (end_r - start_r) / (RAY_SAMPLES - 1)
    score = 0.0
    valid = 0
    for i in range(RAY_SAMPLES):
        r = start_r + step * i
        weight = 0.5 + 0.5 * (i / (RAY_SAMPLES - 1))
        sx = int(round(cx + unit_dx * r))
        sy = int(round(cy + unit_dy * r))
        if not (0 <= sx < W and 0 <= sy < H):
            continue
        if is_in_subdial(cx, cy, sx, sy, max_r):
            continue
        line_luma = luma[sy, sx]
        if line_luma > SATURATION_THRESHOLD:
            continue
        bg_sum = 0.0
        bg_count = 0
        for oi in range(2):
            offset = 2.0 + 2.0 * oi
            for sign in (+1, -1):
                bx = int(round(sx + sign * perp_dx * offset))
                by = int(round(sy + sign * perp_dy * offset))
                if not (0 <= bx < W and 0 <= by < H):
                    continue
                if is_in_subdial(cx, cy, bx, by, max_r):
                    continue
                bl = luma[by, bx]
                if bl <= SATURATION_THRESHOLD:
                    bg_sum += bl
                    bg_count += 1
        if bg_count == 0:
            continue
        contrast = (bg_sum / bg_count) - line_luma
        if contrast <= 0:
            continue
        score += contrast * weight
        valid += 1
    return score / valid if valid > 0 else 0.0


# ---------------------------------------------------------------------------
# Load source image
# ---------------------------------------------------------------------------
src_rgb = Image.open(SRC_IMAGE).convert("RGB").resize((W, H), Image.LANCZOS)
luma = load_luma(SRC_IMAGE)

# ---------------------------------------------------------------------------
# Step 1 — Raw YUV422 luma (greyscale render)
# ---------------------------------------------------------------------------
luma_u8 = np.clip(luma, 0, 255).astype(np.uint8)
step1 = Image.fromarray(luma_u8, mode="L").convert("RGB")
step1 = upscale(step1)
d = ImageDraw.Draw(step1)
draw_label(d, "Step 1: Raw luma (Y channel of YUV422)", (s(4), s(4)))
draw_label(d, "224x224 px  |  1 byte per pixel", (s(4), s(14)))
step1.save(OUT_DIR / "step1_raw_luma.png")
print("Saved step1_raw_luma.png")

# ---------------------------------------------------------------------------
# Step 2 — Training crop highlighted
# ---------------------------------------------------------------------------
step2 = upscale(src_rgb.copy())
d = ImageDraw.Draw(step2, "RGBA")
# semi-transparent overlay outside crop
overlay = Image.new("RGBA", step2.size, (0, 0, 0, 0))
od = ImageDraw.Draw(overlay)
od.rectangle([0, 0, step2.width, step2.height], fill=(0, 0, 0, 120))
od.rectangle(
    [s(CROP_X_MIN), s(CROP_Y_MIN), s(CROP_X_MAX), s(CROP_Y_MAX)],
    fill=(0, 0, 0, 0),
)
step2 = Image.alpha_composite(step2.convert("RGBA"), overlay).convert("RGB")
d = ImageDraw.Draw(step2)
rect_outline(d, CROP_X_MIN, CROP_Y_MIN, CROP_X_MAX, CROP_Y_MAX, (0, 255, 0), width=3)
# training crop center
tcx, tcy = (CROP_X_MIN + CROP_X_MAX) // 2, (CROP_Y_MIN + CROP_Y_MAX) // 2
r = 5
d.ellipse(
    [s(tcx - r), s(tcy - r), s(tcx + r), s(tcy + r)], outline=(0, 255, 0), width=2
)
draw_label(d, "Training crop  x=23..178  y=57..180", (s(4), s(4)))
draw_label(d, f"Center ({tcx},{tcy})  155x123 px", (s(4), s(14)))
step2.save(OUT_DIR / "step2_training_crop.png")
print("Saved step2_training_crop.png")

# ---------------------------------------------------------------------------
# Step 3 — Bright centroid scan
# ---------------------------------------------------------------------------
step3 = upscale(src_rgb.copy())
d = ImageDraw.Draw(step3)
# draw crop boundary
rect_outline(d, CROP_X_MIN, CROP_Y_MIN, CROP_X_MAX, CROP_Y_MAX, (0, 255, 0), width=2)
# highlight qualifying pixels
bright_xs, bright_ys = [], []
for y in range(CROP_Y_MIN, CROP_Y_MAX):
    for x in range(CROP_X_MIN, CROP_X_MAX):
        v = luma[y, x]
        if BRIGHT_THRESHOLD <= v <= SATURATION_THRESHOLD:
            bright_xs.append(x)
            bright_ys.append(y)
            d.point((s(x), s(y)), fill=(255, 200, 0))

if len(bright_xs) >= MIN_BRIGHT_PIXELS:
    cx = int(sum(bright_xs) / len(bright_xs))
    cy = int(sum(bright_ys) / len(bright_ys))
    r = 8
    d.ellipse(
        [s(cx - r), s(cy - r), s(cx + r), s(cy + r)], outline=(255, 0, 0), width=3
    )
    d.line([s(cx - 12), s(cy), s(cx + 12), s(cy)], fill=(255, 0, 0), width=2)
    d.line([s(cx), s(cy - 12), s(cx), s(cy + 12)], fill=(255, 0, 0), width=2)
    draw_label(d, f"Bright centroid ({cx},{cy})", (s(4), s(4)))
    draw_label(d, f"{len(bright_xs)} pixels  150<=luma<=220", (s(4), s(14)))
else:
    draw_label(d, f"Bright centroid FAILED ({len(bright_xs)} px < 1024)", (s(4), s(4)))
    cx, cy = W // 2, H // 2

step3.save(OUT_DIR / "step3_bright_centroid.png")
print("Saved step3_bright_centroid.png")

# ---------------------------------------------------------------------------
# Step 4 — Ray sweep (all 360 rays coloured by score)
# ---------------------------------------------------------------------------
step4 = upscale(src_rgb.copy())
d = ImageDraw.Draw(step4)

center_x, center_y = W // 2, H // 2  # image-center hypothesis for clarity
max_r = min(center_x, W - 1 - center_x, center_y, H - 1 - center_y)
end_r = max_r * RAY_END_FRAC

scores = []
for i in range(ANGLE_BINS):
    frac = i / (ANGLE_BINS - 1)
    angle = math.radians(MIN_ANGLE_DEG + frac * SWEEP_DEG)
    scores.append(score_angle(luma, center_x, center_y, angle))

min_s = min(s_ for s_ in scores if s_ > 0) if any(s_ > 0 for s_ in scores) else 0
max_s = max(scores)

for i, sc in enumerate(scores):
    frac = i / (ANGLE_BINS - 1)
    angle = math.radians(MIN_ANGLE_DEG + frac * SWEEP_DEG)
    dx = math.cos(angle)
    dy = math.sin(angle)
    t = (sc - min_s) / (max_s - min_s + 1e-6)
    r_val = int(255 * t)
    b_val = int(255 * (1 - t))
    color = (r_val, 0, b_val)
    ex = int(round(center_x + dx * end_r))
    ey = int(round(center_y + dy * end_r))
    d.line([s(center_x), s(center_y), s(ex), s(ey)], fill=color, width=1)

r = 5
d.ellipse(
    [s(center_x - r), s(center_y - r), s(center_x + r), s(center_y + r)],
    fill=(255, 255, 0),
)
draw_label(d, "Ray sweep: 360 angles  32 samples each", (s(4), s(4)))
draw_label(d, "Red=high contrast  Blue=low contrast", (s(4), s(14)))
step4.save(OUT_DIR / "step4_ray_sweep.png")
print("Saved step4_ray_sweep.png")

# ---------------------------------------------------------------------------
# Step 5 — Best angle + confidence
# ---------------------------------------------------------------------------
step5 = upscale(src_rgb.copy())
d = ImageDraw.Draw(step5)

best_i = int(np.argmax(scores))
runner_up = sorted(scores)[-2]
best_score = scores[best_i]
snr = best_score / runner_up if runner_up > 0 else 2.0
confidence = (1.0 - 1.0 / snr) if snr > 1 else 0.0

best_frac = best_i / (ANGLE_BINS - 1)
best_angle = math.radians(MIN_ANGLE_DEG + best_frac * SWEEP_DEG)
dx = math.cos(best_angle)
dy = math.sin(best_angle)
tip_x = int(round(center_x + dx * end_r))
tip_y = int(round(center_y + dy * end_r))

d.line([s(center_x), s(center_y), s(tip_x), s(tip_y)], fill=(255, 50, 50), width=3)
r = 6
d.ellipse(
    [s(center_x - r), s(center_y - r), s(center_x + r), s(center_y + r)],
    fill=(255, 255, 0),
)
angle_deg = math.degrees(best_angle)
draw_label(d, f"Best angle: {angle_deg:.1f} deg", (s(4), s(4)))
draw_label(
    d,
    f"Confidence: {confidence:.3f}  (score={best_score:.1f} runner_up={runner_up:.1f})",
    (s(4), s(14)),
)
step5.save(OUT_DIR / "step5_best_angle.png")
print("Saved step5_best_angle.png")

# ---------------------------------------------------------------------------
# Step 6 — Temperature output
# ---------------------------------------------------------------------------
step6 = upscale(src_rgb.copy())
d = ImageDraw.Draw(step6)

# replicate angle→temperature
min_angle_rad = math.radians(MIN_ANGLE_DEG)
sweep_rad = math.radians(SWEEP_DEG)
shifted = best_angle - min_angle_rad
while shifted < 0:
    shifted += 2 * math.pi
while shifted >= 2 * math.pi:
    shifted -= 2 * math.pi
shifted = max(0.0, min(shifted, sweep_rad))
frac = shifted / sweep_rad
temp_c = MIN_VALUE_C + frac * (MAX_VALUE_C - MIN_VALUE_C)

d.line([s(center_x), s(center_y), s(tip_x), s(tip_y)], fill=(255, 50, 50), width=3)
r = 6
d.ellipse(
    [s(center_x - r), s(center_y - r), s(center_x + r), s(center_y + r)],
    fill=(255, 255, 0),
)

big_text = f"{temp_c:.1f} C"
draw_label(d, big_text, (s(60), s(190)), color=(0, 255, 100))
draw_label(d, f"frac={frac:.3f}  conf={confidence:.3f}", (s(4), s(4)))
draw_label(
    d,
    f"angle={angle_deg:.1f}deg  =>  {MIN_VALUE_C}+{frac:.3f}*{MAX_VALUE_C-MIN_VALUE_C:.0f}",
    (s(4), s(14)),
)
step6.save(OUT_DIR / "step6_temperature_output.png")
print("Saved step6_temperature_output.png")

print(f"\nAll 6 step images saved to {OUT_DIR}")
