#!/usr/bin/env python3
"""Generate a synthetic gauge dataset for pretraining a scalar regressor.

The images are intentionally simple but gauge-shaped: a circular dial, tick
marks, a colored needle, a few glare/noise effects, and a small cluttered
subdial. The goal is to teach the network the angle-to-value relationship
before fine-tuning on the strict rectified real data.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


VALUE_MIN: Final[float] = -30.0
VALUE_MAX: Final[float] = 50.0

# Match the real gauge's lower-arc layout instead of the earlier top-arc toy dial.
GAUGE_SWEEP_START_DEG: Final[float] = 225.0
GAUGE_SWEEP_END_DEG: Final[float] = 315.0


@dataclass(frozen=True)
class SyntheticSpec:
    """Parameters describing one synthetic gauge render."""

    image_size: int
    profile: str
    center_x: float
    center_y: float
    radius: float
    value: float
    needle_angle_deg: float
    glare_angle_deg: float
    background_level: int


def _value_to_angle_deg(value: float) -> float:
    """Map a Celsius value onto the synthetic gauge sweep in image coordinates."""
    fraction = (value - VALUE_MIN) / (VALUE_MAX - VALUE_MIN)
    return GAUGE_SWEEP_START_DEG + (GAUGE_SWEEP_END_DEG - GAUGE_SWEEP_START_DEG) * fraction


def _sample_value(rng: np.random.Generator, profile: str) -> float:
    """Sample a gauge value, biasing hard-mode toward the extremes."""
    if profile == "hard":
        selector = float(rng.random())
        if selector < 0.35:
            return float(rng.uniform(VALUE_MIN, -10.0))
        if selector < 0.70:
            return float(rng.uniform(30.0, VALUE_MAX))
        return float(rng.uniform(-10.0, 30.0))
    return float(rng.uniform(VALUE_MIN, VALUE_MAX))


def _rand_clamped(rng: np.random.Generator, center: float, spread: float) -> float:
    """Draw a jittered value and keep it within a comfortable frame margin."""
    return float(center + rng.normal(0.0, spread))


def _make_background(
    image_size: int,
    level: int,
    rng: np.random.Generator,
    profile: str,
) -> Image.Image:
    """Create a noisy background canvas."""
    if profile == "hard":
        # Hard-mode backgrounds look flatter and more like preview captures.
        base = rng.integers(
            max(0, level - 10),
            min(255, level + 10) + 1,
            size=(image_size, image_size, 3),
            dtype=np.uint8,
        )
    else:
        base = rng.integers(
            max(0, level - 18),
            min(255, level + 18) + 1,
            size=(image_size, image_size, 3),
            dtype=np.uint8,
        )
    canvas = Image.fromarray(base, mode="RGB")
    blur_radius = 1.6 if profile == "hard" else 1.2
    return canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def _draw_ticks(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    center_y: float,
    radius: float,
    start_deg: float,
    end_deg: float,
) -> None:
    """Draw major and minor tick marks around the dial arc."""
    for index, deg in enumerate(np.linspace(start_deg, end_deg, 19)):
        angle = math.radians(float(deg))
        outer_x = center_x + math.cos(angle) * radius
        outer_y = center_y - math.sin(angle) * radius
        if index % 3 == 0:
            inner_radius = radius - 16.0
            width = 4
        else:
            inner_radius = radius - 10.0
            width = 2
        inner_x = center_x + math.cos(angle) * inner_radius
        inner_y = center_y - math.sin(angle) * inner_radius
        draw.line((inner_x, inner_y, outer_x, outer_y), fill=(30, 30, 30), width=width)


def _draw_dial_face(image: Image.Image, spec: SyntheticSpec, rng: np.random.Generator) -> None:
    """Draw the synthetic gauge face, needle, and light clutter."""
    draw = ImageDraw.Draw(image)

    outer_bbox = [
        spec.center_x - spec.radius,
        spec.center_y - spec.radius,
        spec.center_x + spec.radius,
        spec.center_y + spec.radius,
    ]
    inner_bbox = [
        spec.center_x - 0.86 * spec.radius,
        spec.center_y - 0.86 * spec.radius,
        spec.center_x + 0.86 * spec.radius,
        spec.center_y + 0.86 * spec.radius,
    ]

    # Draw the main dial body and a softly colored inner face.
    outer_fill = (235, 232, 226) if spec.profile == "standard" else (228, 225, 220)
    inner_fill = (248, 246, 242) if spec.profile == "standard" else (241, 239, 236)
    draw.ellipse(outer_bbox, fill=outer_fill, outline=(35, 35, 35), width=5)
    draw.ellipse(inner_bbox, fill=inner_fill, outline=(80, 80, 80), width=2)

    # Add the lower-right subdial that keeps the network aware of clutter.
    subdial_radius = 0.23 * spec.radius
    subdial_cx = spec.center_x + 0.38 * spec.radius
    subdial_cy = spec.center_y + 0.33 * spec.radius
    subdial_bbox = [
        subdial_cx - subdial_radius,
        subdial_cy - subdial_radius,
        subdial_cx + subdial_radius,
        subdial_cy + subdial_radius,
    ]
    draw.ellipse(subdial_bbox, fill=(225, 223, 218), outline=(55, 55, 55), width=2)

    # Tick marks on the main temperature sweep.
    # The real gauge uses a lower arc, so we place the synthetic scale there too.
    _draw_ticks(
        draw,
        center_x=spec.center_x,
        center_y=spec.center_y,
        radius=spec.radius * 0.92,
        start_deg=GAUGE_SWEEP_START_DEG,
        end_deg=GAUGE_SWEEP_END_DEG,
    )

    # A few labels help keep the synthetic images less toy-like.
    # Keep the endpoints on the lower edge of the dial to mirror the real gauge.
    font = ImageFont.load_default()
    for label_deg, label in [(225.0, "-30"), (255.0, "0"), (285.0, "30"), (315.0, "50")]:
        angle = math.radians(label_deg)
        text_radius = spec.radius * 0.72
        tx = spec.center_x + math.cos(angle) * text_radius
        ty = spec.center_y - math.sin(angle) * text_radius
        draw.text((tx - 8.0, ty - 5.0), label, fill=(25, 25, 25), font=font)

    # Draw a small set of subdial ticks.
    for deg in np.linspace(220.0, 320.0, 7):
        angle = math.radians(float(deg))
        outer_x = subdial_cx + math.cos(angle) * subdial_radius
        outer_y = subdial_cy - math.sin(angle) * subdial_radius
        inner_x = subdial_cx + math.cos(angle) * (subdial_radius - 5.0)
        inner_y = subdial_cy - math.sin(angle) * (subdial_radius - 5.0)
        draw.line((inner_x, inner_y, outer_x, outer_y), fill=(90, 90, 90), width=1)

    # Draw the needle itself.
    needle_angle = math.radians(spec.needle_angle_deg)
    needle_len = spec.radius * 0.83
    needle_x = spec.center_x + math.cos(needle_angle) * needle_len
    needle_y = spec.center_y - math.sin(needle_angle) * needle_len
    needle_base = (45, 20, 20) if spec.profile == "standard" else (58, 38, 34)
    needle_top = (200, 35, 35) if spec.profile == "standard" else (165, 50, 50)
    draw.line((spec.center_x, spec.center_y, needle_x, needle_y), fill=needle_base, width=5)
    draw.line((spec.center_x, spec.center_y, needle_x, needle_y), fill=needle_top, width=2)

    # Add a hub and a small highlight so the dial is less synthetic-looking.
    hub_radius = 0.055 * spec.radius
    draw.ellipse(
        [
            spec.center_x - hub_radius,
            spec.center_y - hub_radius,
            spec.center_x + hub_radius,
            spec.center_y + hub_radius,
        ],
        fill=(60, 60, 60),
        outline=(15, 15, 15),
    )

    # Add a few glare streaks and a mild vignette-like oval.
    glare_angle = math.radians(spec.glare_angle_deg)
    glare_scale = 0.40 if spec.profile == "hard" else 0.45
    glare_x = spec.center_x + math.cos(glare_angle) * spec.radius * glare_scale
    glare_y = spec.center_y - math.sin(glare_angle) * spec.radius * glare_scale
    draw.ellipse(
        [
            glare_x - 18,
            glare_y - 10,
            glare_x + 18,
            glare_y + 10,
        ],
        fill=(255, 255, 255),
    )
    draw.ellipse(
        [
            spec.center_x - 0.94 * spec.radius,
            spec.center_y - 0.94 * spec.radius,
            spec.center_x + 0.94 * spec.radius,
            spec.center_y + 0.94 * spec.radius,
        ],
        outline=(0, 0, 0),
        width=1,
    )

    # Add a soft blur to mix the hard edges a bit.
    blurred = image.filter(ImageFilter.GaussianBlur(radius=0.7 if spec.profile == "hard" else 0.6))
    image.paste(blurred)


def _render_synthetic_sample(spec: SyntheticSpec, rng: np.random.Generator) -> Image.Image:
    """Render one synthetic gauge sample as an RGB image."""
    image = _make_background(spec.image_size, spec.background_level, rng, spec.profile)
    _draw_dial_face(image, spec, rng)

    # Add a low-amplitude pixel-noise overlay and a tiny amount of blur.
    array = np.asarray(image, dtype=np.int16)
    noise_span = 12 if spec.profile == "hard" else 8
    array += rng.integers(-noise_span, noise_span + 1, size=array.shape, dtype=np.int16)
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")

    if spec.profile == "hard":
        # Hard-mode intentionally looks more like low-contrast preview captures.
        if rng.random() < 0.6:
            image = ImageOps.grayscale(image).convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.55, 0.9)))
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.72, 0.98)))
        image = image.rotate(
            float(rng.uniform(-11.0, 11.0)),
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=(235, 232, 226),
        )
        if rng.random() < 0.5:
            crop = int(image.size[0] * float(rng.uniform(0.84, 0.96)))
            left = max(0, (image.size[0] - crop) // 2 + int(rng.integers(-10, 11)))
            top = max(0, (image.size[1] - crop) // 2 + int(rng.integers(-10, 11)))
            right = min(image.size[0], left + crop)
            bottom = min(image.size[1], top + crop)
            image = image.crop((left, top, right, bottom)).resize(
                (spec.image_size, spec.image_size),
                resample=Image.BICUBIC,
            )

    return image.filter(ImageFilter.GaussianBlur(radius=0.45 if spec.profile == "hard" else 0.35))


def _make_spec(
    rng: np.random.Generator,
    image_size: int,
    profile: str,
) -> SyntheticSpec:
    """Sample one synthetic render specification."""
    value = _sample_value(rng, profile)
    center_x = _rand_clamped(rng, image_size * 0.50, image_size * 0.015)
    center_y = _rand_clamped(rng, image_size * 0.48, image_size * 0.015)
    radius = float(rng.uniform(image_size * 0.33, image_size * 0.38))
    needle_angle = _value_to_angle_deg(value)
    glare_angle = float(rng.uniform(0.0, 360.0))
    background_level = int(rng.integers(168, 236))
    return SyntheticSpec(
        image_size=image_size,
        profile=profile,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        value=value,
        needle_angle_deg=needle_angle,
        glare_angle_deg=glare_angle,
        background_level=background_level,
    )


def build_dataset(
    *,
    output_dir: Path,
    manifest_path: Path,
    num_samples: int,
    image_size: int,
    seed: int,
    profile_mode: str = "standard",
) -> None:
    """Generate a synthetic gauge image set and write its CSV manifest."""
    rng = np.random.default_rng(seed)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for index in range(num_samples):
        # Emit a small progress heartbeat so long synthetic runs do not look frozen.
        if index == 0 or (index + 1) % 50 == 0:
            print(
                f"[SYNTH] Rendering {index + 1}/{num_samples} "
                f"({profile_mode}) samples...",
                flush=True,
            )
        profile = "hard" if profile_mode == "hard" else "standard"
        spec = _make_spec(rng, image_size, profile)
        image = _render_synthetic_sample(spec, rng)
        image_name = f"synthetic_gauge_{index:05d}.png"
        image_path = images_dir / image_name
        image.save(image_path)
        rows.append(
            {
                "image_path": image_path.relative_to(output_dir.parent.parent).as_posix(),
                "value": f"{spec.value:.3f}",
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[SYNTH] Finished {profile_mode} dataset: "
        f"{num_samples} images -> {manifest_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate a synthetic gauge dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where synthetic images will be written.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to the CSV manifest that will be written.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1600,
        help="Number of synthetic images to generate.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size for the generated samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="RNG seed for deterministic generation.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="standard",
        choices=("standard", "hard"),
        help="Render profile to use for the synthetic images.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the requested synthetic dataset."""
    args = parse_args()
    build_dataset(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        num_samples=args.num_samples,
        image_size=args.image_size,
        seed=args.seed,
        profile_mode=args.profile,
    )


if __name__ == "__main__":
    main()
