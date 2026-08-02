#!/usr/bin/env python3
"""Fast architecture screening for the ellipse detector.

Screens several encoder/head variants in ONE process on a capped training
subset with short epochs, scoring each on fixed small slices of the three
test sets (FP32 only).  The winner is then promoted to a full-data run with
QAT + int8 export (see the iter5/iter6 launchers).

Why this exists: a full 25-epoch + 10-QAT + export run takes ~40 min per
variant; this harness ranks 5-6 architectures in ~10 min so we only spend
the expensive full run on the winner.

Memory safeguards (AGENTS.md): uint8 image storage, capped shuffle buffer,
per-sample float conversion, preflight estimate, and launch through
``run_wsl_guarded.sh``.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_ellipse_all_test_sets import _load_zip, _metrics  # noqa: E402
from train_ellipse_robust_384 import (  # noqa: E402
    BOARD_TRAIN_ZIPS,
    IMAGE_SIZE,
    SEED,
    _block,
    load_zips,
)
import train_ellipse_multiscale_simcc_384 as simcc_mod  # noqa: E402
import train_ellipse_multiscale_universal_384 as univ_mod  # noqa: E402

SHUFFLE_BUFFER = 4096
MEMORY_BUDGET_MB = 40000


def configure_gpu() -> None:
    """Cap TensorFlow's first GPU at 15 GB so WSL retains headroom."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=15000)]
        )


# ---------------------------------------------------------------------------
# Architecture builders (all share the SimCC 14,415-value contract)
# ---------------------------------------------------------------------------

def _encoder(inputs: tf.Tensor, channels: tuple[int, ...], attention: str = "none") -> tuple[tf.Tensor, list[tf.Tensor]]:
    """Build the shared 5/6-stage encoder with optional ECA attention."""
    layers = keras.layers
    skips: list[tf.Tensor] = []
    x = inputs
    for stage, filters in enumerate(channels):
        x = _block(x, filters, 2, f"enc{stage}_down")
        x = _block(x, filters, 1, f"enc{stage}_refine")
        if attention == "eca" and stage >= 2:
            # why: ECA (efficient channel attention) is a cheap, QAT-friendly
            # squeeze-excite that should help the model pick gauge-relevant
            # channels without adding many parameters.
            gap = layers.GlobalAveragePooling2D(name=f"eca_gap{stage}")(x)
            k = max(3, int((np.log2(filters) + 1) // 2))
            if k % 2 == 0:
                k += 1
            attn = layers.Reshape((1, 1, filters), name=f"eca_r{stage}")(gap)
            attn = layers.Conv2D(1, k, padding="same", activation="sigmoid", name=f"eca_conv{stage}")(attn)
            x = layers.Multiply(name=f"eca_mul{stage}")([x, attn])
        skips.append(x)
    return x, skips


def build_simcc_arch(channels: tuple[int, ...], attention: str = "none") -> keras.Model:
    """Build a SimCC-center multiscale model with the given encoder width."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x, skips = _encoder(inputs, channels, attention)

    bottleneck_gap = layers.GlobalAveragePooling2D(name="scale_gap")(x)
    scale_confidence = layers.Dense(3, activation="softmax", name="scale_confidence")(bottleneck_gap)

    heads: list[tf.Tensor] = []
    geometry_heads: list[tf.Tensor] = []
    # why: the decoder starts at the encoder bottleneck (384/2^n) and
    # upsamples to 24/48/96; the matching skips are the three stages just
    # above the bottleneck: indices (n-2, n-3, n-4).
    n_stages = len(channels)
    skip_indices = (n_stages - 2, n_stages - 3, n_stages - 4)
    for head_index, (filters, skip_index) in enumerate(zip((48, 32, 24), skip_indices)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{head_index}")(x)
        x = layers.Concatenate(name=f"join{head_index}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"dec{head_index}")
        rim = layers.Conv2D(1, 1, activation="sigmoid", name=f"rim_{simcc_mod.SIZES[head_index]}")(x)
        heads.append(layers.Flatten(name=f"rim_flat_{simcc_mod.SIZES[head_index]}")(rim))
        geometry = layers.GlobalAveragePooling2D(name=f"geometry_gap_{simcc_mod.SIZES[head_index]}")(x)
        geometry = layers.Dense(24, activation="relu", name=f"geometry_shared_{simcc_mod.SIZES[head_index]}")(geometry)
        geometry_heads.append(layers.Dense(4, activation="sigmoid", name=f"geometry_{simcc_mod.SIZES[head_index]}")(geometry))

    fine_features = skips[0]
    simcc_proj = layers.Conv2D(32, 1, padding="same", activation="relu", name="simcc_proj")(fine_features)
    simcc_pool = layers.GlobalAveragePooling2D(name="simcc_gap")(simcc_proj)
    simcc_shared = layers.Dense(128, activation="relu", name="simcc_shared")(simcc_pool)
    simcc_x = layers.Dense(simcc_mod.SIMCC_BINS, name="simcc_x")(simcc_shared)
    simcc_y = layers.Dense(simcc_mod.SIMCC_BINS, name="simcc_y")(simcc_shared)

    output = layers.Concatenate(name="simcc_contract")(
        [*heads, *geometry_heads, scale_confidence, simcc_x, simcc_y]
    )
    # why: Keras model names must be valid root scopes (no parens/commas).
    width_code = "-".join(str(filters) for filters in channels)
    return keras.Model(inputs, output, name=f"simcc_{width_code}_{attention}")


def _multiscale_decoder(
    x: tf.Tensor,
    skips: list[tf.Tensor],
    *,
    with_center: bool,
    with_offset: bool = False,
    head_filters: tuple[int, ...] = (48, 32, 24),
) -> tuple[list[tf.Tensor], list[tf.Tensor], tf.Tensor, tf.Tensor | None]:
    """Shared 3-scale decoder: center+rim or rim-only, geometry, and offsets.

    Handles arbitrary encoder depth: the decoder starts at the bottleneck
    (384/2^n) and does one warm-up upsample for each extra stage beyond 5 so
    the heads always land on 24/48/96.
    """
    layers = keras.layers
    n_stages = len(skips)
    warmups = n_stages - 5
    for warmup in range(warmups):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"warmup_up{warmup}")(x)
        x = layers.Concatenate(name=f"warmup_join{warmup}")([x, skips[4 - warmup]])
        x = _block(x, 48, 1, f"warmup_block{warmup}")
    heads: list[tf.Tensor] = []
    geometry_heads: list[tf.Tensor] = []
    offset_tensor: tf.Tensor | None = None
    for head_index, (filters, skip_index) in enumerate(zip(head_filters, (3, 2, 1))):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{head_index}")(x)
        x = layers.Concatenate(name=f"join{head_index}")([x, skips[skip_index]])
        x = _block(x, filters, 1, f"dec{head_index}")
        if with_center:
            center = layers.Conv2D(1, 1, activation="sigmoid", name=f"center_{univ_mod.SIZES[head_index]}")(x)
            heads.append(layers.Flatten(name=f"center_flat_{univ_mod.SIZES[head_index]}")(center))
        rim = layers.Conv2D(1, 1, activation="sigmoid", name=f"rim_{univ_mod.SIZES[head_index]}")(x)
        heads.append(layers.Flatten(name=f"rim_flat_{univ_mod.SIZES[head_index]}")(rim))
        geometry = layers.GlobalAveragePooling2D(name=f"geometry_gap_{univ_mod.SIZES[head_index]}")(x)
        geometry = layers.Dense(24, activation="relu", name=f"geometry_shared_{univ_mod.SIZES[head_index]}")(geometry)
        geometry_heads.append(layers.Dense(4, activation="sigmoid", name=f"geometry_{univ_mod.SIZES[head_index]}")(geometry))
        if with_offset and head_index == 2:
            # why: the fine head features carry the highest-resolution
            # position signal; CenterNet regresses the sub-cell offset here.
            offset_gap = layers.GlobalAveragePooling2D(name="offset_gap")(x)
            offset_shared = layers.Dense(24, activation="relu", name="offset_shared")(offset_gap)
            offset_tensor = layers.Dense(2, name="center_offset")(offset_shared)
    return heads, geometry_heads, x, offset_tensor


def build_universal_arch(channels: tuple[int, ...], attention: str = "none") -> keras.Model:
    """Build the iter3/iter4 pure-heatmap universal model (reference)."""
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x, skips = _encoder(inputs, channels, attention)
    bottleneck_gap = layers.GlobalAveragePooling2D(name="scale_gap")(x)
    scale_confidence = layers.Dense(3, activation="softmax", name="scale_confidence")(bottleneck_gap)
    heads, geometry_heads, _, _ = _multiscale_decoder(x, skips, with_center=True)
    output = layers.Concatenate(name="multiscale_contract")(
        [*heads, *geometry_heads, scale_confidence]
    )
    width_code = "-".join(str(filters) for filters in channels)
    return keras.Model(inputs, output, name=f"universal_{width_code}")


def build_universal_offset_arch(channels: tuple[int, ...], attention: str = "none") -> keras.Model:
    """Universal heatmap model + CenterNet-style center-offset refinement.

    Literature: Zhou et al. 2019 ("Objects as Points") regresses a sub-cell
    offset from the keypoint location to recover sub-pixel precision that
    the heatmap grid loses.  We append a 2-value center-offset head to the
    fine decoder features; at decode time the offset is added to the
    heatmap-decoded center.

    Contract layout (universal 24,207 values + 2 offset = 24,209):
    [3x(center+rim) maps, 3x4 geometry, 3 confidence, cx_offset, cy_offset]
    """

    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x, skips = _encoder(inputs, channels, attention)
    bottleneck_gap = layers.GlobalAveragePooling2D(name="scale_gap")(x)
    scale_confidence = layers.Dense(3, activation="softmax", name="scale_confidence")(bottleneck_gap)
    heads, geometry_heads, _, offset_tensor = _multiscale_decoder(
        x, skips, with_center=True, with_offset=True, head_filters=(48, 32, 24)
    )
    assert offset_tensor is not None
    output = layers.Concatenate(name="offset_contract")(
        [*heads, *geometry_heads, scale_confidence, offset_tensor]
    )
    width_code = "-".join(str(filters) for filters in channels)
    return keras.Model(inputs, output, name=f"universal_offset_{width_code}")


def build_simcc_quad_arch(channels: tuple[int, ...], attention: str = "none") -> keras.Model:
    """SimCC center head with 2x2 spatial pooling instead of GAP.

    Why: the plain SimCC variant used GlobalAveragePooling2D, which erases
    WHERE the gauge is and collapsed on tiny/off-center gauges (test_2
    ~188px in screen #1).  RTMPose-style heads keep spatial structure; we
    pool the fine features into 4 quadrants so position survives.

    Contract: same as simcc (14,415 values) with the same offsets, only the
    head pooling differs.
    """

    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x, skips = _encoder(inputs, channels, attention)
    bottleneck_gap = layers.GlobalAveragePooling2D(name="scale_gap")(x)
    scale_confidence = layers.Dense(3, activation="softmax", name="scale_confidence")(bottleneck_gap)
    heads, geometry_heads, _, _ = _multiscale_decoder(x, skips, with_center=False)

    fine_features = skips[0]
    simcc_proj = layers.Conv2D(32, 1, padding="same", activation="relu", name="simcc_proj")(fine_features)
    # why: 4x4 average pooling keeps coarse position while shrinking the
    # head input (plain GAP collapsed on off-center gauges in screen #1);
    # pool size 48 collapses the 192x192 features to a 4x4 grid so the
    # dense head stays small enough for the 2.5 MB SRAM budget.
    quad = layers.AveragePooling2D(48, strides=48, name="simcc_quad")(simcc_proj)
    quad_flat = layers.Flatten(name="simcc_quad_flat")(quad)
    simcc_shared = layers.Dense(128, activation="relu", name="simcc_shared")(quad_flat)
    simcc_x = layers.Dense(simcc_mod.SIMCC_BINS, name="simcc_x")(simcc_shared)
    simcc_y = layers.Dense(simcc_mod.SIMCC_BINS, name="simcc_y")(simcc_shared)
    output = layers.Concatenate(name="simcc_contract")(
        [*heads, *geometry_heads, scale_confidence, simcc_x, simcc_y]
    )
    width_code = "-".join(str(filters) for filters in channels)
    return keras.Model(inputs, output, name=f"simcc_quad_{width_code}")


def build_universal_noskip_arch(channels: tuple[int, ...], attention: str = "none") -> keras.Model:
    """Universal heatmap model WITHOUT decoder skip connections.

    Why: tests whether the skip joins matter or the encoder bottleneck alone
    carries enough position information.  Same 24,207-value contract.
    """
    layers = keras.layers
    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="image")
    x, skips = _encoder(inputs, channels, attention)
    bottleneck_gap = layers.GlobalAveragePooling2D(name="scale_gap")(x)
    scale_confidence = layers.Dense(3, activation="softmax", name="scale_confidence")(bottleneck_gap)
    heads: list[tf.Tensor] = []
    geometry_heads: list[tf.Tensor] = []
    for head_index, filters in enumerate((48, 32, 24)):
        x = layers.UpSampling2D(2, interpolation="nearest", name=f"up{head_index}")(x)
        x = _block(x, filters, 1, f"dec{head_index}")
        center = layers.Conv2D(1, 1, activation="sigmoid", name=f"center_{univ_mod.SIZES[head_index]}")(x)
        rim = layers.Conv2D(1, 1, activation="sigmoid", name=f"rim_{univ_mod.SIZES[head_index]}")(x)
        heads.extend([layers.Flatten(name=f"center_flat_{univ_mod.SIZES[head_index]}")(center),
                      layers.Flatten(name=f"rim_flat_{univ_mod.SIZES[head_index]}")(rim)])
        geometry = layers.GlobalAveragePooling2D(name=f"geometry_gap_{univ_mod.SIZES[head_index]}")(x)
        geometry = layers.Dense(24, activation="relu", name=f"geometry_shared_{univ_mod.SIZES[head_index]}")(geometry)
        geometry_heads.append(layers.Dense(4, activation="sigmoid", name=f"geometry_{univ_mod.SIZES[head_index]}")(geometry))
    width_code = "-".join(str(f) for f in channels)
    output = layers.Concatenate(name="multiscale_contract")([*heads, *geometry_heads, scale_confidence])
    return keras.Model(inputs, output, name=f"universal_noskip_{width_code}")


def build_simcc_quad_wide_arch(channels: tuple[int, ...], attention: str = "none") -> keras.Model:
    """RTMPose-style spatial SimCC with a wider encoder."""
    return build_simcc_quad_arch(channels, attention)


ARCHES = {
    # name: (builder, channels, attention)
    # why: 10-variant roster covering width, depth, attention, CenterNet
    # offset, spatial SimCC, and skip-connection ablation, all within the
    # universal heatmap family that won screens #1-#2.
    "universal_v1": (build_universal_arch, (16, 24, 32, 48, 64), "none"),
    "universal_wider": (build_universal_arch, (24, 32, 48, 64, 96), "none"),
    "universal_eca": (build_universal_arch, (16, 24, 32, 48, 64), "eca"),
    "universal_deep6": (build_universal_arch, (16, 24, 32, 48, 64, 80), "none"),
    "universal_offset": (build_universal_offset_arch, (16, 24, 32, 48, 64), "none"),
    "universal_wide_deep": (build_universal_arch, (24, 32, 48, 64, 96, 128), "none"),
    "universal_noskip": (build_universal_noskip_arch, (16, 24, 32, 48, 64), "none"),
    "universal_wide_eca": (build_universal_arch, (24, 32, 48, 64, 96), "eca"),
    "simcc_quad": (build_simcc_quad_arch, (16, 24, 32, 48, 64), "none"),
    "simcc_quad_wide": (build_simcc_quad_wide_arch, (24, 32, 48, 64, 96), "none"),
}


# ---------------------------------------------------------------------------
# Data helpers (memory-safe)
# ---------------------------------------------------------------------------

def _iter_samples(images: np.ndarray, targets: np.ndarray) -> object:
    """Return a generator over (image, contract) host arrays (uint8 storage)."""

    def samples() -> object:
        """Yield one sample at a time from the loaded host arrays."""
        for index in range(len(images)):
            yield images[index].astype(np.float32) / 255.0, targets[index]

    return samples


def _augment_uint8(images: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale/translate/contrast-augment uint8 images, keeping targets aligned."""
    rng = np.random.default_rng(SEED)
    scales = rng.choice(np.asarray([0.20, 0.30, 0.42, 0.60, 0.80]), size=len(images))
    aug_images = np.empty_like(images)
    aug_targets = np.empty_like(targets)
    for index, (image, target, scale) in enumerate(zip(images, targets, scales)):
        scaled_radius = target[2:4] * float(scale)
        base_center = 0.5 + float(scale) * (target[:2] - 0.5)
        lower = scaled_radius + 0.01
        upper = 1.0 - scaled_radius - 0.01
        desired_center = rng.uniform(lower, upper)
        translation = desired_center - base_center
        source = Image.fromarray(image[..., 0])
        scaled_size = max(1, int(round(IMAGE_SIZE * float(scale))))
        scaled = source.resize((scaled_size, scaled_size), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=int(np.mean(image)))
        offset = (
            int(round((0.5 - 0.5 * scale + translation[0]) * IMAGE_SIZE)),
            int(round((0.5 - 0.5 * scale + translation[1]) * IMAGE_SIZE)),
        )
        canvas.paste(scaled, offset)
        canvas = ImageEnhance.Contrast(canvas).enhance(float(rng.uniform(0.75, 1.25)))
        aug_images[index] = np.asarray(canvas, dtype=np.uint8)[..., None]
        transformed = target.copy()
        transformed[:2] = 0.5 + scale * (target[:2] - 0.5) + translation
        transformed[2:4] = target[2:4] * scale
        aug_targets[index] = transformed
    return (
        np.concatenate([images, aug_images], axis=0),
        np.concatenate([targets, aug_targets], axis=0),
    )


def _memory_preflight(n_samples: int, contract_size: int) -> None:
    """Abort with a readable error if the dataset footprint exceeds the budget."""
    image_bytes = n_samples * IMAGE_SIZE * IMAGE_SIZE  # uint8
    contract_bytes = n_samples * contract_size * 4  # float32
    shuffle_bytes = min(n_samples, SHUFFLE_BUFFER) * (
        IMAGE_SIZE * IMAGE_SIZE + contract_size * 4
    )
    total_mb = (image_bytes + contract_bytes + shuffle_bytes) / 1e6
    print(f"memory preflight: {total_mb / 1024:.1f} GiB estimated for {n_samples} samples", flush=True)
    if total_mb > MEMORY_BUDGET_MB:
        raise SystemExit(
            f"aborting: estimated {total_mb / 1024:.1f} GiB exceeds {MEMORY_BUDGET_MB / 1024:.0f} GiB budget"
        )


def _load_training_subset(generic_limit: int, tiny_repeats: int, board_repeats: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a capped, balanced training subset (uint8 images + geometry)."""
    generic_images, generic_targets = load_zips(["train_1.zip", "val_1.zip"], labels=("GaugeFace",))
    tiny_images, tiny_targets = load_zips(["train_2.zip", "val_2.zip"], labels=("GaugeFace",))
    board_images, board_targets = load_zips(simcc_mod.TRAIN_BOARD_ZIPS, labels=("temp_dial",))
    # why: deterministic subsample of the generic pool so every architecture
    # sees the same screening set; tiny/board keep full repeats for balance.
    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(generic_images), min(generic_limit, len(generic_images)), replace=False)
    generic_images = generic_images[keep]
    generic_targets = generic_targets[keep]
    images = np.concatenate([
        generic_images,
        np.repeat(tiny_images, tiny_repeats, axis=0),
        np.repeat(board_images, board_repeats, axis=0),
    ])
    targets = np.concatenate([
        generic_targets,
        np.repeat(tiny_targets, tiny_repeats, axis=0),
        np.repeat(board_targets, board_repeats, axis=0),
    ])
    images_u8 = np.clip(np.round(images * 255.0), 0, 255).astype(np.uint8)
    del images
    return images_u8, targets


def _test_slices(test_1_limit: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load fixed slices of each test zip (same seed across architectures).

    Keys are "test_1", "test_2", "test_3" so the leaderboard scorer can use
    them directly.
    """
    slices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    test_1_images, test_1_targets = _load_zip("test_1.zip")
    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(test_1_images), min(test_1_limit, len(test_1_images)), replace=False)
    slices["test_1"] = (test_1_images[keep], test_1_targets[keep])
    for zip_name in ("test_2.zip", "test_3.zip"):
        images, targets = _load_zip(zip_name)
        slices[zip_name.removesuffix(".zip")] = (images, targets)
    return slices


def score_metrics(metrics: dict[str, object]) -> float:
    """Rank by mean center MAE across the three test slices (lower is better)."""
    return float(np.mean([metrics[split]["center_mae_px"] for split in ("test_1", "test_2", "test_3")]))


class OffsetUniversalLoss(keras.losses.Loss):
    """Universal multiscale loss + L1 on the CenterNet center-offset head.

    The first 24,207 contract values are scored exactly like the universal
    model; the final 2 values (cx, cy offset) are regressed with L1 against
    the ground-truth center.
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return the combined universal + offset loss."""
        base_size = univ_mod.TOTAL_MAP_VALUES + 12 + 3
        base_loss = univ_mod.MultiScaleLoss()(
            y_true[:, :base_size], y_pred[:, :base_size]
        )
        offset_loss = tf.reduce_sum(
            tf.abs(y_true[:, base_size : base_size + 2] - y_pred[:, base_size : base_size + 2]),
            axis=-1,
        )
        return base_loss + 2.0 * offset_loss

    def get_config(self) -> dict[str, object]:
        """Return the serializable loss configuration."""
        return super().get_config()


def decode_offset(contract: np.ndarray) -> np.ndarray:
    """Decode universal contract, then refine center with the offset head.

    The offset head predicts the ground-truth center directly; we blend it
    with the heatmap-decoded center 50/50 so the refinement cannot drag a
    confident heatmap peak off-target.
    """
    base_size = univ_mod.TOTAL_MAP_VALUES + 12 + 3
    predictions = univ_mod.decode_contract(contract[:, :base_size])
    offset = contract[:, base_size : base_size + 2]
    predictions[:, :2] = 0.5 * (predictions[:, :2] + offset)
    return predictions


def main() -> None:
    """Screen all architectures on the capped subset and write a leaderboard."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "ellipse_screen_fast")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--generic-limit", type=int, default=2500)
    parser.add_argument("--tiny-repeats", type=int, default=60)
    parser.add_argument("--board-repeats", type=int, default=2)
    parser.add_argument("--test-1-limit", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--arches", type=str, default=",".join(ARCHES.keys()))
    args = parser.parse_args()

    configure_gpu()
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)

    selected = [name.strip() for name in args.arches.split(",") if name.strip() in ARCHES]
    print(f"screening {len(selected)} architectures: {selected}", flush=True)

    images_u8, geometry = _load_training_subset(args.generic_limit, args.tiny_repeats, args.board_repeats)
    images_u8, geometry = _augment_uint8(images_u8, geometry)
    test_slices = _test_slices(args.test_1_limit)

    leaderboard: dict[str, object] = {}
    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        builder, channels, attention = ARCHES[name]
        if name.startswith("simcc"):
            contract_size = simcc_mod.CONTRACT_SIZE
            contract_targets = simcc_mod.make_contract_targets(geometry)
            loss_fn = simcc_mod.SimCCLoss()
            decode_fn = simcc_mod.decode_contract
        elif name == "universal_offset":
            # why: universal contract (24,207) + 2 CenterNet offset values.
            contract_size = univ_mod.TOTAL_MAP_VALUES + 12 + 3 + 2
            contract_targets = np.concatenate(
                [univ_mod.make_map_targets(geometry), geometry[:, :2]], axis=1
            ).astype(np.float32)
            loss_fn = OffsetUniversalLoss()
            decode_fn = decode_offset
        else:
            contract_size = univ_mod.TOTAL_MAP_VALUES + 12 + 3
            contract_targets = univ_mod.make_map_targets(geometry)
            loss_fn = univ_mod.MultiScaleLoss()
            decode_fn = univ_mod.decode_contract
        _memory_preflight(len(images_u8), contract_size)
        dataset = (
            tf.data.Dataset.from_generator(
                _iter_samples(images_u8, contract_targets),
                output_signature=(
                    tf.TensorSpec((IMAGE_SIZE, IMAGE_SIZE, 1), tf.float32),
                    tf.TensorSpec((contract_size,), tf.float32),
                ),
            )
            .shuffle(min(SHUFFLE_BUFFER, len(images_u8)), seed=SEED, reshuffle_each_iteration=True)
            .batch(args.batch_size)
            .prefetch(1)
        )

        # why: every architecture must start from the SAME weight-init seed
        # or the leaderboard ranking is polluted by random init order; the
        # dataset shuffle also re-seeds per arch so the data order is
        # identical for every variant.
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        model = builder(channels, attention)
        n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss_fn)
        model.fit(dataset, epochs=args.epochs, verbose=2)

        split_metrics: dict[str, object] = {}
        for split_name, (test_images, test_targets) in test_slices.items():
            predictions = decode_fn(model.predict(test_images, batch_size=16, verbose=0))
            split_metrics[split_name] = _metrics(predictions, test_targets)
            print(f"  {split_name}: center_mae {split_metrics[split_name]['center_mae_px']:.2f}px "
                  f"radius_mae {split_metrics[split_name]['radius_mae_px']:.2f}px", flush=True)

        leaderboard[name] = {
            "params": n_params,
            "split_metrics": split_metrics,
            "mean_center_mae": score_metrics(split_metrics),
        }
        print(f"  mean_center_mae {leaderboard[name]['mean_center_mae']:.2f}px params {n_params:,}", flush=True)

        # why: free the GPU graph between architectures; the next build must
        # not compete with a stale session for the 15 GB allocator.
        del model
        keras.backend.clear_session()
        gc.collect()

    ranked = sorted(leaderboard.items(), key=lambda item: item[1]["mean_center_mae"])
    print("\n=== LEADERBOARD ===", flush=True)
    for name, entry in ranked:
        print(f"{entry['mean_center_mae']:6.2f}px  {name:16s} {entry['params']:>10,} params", flush=True)
    (args.output / "leaderboard.json").write_text(
        json.dumps({"ranked": ranked, "selected": selected}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
