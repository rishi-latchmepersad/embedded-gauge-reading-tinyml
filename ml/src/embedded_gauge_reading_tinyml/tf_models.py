"""
TF/Keras backbones for STM32N6 Neural-ART NPU deployment.

Architecture 3b — QARepVGG-Pro (scalable, reparametrizable, QAT-optimized).

Key improvements over QARepVGG-Mini:
  - Width multiplier α for capacity scaling (default 1.5× → ~1.7 MB INT8)
  - Full reparameterization: train multi-branch → fuse single 3×3 conv → QAT
  - Coordinate-preserving Squeeze-Excitation blocks
  - Sub-pixel peak decoder (parabolic Taylor refinement)
  - ReLU-only throughout (NPU compatible)

Training flow:
  1. build_qarepvgg_multi()   — train with multi-branch blocks
  2. reparameterize_model()    — fuse to single-branch inference graph
  3. apply QAT on fused model  — tfmot.quantize_model (no custom layers)
  4. Export TFLite int8
"""

from __future__ import annotations

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _conv_norm(x: tf.Tensor, filters: int, kernel_size: int = 3,
               strides: int = 1, use_bias: bool = False, name: str = ""
               ) -> tuple[tf.Tensor, layers.Conv2D, layers.BatchNormalization]:
    """Conv2D → BatchNormalization, returning (output_tensor, conv_layer, bn_layer)."""
    conv = layers.Conv2D(filters, kernel_size, strides=strides,
                         padding="same", use_bias=use_bias, name=f"{name}_conv")
    bn = layers.BatchNormalization(name=f"{name}_bn")
    return bn(conv(x)), conv, bn


def _conv_bn_relu(x: tf.Tensor, filters: int, kernel_size: int = 3,
                  strides: int = 1, name: str = "") -> tf.Tensor:
    y, _, _ = _conv_norm(x, filters, kernel_size, strides, name=name)
    return layers.ReLU(name=f"{name}_relu")(y)


def _se_block(x: tf.Tensor, ratio: float = 0.25, name: str = "") -> tf.Tensor:
    """Channel SE: GAP → 1×1 (down) → ReLU → 1×1 (up) → sigmoid → scale."""
    c = int(x.shape[-1])
    reduced = max(1, int(c * ratio))
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Reshape([1, 1, c], name=f"{name}_reshape")(se)
    se = layers.Conv2D(reduced, 1, activation="relu", name=f"{name}_down")(se)
    se = layers.Conv2D(c, 1, activation="sigmoid", name=f"{name}_up")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage-wise channel plan given width multiplier α
# ═══════════════════════════════════════════════════════════════════════════════
#  Base (α=1.0):  Stem:32  S1:64  S2:96  S3:128  S4:160 → 724k params
#  α=1.5:         Stem:48  S1:96  S2:144  S3:192  S4:240 → ~1.63M params
#  α=2.0:         Stem:64  S1:128 S2:192  S3:256  S4:320 → ~2.90M params

def _channel_plan(alpha: float) -> dict[str, int]:
    return {
        "stem":   max(16, int(32 * alpha)),
        "s1":     max(32, int(64 * alpha)),
        "s2":     max(48, int(96 * alpha)),
        "s3":     max(64, int(128 * alpha)),
        "s4":     max(80, int(160 * alpha)),
        "proj":   max(32, int(64 * alpha)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  BN-fusion helpers (used by reparameterize_model)
# ═══════════════════════════════════════════════════════════════════════════════

def _fuse_bn_weights(conv_w: np.ndarray, bn_gamma: np.ndarray, bn_beta: np.ndarray,
                     bn_mean: np.ndarray, bn_var: np.ndarray, bn_eps: float = 1e-3
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Fuse BatchNorm into preceding Conv2D weights: W' = W * γ/σ, b' = β - γ*μ/σ."""
    std = np.sqrt(bn_var + bn_eps)
    scale = bn_gamma / std
    w_fused = conv_w * scale.reshape(1, 1, 1, -1)
    b_fused = bn_beta - bn_gamma * bn_mean / std
    return w_fused, b_fused


def _pad_kernel_1x1_to_3x3(w: np.ndarray, stride: int = 1) -> np.ndarray:
    """Pad a 1×1 kernel to 3×3, matching TF's asymmetric `padding='same'`.

    TF's SAME padding for stride=2 with odd kernel adds pad_before=0, pad_after=1
    (asymmetric).  This means the 3×3 kernel's centre reads different input pixels
    than the 1×1 kernel at the same output position.  To compensate:
      - stride=1 → centre-pad (original 1×1 weight at kernel index [1,1])
      - stride=2 → top-left-pad (weight at kernel index [0,0])
    """
    if stride == 1:
        return np.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]])
    elif stride == 2:
        return np.pad(w, [[0, 2], [0, 2], [0, 0], [0, 0]])


def _make_identity_kernel(c: int) -> np.ndarray:
    """3×3 identity kernel: centre pixel = 1, rest = 0."""
    w = np.zeros((3, 3, c, c), dtype=np.float32)
    for i in range(c):
        w[1, 1, i, i] = 1.0
    return w


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 3b: QARepVGG-Pro (Scalable + Reparameterisable)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Training-time: multi-branch block (3×3 + 1×1 + optional identity BN)
#  Inference:     fused single 3×3 conv via reparameterize_model()
#  QAT:           applied to the fused model for clean quantization

def _qa_block_multi(x: tf.Tensor, filters: int, stride: int = 1,
                    use_se: bool = False, se_ratio: float = 0.25,
                    name: str = "") -> tf.Tensor:
    """Training-time multi-branch RepVGG block.

    3×3 Conv → BN
    1×1 Conv → BN
    (Identity BN if stride==1 and dims match)
    └→ Add → (SE) → ReLU
    """
    in_c = int(x.shape[-1])
    use_id = (stride == 1 and in_c == filters)

    # 3×3 branch
    b3, c3, bn3 = _conv_norm(x, filters, 3, stride, name=f"{name}_3x3")

    # 1×1 branch
    b1, c1, bn1 = _conv_norm(x, filters, 1, stride, name=f"{name}_1x1")

    branches = [b3, b1]
    if use_id:
        bn_id = layers.BatchNormalization(name=f"{name}_id_bn")
        branches.append(bn_id(x))

    y = layers.Add(name=f"{name}_add")(branches)
    if use_se and stride == 1:
        y = _se_block(y, ratio=se_ratio, name=f"{name}_se")
    return layers.ReLU(name=f"{name}_relu")(y)


def _qa_block_fused(x: tf.Tensor, filters: int, stride: int = 1,
                    use_se: bool = False, se_ratio: float = 0.25,
                    name: str = "") -> tf.Tensor:
    """Inference-time single-branch block — after reparameterization.

    Single 3×3 Conv (bias=True) → (SE) → ReLU.
    """
    y = layers.Conv2D(filters, 3, strides=stride, padding="same",
                       use_bias=True, name=f"{name}_fused")(x)
    if use_se and stride == 1:
        y = _se_block(y, ratio=se_ratio, name=f"{name}_se")
    return layers.ReLU(name=f"{name}_relu")(y)


def _qa_stage_multi(x, filters, n_blocks, stride, use_se=False, se_ratio=0.25, name=""):
    for i in range(n_blocks):
        s = stride if i == 0 else 1
        x = _qa_block_multi(x, filters, stride=s, use_se=use_se,
                            se_ratio=se_ratio, name=f"{name}block{i}")
    return x


def _qa_stage_fused(x, filters, n_blocks, stride, use_se=False, se_ratio=0.25, name=""):
    for i in range(n_blocks):
        s = stride if i == 0 else 1
        x = _qa_block_fused(x, filters, stride=s, use_se=use_se,
                            se_ratio=se_ratio, name=f"{name}block{i}")
    return x


def _decoder_heads_upgraded(x: tf.Tensor, proj_c: int = 64,
                            name: str = "head") -> dict[str, tf.Tensor]:
    """Decoder with wider projection + sub-pixel-friendly 3×3 refinement."""
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest",
                             name=f"{name}_up")(x)
    x = _conv_bn_relu(x, proj_c, 3, name=f"{name}_refine")
    heatmap = layers.Conv2D(1, 1, name=f"{name}_heatmap")(x)
    box_size = layers.Conv2D(2, 1, name=f"{name}_box")(x)
    angle = layers.Activation("tanh", name=f"{name}_angle_tanh")(
        layers.Conv2D(2, 1, name=f"{name}_angle")(x),
    )
    return {"heatmap": heatmap, "box_size": box_size, "angle": angle}


# ── Build multi-branch model (for training) ──────────────────────────────

def build_qarepvgg_multi(input_shape=(320, 320, 3), alpha: float = 1.5,
                         use_se: bool = True, se_ratio: float = 0.25) -> Model:
    """Training model with multi-branch RepVGG blocks + optional SE.

    Channel plan scaled by α. Params at α=1.5: ~1.63M.
    """
    cp = _channel_plan(alpha)
    inputs = keras.Input(shape=input_shape, name="image")
    x = _conv_bn_relu(inputs, cp["stem"], 3, strides=2, name="stem")

    x = _qa_stage_multi(x, cp["s1"], n_blocks=2, stride=2,
                        use_se=False, name="s1")
    x = _qa_stage_multi(x, cp["s2"], n_blocks=2, stride=2,
                        use_se=use_se, se_ratio=se_ratio, name="s2")
    x = _qa_stage_multi(x, cp["s3"], n_blocks=2, stride=2,
                        use_se=use_se, se_ratio=se_ratio, name="s3")
    x = _qa_stage_multi(x, cp["s4"], n_blocks=1, stride=1,
                        use_se=use_se, se_ratio=se_ratio, name="s4")

    outputs = _decoder_heads_upgraded(x, proj_c=cp["proj"], name="head")
    model = Model(inputs, outputs, name=f"qarepvgg_pro_multi_a{alpha}")
    return model


# ── Reparameterisation: multi-branch → single-branch ─────────────────────

def _extract_fused_conv(model: Model, block_name: str,
                         stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fuse one multi-branch RepVGG block into a single (W, b). Returns numpy arrays."""
    def _get_weights(name):
        c = model.get_layer(f"{block_name}_{name}_conv")
        bn = model.get_layer(f"{block_name}_{name}_bn")
        kw = c.kernel.numpy() if hasattr(c.kernel, 'numpy') else np.array(c.kernel)
        return kw, bn

    # 3×3 branch
    kw3, bn3 = _get_weights("3x3")
    w3, b3 = _fuse_bn_weights(
        kw3, np.array(bn3.gamma), np.array(bn3.beta),
        np.array(bn3.moving_mean), np.array(bn3.moving_variance),
    )

    # 1×1 branch
    kw1, bn1 = _get_weights("1x1")
    w1_raw, b1 = _fuse_bn_weights(
        kw1, np.array(bn1.gamma), np.array(bn1.beta),
        np.array(bn1.moving_mean), np.array(bn1.moving_variance),
    )
    w1 = _pad_kernel_1x1_to_3x3(w1_raw, stride=stride)

    w = w3 + w1
    b = b3 + b1

    # Identity branch (if exists — stride=1 only)
    try:
        bn_id = model.get_layer(f"{block_name}_id_bn")
        out_c = kw3.shape[-1]
        w_id = _make_identity_kernel(out_c)
        w_id_fused, b_id = _fuse_bn_weights(
            w_id, np.array(bn_id.gamma), np.array(bn_id.beta),
            np.array(bn_id.moving_mean), np.array(bn_id.moving_variance),
        )
        w = w + w_id_fused
        b = b + b_id
    except Exception:
        pass

    return w, b


def reparameterize_model(multi_model: Model, use_se: bool = True,
                         se_ratio: float = 0.25) -> Model:
    """Build single-branch inference model with fused weights.

    Transfers every multi-branch RepVGG block into a single 3×3 Conv2D.
    Leaves SE blocks, stem, and decoder heads intact (no fusion needed).
    """
    alpha = float(multi_model.get_layer("stem_conv").filters) / 32.0
    cp = _channel_plan(alpha)

    inputs = keras.Input(shape=multi_model.input_shape[1:], name="image")
    x = _conv_bn_relu(inputs, cp["stem"], 3, strides=2, name="stem")

    # Build fused stages — same topology, single conv per block
    fusion_map = {
        "s1block0": (cp["s1"], 2), "s1block1": (cp["s1"], 1),
        "s2block0": (cp["s2"], 2), "s2block1": (cp["s2"], 1),
        "s3block0": (cp["s3"], 2), "s3block1": (cp["s3"], 1),
        "s4block0": (cp["s4"], 1),
    }

    x = _qa_stage_fused(x, cp["s1"], n_blocks=2, stride=2,
                        use_se=False, name="s1")
    x = _qa_stage_fused(x, cp["s2"], n_blocks=2, stride=2,
                        use_se=use_se, se_ratio=se_ratio, name="s2")
    x = _qa_stage_fused(x, cp["s3"], n_blocks=2, stride=2,
                        use_se=use_se, se_ratio=se_ratio, name="s3")
    x = _qa_stage_fused(x, cp["s4"], n_blocks=1, stride=1,
                        use_se=use_se, se_ratio=se_ratio, name="s4")

    outputs = _decoder_heads_upgraded(x, proj_c=cp["proj"], name="head")
    fused_model = Model(inputs, outputs, name=f"qarepvgg_pro_fused_a{alpha}")

    # Copy weights for stem, decoder, SE blocks from trained model
    for layer in fused_model.layers:
        try:
            src = multi_model.get_layer(layer.name)
            if src.weights and "fused" not in layer.name:
                layer.set_weights(src.get_weights())
        except Exception:
            pass

    # Fuse each RepVGG block and set the fused conv weights
    for block_name, (filters, stride) in fusion_map.items():
        w_fused, b_fused = _extract_fused_conv(multi_model, block_name, stride=stride)
        target_name = f"{block_name}_fused"
        try:
            conv_layer = fused_model.get_layer(target_name)
            conv_layer.kernel.assign(w_fused)
            conv_layer.bias.assign(b_fused)
        except Exception as e:
            print(f"  Warning: could not set weights for {target_name}: {e}")

    return fused_model


# ═══════════════════════════════════════════════════════════════════════════════
#  Sub-pixel peak decoder (Task C)
# ═══════════════════════════════════════════════════════════════════════════════

def decode_heatmap_peak(heatmap: np.ndarray) -> tuple[float, float, float]:
    """2D sub-pixel centre from heatmap via parabolic fit around argmax.

    Returns (cx_norm, cy_norm, peak_value).

    Mathematical principle:
      Given argmax at (x₀, y₀), fit a parabola f(x) = a(x-x₀)² + b(x-x₀) + c
      using neighbours f(x₀-1), f(x₀), f(x₀+1). The vertex is at:
        Δx = [f(x₀-1) - f(x₀+1)] / [2·(2·f(x₀) - f(x₀-1) - f(x₀+1))]
      Same for Δy. This recovers ≈0.1–0.3 px sub-pixel accuracy vs 0.5 px for argmax.

    Reference: "A Parabolic Model for Accurate Sub-Pixel Peak Estimation"
    """
    h, w = heatmap.shape
    flat = heatmap.ravel()
    idx = np.argmax(flat)
    y0, x0 = idx // w, idx % w
    peak = float(heatmap[y0, x0])

    # Clamp neighbourhood to image bounds
    x_left  = heatmap[y0, max(x0-1, 0)]
    x_right = heatmap[y0, min(x0+1, w-1)]
    y_up    = heatmap[max(y0-1, 0), x0]
    y_down  = heatmap[min(y0+1, h-1), x0]

    def _parabolic_peak(c1, c2, c3):
        """Sub-pixel offset from three points c1(left/up), c2(centre), c3(right/down)."""
        denom = 2.0 * (2.0 * c2 - c1 - c3)
        if abs(denom) < 1e-8:
            return 0.0
        return (c1 - c3) / denom

    dx = _parabolic_peak(x_left, peak, x_right)
    dy = _parabolic_peak(y_up, peak, y_down)

    # Clamp to prevent overshoot beyond ±0.5 pixels
    dx = max(-0.5, min(0.5, dx))
    dy = max(-0.5, min(0.5, dy))

    x_sub = (x0 + dx) / (w - 1)
    y_sub = (y0 + dy) / (h - 1)
    return x_sub, y_sub, peak


def decode_heatmap_softargmax(heatmap: np.ndarray) -> tuple[float, float]:
    """Soft-argmax centre-of-mass for comparison."""
    h, w = heatmap.shape
    s = np.sum(heatmap)
    if s <= 0:
        return 0.5, 0.5
    hm_n = heatmap / s
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                          np.arange(w, dtype=np.float32), indexing="ij")
    cy = float(np.sum(hm_n * ys)) / (h - 1)
    cx = float(np.sum(hm_n * xs)) / (w - 1)
    return cx, cy


# ═══════════════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════════════

TF_MODEL_REGISTRY: dict[str, callable] = {
    "qarepvgg_pro_multi": build_qarepvgg_multi,
}

__all__ = [
    "build_qarepvgg_multi",
    "reparameterize_model",
    "decode_heatmap_peak",
    "decode_heatmap_softargmax",
    "_channel_plan",
]
