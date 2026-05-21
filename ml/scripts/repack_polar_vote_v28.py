"""Repack the V28 polar-vote model, replacing Lambda layers with serializable custom Keras layers.

The original model uses ``tf.keras.layers.Lambda`` wrappers around ``tf.reduce_mean``
and ``tf.reduce_max`` for the radial-pooling stage. Lambda layers do not serialize
cleanly, so they block model export to TFLite / TFLM. This script rebuilds the
same architecture with custom ``ReduceMeanAxis`` / ``ReduceMaxAxis`` layers that
carry their ``axis`` parameter in ``get_config``, transfers weights from the
source model, and saves a fully-serializable ``.keras`` artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

# ---------------------------------------------------------------------------
# Project root -- resolves relative to this script so the defaults work
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]


# -- Custom serialisable layers ---------------------------------------------


class ReduceMeanAxis(keras.layers.Layer):
    """Compute ``tf.reduce_mean`` over a configurable axis.

    This replaces a ``Lambda(lambda x: tf.reduce_mean(x, axis=1))`` that
    cannot be serialized to a '.keras' file.
    """

    def __init__(self, axis: int = -1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.axis = axis  # axis along which to average

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Reduce along the stored axis, collapsing that dimension.
        return tf.reduce_mean(inputs, axis=self.axis)

    def get_config(self) -> dict:
        # Include `axis` so the layer can be re-instantiated from a saved model.
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class ReduceMaxAxis(keras.layers.Layer):
    """Compute ``tf.reduce_max`` over a configurable axis.

    Mirrors ``ReduceMeanAxis`` but takes the maximum instead of the mean.
    """

    def __init__(self, axis: int = -1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.axis = axis  # axis along which to take the max

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Reduce along the stored axis, collapsing that dimension.
        return tf.reduce_max(inputs, axis=self.axis)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


# -- Model construction -----------------------------------------------------


def _build_vote_backbone_clean() -> tuple[keras.Input, keras.layers.Layer]:
    """Build the polar-vote backbone without Lambda layers.

    Returns the input tensor and the output tensor so the caller can
    attach different heads (e.g. vote head vs. future regression head).
    """
    # Input is a 7-channel polar image at 224x224 resolution.
    inputs = keras.Input(shape=(224, 224, 7), name="polar_image")

    # --- Block 1: standard conv down-sample ---
    x = keras.layers.Conv2D(
        32, 3, strides=(2, 1), padding="same", use_bias=False, name="vote_conv2d_1"
    )(inputs)
    x = keras.layers.BatchNormalization(name="vote_bn2d_1")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_1")(x)

    # --- Block 2: separable conv ---
    x = keras.layers.SeparableConv2D(
        32, 3, padding="same", use_bias=False, name="vote_sepconv2d_2"
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_2")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_2")(x)

    # --- Pool along angular dimension (width axis) ---
    x = keras.layers.MaxPooling2D(pool_size=(2, 1), name="vote_pool2d_1")(x)

    # --- Block 3-4: deeper separable convs ---
    x = keras.layers.SeparableConv2D(
        64, 3, padding="same", use_bias=False, name="vote_sepconv2d_3"
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_3")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_3")(x)

    x = keras.layers.SeparableConv2D(
        64, 3, padding="same", use_bias=False, name="vote_sepconv2d_4"
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn2d_4")(x)
    x = keras.layers.Activation("swish", name="vote_act2d_4")(x)

    # --- Radial pooling: collapse the spatial height (axis=1) ---
    # These replace the non-serialisable Lambda layers from the original model.
    mean_pool = ReduceMeanAxis(axis=1, name="vote_radial_mean")(x)
    max_pool = ReduceMaxAxis(axis=1, name="vote_radial_max")(x)

    # Fuse mean and max radial profiles into a 2x channel descriptor.
    x = keras.layers.Concatenate(axis=-1, name="vote_radial_fuse")([mean_pool, max_pool])

    # --- 1-D refinement ---
    x = keras.layers.Conv1D(
        128, 3, padding="same", use_bias=False, name="vote_conv1d_1"
    )(x)
    x = keras.layers.BatchNormalization(name="vote_bn1d_1")(x)
    x = keras.layers.Activation("swish", name="vote_act1d_1")(x)

    x = keras.layers.Dropout(0.2, name="vote_dropout")(x)

    return inputs, x


def _build_model_vote_clean() -> keras.Model:
    """Attach the angle-logit head to the vote backbone and return the full model."""
    inputs, x = _build_vote_backbone_clean()

    # One logit per angular position (dense per-position prediction).
    angle_logits = keras.layers.Conv1D(
        1, 1, padding="same", name="angle_logits"
    )(x)
    # Flatten to a 1-D vector of angle votes.
    angle_logits_flat = keras.layers.Flatten(name="angle_logits_flat")(angle_logits)

    return keras.Model(inputs=inputs, outputs=angle_logits_flat, name="polar_angle_vote")


# -- Weight transfer --------------------------------------------------------


def _transfer_weights(source: keras.Model, target: keras.Model) -> None:
    """Copy weights from *source* to *target* by matching layer names.

    Layers that exist in both models share the same name; their weight tensors
    are copied directly. Layers present only in the target (the custom
    ReduceMeanAxis / ReduceMaxAxis layers) have no weights to transfer and are
    skipped silently.
    """
    # Build an index so we can look up source layers by name in O(1).
    source_layers: dict[str, keras.layers.Layer] = {
        layer.name: layer for layer in source.layers
    }

    for tgt_layer in target.layers:
        # Skip layers with no weights (InputLayer, custom reduce layers, etc.).
        if not tgt_layer.weights:
            continue

        src_layer = source_layers.get(tgt_layer.name)
        if src_layer is None:
            # Weighted layer in target has no counterpart in source -- log and skip.
            print(f"[warn] no source layer named '{tgt_layer.name}', skipping")
            continue

        if not src_layer.weights:
            continue

        # Verify shape compatibility before copying.
        for sw, tw in zip(src_layer.weights, tgt_layer.weights):
            if sw.shape != tw.shape:
                raise ValueError(
                    f"Shape mismatch for '{tgt_layer.name}': "
                    f"source {sw.shape} vs target {tw.shape}"
                )

        tgt_layer.set_weights(src_layer.get_weights())
        print(f"  + {tgt_layer.name}")


# -- Entry point ------------------------------------------------------------


def main() -> None:
    """Load the source model, rebuild with serialisable layers, transfer weights, and save."""
    parser = argparse.ArgumentParser(
        description="Repack polar-vote v28: replace Lambda layers with serialisable equivalents"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_REPO_ROOT / "ml" / "artifacts" / "training" / "polar_vote_circular_v28" / "model.keras",
        help="Path to the source .keras model (default: ml/artifacts/training/polar_vote_circular_v28/model.keras)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "ml" / "artifacts" / "runtime" / "polar_vote_circular_v28_repack",
        help="Directory to write the repacked model into",
    )
    args = parser.parse_args()

    source_path: Path = args.source
    output_dir: Path = args.output_dir
    output_path = output_dir / "model.keras"

    # --- 1. Load source model (allow unsafe deserialisation for Lambda layers) ---
    print(f"Loading source model from {source_path} ...")
    keras.config.enable_unsafe_deserialization()
    source_model = keras.models.load_model(str(source_path), safe_mode=False)
    print(f"  Source model: {source_model.name}, {len(source_model.layers)} layers")

    # --- 2. Build target model with serialisable layers ---
    print("Building target model (no Lambda layers) ...")
    target_model = _build_model_vote_clean()
    print(f"  Target model: {target_model.name}, {len(target_model.layers)} layers")

    # --- 3. Transfer weights ---
    print("Transferring weights:")
    _transfer_weights(source_model, target_model)

    # --- 4. Quick sanity check with dummy input ---
    dummy = np.random.randn(1, 224, 224, 7).astype(np.float32)
    src_out = source_model.predict(dummy, verbose=0)
    tgt_out = target_model.predict(dummy, verbose=0)
    # The custom layers compute the same operation, so outputs should match closely.
    max_diff = float(np.max(np.abs(src_out - tgt_out)))
    print(f"  Max output difference on dummy input: {max_diff:.6e}")
    if max_diff > 1e-4:
        print("[warn] Output difference exceeds 1e-4 - verify weight transfer!")
    else:
        print("  Output match confirmed.")

    # --- 5. Save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    target_model.save(str(output_path))
    print(f"Repacked model saved to {output_path}")


if __name__ == "__main__":
    main()
