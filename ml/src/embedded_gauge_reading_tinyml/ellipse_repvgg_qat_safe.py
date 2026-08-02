"""Build a QAT-ready RepVGG ellipse detector by inserting BatchNorm back into
the fused model and applying tfmot.quantization.keras.quantize_model.

Why:
- The fused single-branch model has Conv2D(use_bias=True) + ReLU per block.
  When QAT wraps these, the int8 calibration collapses to a constant output
  (the bias-only failure mode documented in
  docs/ai-memory/lessons-learned/2026-07-23-qat-safe-architecture.md).
- Inserting BatchNorm after each Conv2D (with gamma=1, beta=0 so the BN is
  a no-op at init) gives us Conv+BN+ReLU, which is QAT-safe.
- The optimizer can then re-tune the BN gammas to make the int8 grid
  cover the actual activation range.

The TFLite converter can fold the BN back into the conv at compile time
because BN is a linear op. So the deployed NPU graph is still a single
3x3 conv per block — RepVGG's whole point.
"""

from __future__ import annotations

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model


def _insert_bn_after_each_conv(model: keras.Model) -> keras.Model:
    """Return a new model where every Conv2D is followed by a frozen BatchNorm.

    The BN is initialised with gamma=1, beta=0 and moving_mean=0, moving_var=1
    so it is an identity transform at init. QAT will move the BN params
    around to make the int8 grid cover the actual activation range.
    """
    config = model.get_config()
    new_config = {"name": config["name"], "layers": [], "input_layers": config["input_layers"],
                  "output_layers": config["output_layers"]}
    layer_map = {layer["name"]: layer for layer in config["layers"]}
    new_layers = []

    for layer_cfg in config["layers"]:
        new_layers.append(layer_cfg)
        # If this layer is a Conv2D, insert a BN after it
        if layer_cfg["class_name"] == "Conv2D":
            bn_name = f"{layer_cfg['name']}_bn"
            # Build the BN config
            bn_config = {
                "name": bn_name,
                "class_name": "BatchNormalization",
                "config": {
                    "name": bn_name,
                    "trainable": True,
                    "dtype": layer_cfg["config"].get("dtype", "float32"),
                    "axis": [3] if layer_cfg["config"].get("data_format") == "channels_last" else [1],
                    "momentum": 0.9,
                    "epsilon": 0.001,
                    "center": True,
                    "scale": True,
                    "beta_initializer": {"class_name": "Zeros", "config": {}},
                    "gamma_initializer": {"class_name": "Ones", "config": {}},
                    "moving_mean_initializer": {"class_name": "Zeros", "config": {}},
                    "moving_variance_initializer": {"class_name": "Ones", "config": {}},
                    "beta_regularizer": None,
                    "gamma_regularizer": None,
                },
                "inbound_nodes": [[[layer_cfg["name"], 0, 0, {}]]],
            }
            # Update inbound nodes of any layer that was connected to the conv
            for other_cfg in config["layers"]:
                if other_cfg is layer_cfg:
                    continue
                for node in other_cfg.get("inbound_nodes", []):
                    for entry in node:
                        if entry[0] == layer_cfg["name"]:
                            entry[0] = bn_name
            new_layers.append(bn_config)

    new_config["layers"] = new_layers
    new_model = keras.Model.from_config(new_config)
    return new_model


def insert_bn_into_fused(fused_model: keras.Model) -> keras.Model:
    """Insert a frozen-initial BatchNorm after every Conv2D in the fused model.

    Returns a new model with the same outputs but with BN between every
    conv and its ReLU. The fused weights are copied over verbatim so the
    model produces the same output at init; subsequent QAT training
    tweaks the BN params to make the int8 grid usable.
    """
    # Build the topology first.
    new_model = _insert_bn_after_each_conv(fused_model)
    # Now copy weights from the fused model into the new model. Each conv
    # has weights [kernel, bias]; each new BN has weights [gamma, beta,
    # moving_mean, moving_variance]. The conv weights map directly; the
    # BN keeps its initialization (gamma=1, beta=0, mean=0, var=1).
    fused_weights = {w.name: w.numpy() for w in fused_model.weights}
    for layer in new_model.layers:
        if isinstance(layer, layers.Conv2D):
            for w in layer.weights:
                if w.name in fused_weights:
                    w.assign(fused_weights[w.name])
    return new_model
