# QAT-safe architecture is essential for int8 ellipse detection — 2026-07-23

Date: 2026-07-23
Status: validated
Scope: ellipse detector int8 deployment
Evidence: `ml/artifacts/gauge_ellipse_{v9,v10,v11}/`, `ml/artifacts/gauge_ellipse_qat_encoder_v1/`
Decision: Always use Conv+BN+ReLU for any model that needs int8 QAT export.
         Never use bias-only convolutions (no BatchNorm) for int8 models.

## The finding

**Bias-only convolutions collapse to a constant output after int8 quantization.**
This affects both QAT and PTQ. The model produces the same output for ALL
inputs, regardless of content. This was observed across three model variants
(v9, v10, v11) using the same architecture.

**QAT-safe architecture (Conv+BN+ReLU) produces meaningful int8 outputs.**
The `gauge_ellipse_qat_encoder_v1` model uses only standard Keras layers
(Conv2D + BatchNorm + ReLU) that tfmot can cleanly quantize. It produces
varying predictions for different inputs in int8.

## Evidence

| Model | Architecture | FP32 test MAE | int8 on test_3 | Status |
|-------|-------------|---------------|-----------------|--------|
| v9 | bias-only Conv, no BN | 0.0123 | Constant (0.479, 0.463) | BROKEN |
| v10 | bias-only Conv, no BN | 0.0119 | Constant (0.479, 0.463) | BROKEN |
| v11 | bias-only Conv, no BN | 0.0120 | Constant (0.479, 0.463) | BROKEN |
| MobileNetV2 PTQ | MobileNetV2, relu6 | N/A | Varies but imprecise | PARTIAL |
| **QAT encoder** | **Conv+BN+ReLU** | **0.0093** | **Varies (3.5-9.7px)** | **WORKS** |

## Root cause

Without BatchNorm, the activation ranges drift across a wide dynamic range
that depends on the input. A single representative dataset calibration pass
cannot capture that range, so the quantization scale factor becomes either
too wide (losing precision) or too narrow (clipping outliers). The result
is that all inputs quantize to the same int8 values.

With BatchNorm, the activations are normalized to approximately zero mean
and unit variance. The quantization grid captures the actual data distribution
faithfully, producing meaningful int8 outputs.

## Architecture rules for int8 models

1. **Always use BatchNorm** after every Conv2D layer
2. **Always use ReLU** (not relu6, swish, or hard-swish) as the activation
3. **No Lambda layers** - tfmot cannot clone them
4. **No tf.nn wrappers** - use standard Keras layers only
5. **Use GAP** (GlobalAveragePooling2D) instead of learned spatial collapse
   for the transition from spatial to vector features

## Recommended architecture

```python
def conv_bn_relu(x, filters, strides=1, name=""):
    x = layers.Conv2D(filters, 3, strides=strides, padding="same",
                      use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x
```

## References

- tfmot QAT registry: only `ReLU`, `Softmax`, `LeakyReLU` are allowlisted
- MobileNetV2 uses relu6 internally - not QAT-safe with `keras.applications`
- MobileNetV3 uses hard-swish - not QAT-safe
- The custom QAT encoder in `models_geometry_v2.py` is the reference implementation
