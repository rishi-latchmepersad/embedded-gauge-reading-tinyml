# tfmot QAT layer compatibility — 2026-07-23

Date: 2026-07-23
Status: validated
Scope: all models using tfmot.quantization.keras.quantize_model()
Evidence: multiple failed QAT attempts during v10/v11 training

## Supported layers (tfmot QAT)

These layers work with `tfmot.quantization.keras.quantize_model()`:
- `keras.layers.Conv2D`
- `keras.layers.DepthwiseConv2D`
- `keras.layers.Dense`
- `keras.layers.BatchNormalization`
- `keras.layers.ReLU`
- `keras.layers.ReLU(6.0)` — works but may cause issues on some NPU backends
- `keras.layers.MaxPooling2D`
- `keras.layers.GlobalAveragePooling2D`
- `keras.layers.UpSampling2D`
- `keras.layers.Concatenate`
- `keras.layers.Add` — works for residual connections
- `keras.layers.Reshape`
- `keras.layers.Flatten`
- `keras.layers.Dropout`
- `keras.layers.Activation("sigmoid")`
- `keras.layers.Activation("relu")`

## NOT supported layers

These layers FAIL with `tfmot.quantization.keras.quantize_model()`:
- `keras.layers.Lambda` — cannot be cloned
- `keras.layers.Multiply` — not in QAT registry
- Custom layers without `@keras.saving.register_keras_serializable()`
- `keras.layers.Activation("swish")` / `keras.layers.Activation("hard_sigmoid")`
- Any layer using `tf.nn.*` operations directly

## Workarounds

- For pooling: use `GlobalAveragePooling2D` instead of `Lambda(lambda t: tf.reduce_mean(t, axis=...))`
- For element-wise multiply: use `keras.layers.Add` with log-space tricks, or restructure to avoid multiply
- For custom operations: register with `@keras.saving.register_keras_serializable()` AND pass via `quantize_scope`
- For attention mechanisms: use only Dense + sigmoid (SE-style) without Multiply layer

## Impact on architecture design

- **Coordinate Attention**: Uses Lambda for pooling → NOT QAT-compatible
- **Squeeze-and-Excitation**: Uses Multiply layer → NOT QAT-compatible
- **CBAM**: Uses both Lambda and Multiply → NOT QAT-compatible
- **RepVGG**: Multi-branch with Add → QAT-compatible (after fusion)
- **MobileNetV2 inverted residual**: Uses Add for skip → QAT-compatible

## Recommendation

For QAT-compatible models, stick to:
- Conv2D + BatchNorm + ReLU blocks
- GlobalAveragePooling2D for spatial reduction
- Dense + sigmoid for channel attention (without multiply)
- Add for residual connections
- Standard Keras Functional API only
