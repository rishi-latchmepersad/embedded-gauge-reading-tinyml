# v5 Improvement Plan

Goal: improve the current strict rectified scalar baseline without losing the
good middle-band performance we already have.

Current baseline:
- Model: `ml/artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras`
- Held-out test MAE: about `5.02C`
- Hard-case MAE: about `17.04C`
- Failure mode: the predictor collapses toward the middle of the temperature
  range on cold/hot samples.

Research-guided direction:
1. Make the model geometry-aware before trying to make it more clever.
2. Keep the scalar head, but add explicit spatial supervision for the needle.
3. Warm-start from the proven v5 backbone instead of training the geometry
   model from scratch.
4. Hold out the hard-case manifest so we can tell whether the new model really
   generalizes or only memorizes the training mix.

First implemented experiment:
- Train a `mobilenet_v2_geometry` model warm-started from strict v5.
- Use a smaller MobileNetV2 (`alpha=0.35`) so the GPU stays comfortable.
- Add hard-case repeats during training, but keep the hard-case holdout unseen.
- Monitor the same hard-case manifest we used for the scalar audit.

Follow-up scalar experiment:
- Keep the strict rectified manifest and crop boxes.
- Switch the MobileNetV2 scalar head from sigmoid-rescaled output to linear output.
- Warm-start from strict v5 so we preserve the good backbone and only change the output compression.
- Use the hard-case manifest as the truth test again, not the random split alone.

Observed outcome:
- The linear head improved the held-out split to around `6.10C` MAE.
- The hard-case manifest still sat at about `16.82C` MAE, so the cold/hot generalization gap remains.

What to do next if this helps:
- Add synthetic gauge pretraining with randomized lighting, blur, and needle
  appearance.
- Replace the scalar-only objective with a geometry-aware regularizer or a
  compact ordinal consistency loss.
- Fine-tune again on the rectified scalar path after geometry pretraining.
