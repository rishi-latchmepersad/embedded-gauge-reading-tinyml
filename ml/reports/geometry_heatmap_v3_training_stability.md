# Geometry Heatmap v3 Training Stability

## Summary

The v3 trainer was initially collapsing to NaN after optimization started. The raw batch tensors and the raw per-batch losses were already finite, so the failure mode was not bad inputs or a broken forward pass. The most likely cause was optimization instability from a combination of:

- unnormalized coordinate/temperature objectives,
- a too-aggressive output noise injection schedule on step 1,
- and gradient explosion in the first few updates.

After normalizing the unstable losses, adding global gradient clipping, and ramping the fake-quant noise from a small starting value, the trainer stayed numerically stable in both the fixed-batch smoke and the short full-data smoke.

## Missing-Label Masking

One train example had a missing angle target. That example is now masked in the supervised losses, and the validation/test splits do not contain missing angle targets.

Result:

- masked missing angle/temperature supervision instead of propagating NaNs
- raw batch tensors remained finite
- raw per-batch losses remained finite before optimization

## Loss-Scale Diagnostics

The `--debug-one-batch-losses` path reported finite values for all major losses and outputs.

Initial one-batch diagnostics:

- center_heatmap_loss: `0.36691570`
- tip_heatmap_loss: `4.26893425`
- center_coord_loss: `0.00010139`
- tip_coord_loss: `0.00141752`
- angle_loss: `0.00391468`
- temperature_loss: `0.00071089`
- confidence_loss: `0.42491069`
- distillation_loss: `0.64158547`
- total_loss: `9.05072021`
- global_gradient_norm: `1103.00842285`

Finite-status checks:

- raw outputs finite: `True`
- fake-quant outputs finite: `True`
- losses finite: `True`
- gradients finite: `True`

Raw and fake-quant output stats were also finite and well-bounded:

- center heatmap values stayed in `[0, 1]`
- tip heatmap values stayed in `[0, 1]`
- confidence stayed in `[0, 1]`

## One-Batch Train Smoke

The fixed-batch optimization smoke ran for 50 steps with:

- backbone frozen
- decoder/head trainable only
- learning rate `3e-6`
- batch size `8`
- fake-quant noise active from epoch 1
- gradient clipping by global norm at `1.0`

The training trace stayed finite and the geometry losses decreased materially.

Selected checkpoints from the 50-step smoke:

- step 1 total loss: `9.04077244`
- step 25 total loss: `1.26287520`
- step 50 total loss: `0.37911019`
- final check total loss: `0.36787653`
- final global gradient norm: `11.22256088`
- geometry_loss_improved: `True`

The gradient norm fell from about `1103` at the start to about `11` at the final check, so the earlier NaN behavior is now strongly consistent with gradient-instability rather than bad tensors.

## Loss Normalization Changes

The v3 objective was updated so the training losses live on comparable scales:

- coordinate losses now use normalized `[0, 1]` coordinates from soft-argmax decoding
- angle loss now uses a bounded circular loss, `0.5 * (1 - cos(delta_rad))`
- temperature loss now uses normalized temperature values with a bounded Huber loss
- fake-quant losses use the same normalized coordinate/angle/temperature targets
- coordinate regression is now aligned with deployment-space normalized geometry instead of raw pixel-scale penalties

## Stability Controls Added

Additional stabilization measures:

- explicit global gradient clipping at `clipnorm = 1.0`
- output noise ramp from `0.001` to `0.008` over the early epochs
- low smoke learning rate (`3e-6`)
- backbone frozen for the stability smoke

## Short Full-Data Smoke

The 3-epoch frozen smoke completed without NaNs and wrote the expected training artifacts.

Validation replay results by epoch:

| Epoch | Accepted MAE | Acceptance | Worst accepted | >20 C failures | Temp drift mean | Tip drift mean | Guardrail disagreements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `2.4206 C` | `0.8298` | `8.3675 C` | `0` | `0.4012 C` | `2.5986 px` | `2` |
| 2 | `2.2977 C` | `0.8511` | `8.2227 C` | `0` | `0.2531 C` | `2.3974 px` | `3` |
| 3 | `2.2676 C` | `0.8511` | `8.1561 C` | `0` | `0.3108 C` | `2.5388 px` | `3` |

The final smoke output remained finite and stable:

- history CSV written: yes
- validation replay ran: yes
- validation accepted MAE finite: yes
- acceptance rate nonzero: yes
- worst accepted error finite: yes

## Decision

Full Phase 9 training is now allowed.

The trainer is no longer collapsing to NaN on a fixed-batch optimization test, and the short real-data smoke completed successfully with finite replay metrics.
