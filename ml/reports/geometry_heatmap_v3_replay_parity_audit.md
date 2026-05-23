# Geometry Heatmap v3 Replay Parity Audit

- Validation rows: 47
- Decoder: softargmax w3
- Canonical preprocessing: python_training_rgb_bilinear

## Replay Paths
- Trainer-style replay accepted MAE: 2.3358 C
- Trainer-style replay acceptance rate: 0.8511
- Trainer-style replay worst accepted error: 8.3222 C
- Standalone Keras accepted MAE: 3.6002 C
- Standalone Keras acceptance rate: 0.7021
- Standalone Keras worst accepted error: 13.5297 C
- TFLite FP32 accepted MAE: 3.6002 C
- TFLite FP32 acceptance rate: 0.7021
- TFLite FP32 worst accepted error: 13.5295 C
- TFLite INT8 accepted MAE: 3.3048 C
- TFLite INT8 acceptance rate: 0.5957
- TFLite INT8 worst accepted error: 11.8713 C

## Parity Deltas
- Trainer vs standalone Keras temp drift mean/median/p90: 2.1192 / 1.3026 / 4.3893
- Trainer vs standalone Keras center drift mean/median: 4.0770 / 3.9597
- Trainer vs standalone Keras tip drift mean/median: 14.1735 / 8.9031
- Canonical Keras vs FP32 temp drift mean/median/p90: 0.0000 / 0.0000 / 0.0001
- Canonical Keras vs FP32 center drift mean/median: 0.0001 / 0.0001
- Canonical Keras vs FP32 tip drift mean/median: 0.0003 / 0.0003
- Canonical Keras vs INT8 temp drift mean/median/p90: 1.9923 / 1.1288 / 4.2724
- Canonical Keras vs INT8 center drift mean/median: 3.0205 / 2.3763
- Canonical Keras vs INT8 tip drift mean/median: 14.7833 / 13.0247

## Crop Parity
- Crop metadata match rate: 100.00%
- Trainer preprocessing: legacy_trainer_heatmap
- Canonical preprocessing: python_training_rgb_bilinear
- Trainer resize method: 
- Canonical resize method: rgb_bilinear

## Training-vs-Inference Behavior
- Fake-quant round-trip is only active when the trainer explicitly applies the quantized-style replay path.
- Inference paths use `training=False`.
- The exported FP32 and INT8 replay paths are deterministic on repeated calls.

## Checkpoint Reload Parity
- Reload max abs diff (center_heatmap): 0.00000000
- Reload max abs diff (tip_heatmap): 0.00000000
- Reload max abs diff (confidence): 0.00000000
- Reload mean abs diff (center_heatmap): 0.00000000
- Reload mean abs diff (tip_heatmap): 0.00000000
- Reload mean abs diff (confidence): 0.00000000
- Reload parity passed: yes

## Decision
- The canonical validation replay is the standalone Keras / FP32 replay path.
- The trainer-side selection metric was optimistic because it used a non-canonical preprocessing/scoring path.
- Previous checkpoint selection should not be trusted for export decisions.
- The next architecture step should be decided only after rerunning v3 training with canonical validation scoring.