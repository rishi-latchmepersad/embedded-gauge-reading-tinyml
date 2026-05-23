# Geometry Heatmap v2 Quantization Run Notes

Date: 2026-05-22

## Stalled Jobs

- Broad replay jobs for `ml/scripts/eval_geometry_heatmap_v2_tflite_variants.py` stalled in WSL mounted-drive I/O.
- The earlier `ml/scripts/eval_geometry_heatmap_v2_decode_methods.py` replay jobs also remained stuck.
- The long-running `pytest` session for the requested geometry tests also hit the same uninterruptible I/O pattern.
- The surviving processes were in Linux `D` state, so `kill -9` could not clear them until the blocked I/O returned.

## Why The Broad Replay Was Abandoned

- The broad replay path had already proven unstable in WSL when it tried to build the full split cache and sweep every exported variant.
- The broad loop was not necessary to answer the readiness question once the cached decode rows and per-variant exports were available.
- A narrower selected-variant replay gives the same decision signal with far less I/O and avoids repeatedly hanging on the mounted workspace.

## Trusted Cached Artifacts

- `ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/quantization_drift_analysis.csv`
- `ml/reports/geometry_heatmap_v2_quantization_drift_v2.md`
- `ml/reports/geometry_heatmap_v2_decode_method_comparison.md`
- `ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/decode_method_predictions.csv`
- `ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/decode_method_summary.csv`
- `ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/selected_decode_method.json`
- `ml/artifacts/training/geometry_heatmap_v2_board_replay/board_replay_predictions.csv`
- `ml/artifacts/deployment/geometry_heatmap_v2_tflite/tflite_replay_predictions.csv`
- `ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/variant_index.csv`
- `ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/variant_*/model.tflite`
- `ml/artifacts/deployment/geometry_heatmap_v2_tflite_v2/variant_*/tflite_tensor_contract.json`

## Scripts Patched

- `ml/scripts/eval_geometry_heatmap_v2_tflite_variants.py`
- `ml/scripts/eval_geometry_heatmap_v2_selected_tflite_variant_fast.py`
- `ml/scripts/analyze_geometry_heatmap_v2_quantization_drift.py`
- `ml/scripts/eval_geometry_heatmap_v2_decode_methods.py`

## Current Decode Lock

- The selected decode method artifact currently says `peak_weighted_centroid_w5`.
- The replay scripts now read that artifact instead of silently falling back to `softargmax`.

## Latest Narrow Replays

- The selected-variant smoke replay completed with `--split test --max-samples 20` using the cached Keras and current-INT8 replay rows plus one live candidate TFLite export.
- The full `--split test` replay also completed for the same narrow path.
- The priority float-I/O candidate and the next-priority int8-input/float-output candidate both showed the same poor drift profile, so the export format change did not solve the deployment gap.

## Pytest Status

- The requested pytest batch began collecting normally when run with `-s`.
- The run progressed through `tests/test_gauge_geometry.py`, `tests/test_geometry_crop_dataset.py`, and `tests/test_heatmap_utils.py`, and then printed `tests/test_heatmap_losses.py` before stalling in uninterruptible I/O.
- The active pytest process was `10471` in Linux `D` state, so it could not be interrupted cleanly.
