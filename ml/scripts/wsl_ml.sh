#!/usr/bin/env bash
set -euo pipefail

# Run all ML commands from the ml/ directory so Poetry finds the project root.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer the Poetry install we verified in WSL, but fall back to PATH if needed.
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "Poetry was not found. Install it with pipx inside WSL first." >&2
  exit 1
fi

cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/wsl_ml.sh setup
  bash scripts/wsl_ml.sh gpu-check
  bash scripts/wsl_ml.sh baseline [baseline args...]
  bash scripts/wsl_ml.sh baseline-manifest [manifest args...]
  bash scripts/wsl_ml.sh train [training args...]
  bash scripts/wsl_ml.sh train-tiny [tiny training args...]
  bash scripts/wsl_ml.sh train-compact-interval [compact interval args...]
  bash scripts/wsl_ml.sh train-compact-geometry [compact geometry args...]
  bash scripts/wsl_ml.sh train-compact-geometry-longterm [compact geometry longterm args...]
    bash scripts/wsl_ml.sh train-compact-geometry-cascade-localizer [compact geometry cascade args...]
    bash scripts/wsl_ml.sh train-compact-geometry-cascade-localizer-longterm [compact geometry cascade longterm args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-geometry-uncertainty [geometry uncertainty args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-geometry-longterm [geometry longterm args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-geometry-cascade-localizer-longterm [geometry cascade longterm args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-direction-longterm [direction longterm args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-obb-longterm [obb longterm args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectifier [rectifier args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectifier-finetune [rectifier fine-tune args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectifier-hardcase-finetune [rectifier hard-case args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar [rectified-scalar args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar-linear [rectified-scalar linear args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar-tailbalanced [tail-balanced args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-synthetic-pretrain [synthetic pretrain args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar-from-synth [synthetic fine-tune args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar-from-hard-synth-previewaug [preview-heavy synthetic fine-tune args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar-full-labelled [full CVAT-labelled fine-tune args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar-from-geometry [geometry-derived rectified fine-tune args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar-all-data [all-data V5 args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar-combined [combined real + synthetic args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-sweep-distribution [sweep-distribution args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-polar-sweep-distribution [polar sweep-distribution args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar-raw-tail [raw-tail fine-tune args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-geometry-raw-cvat [raw CVAT geometry args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-rectified-scalar-interval [rectified interval args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-dual-resolution-interval [dual-resolution interval args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-polar-dualview [polar dual-view args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-polar-only [polar only args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-geometry-from-rectified-v5 [geometry args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-geometry-literature [geometry literature args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-geometry-localizer-only [geometry localizer args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-obb-localizer [obb localizer args...]
  bash scripts/wsl_ml.sh train-mobilenetv2-keypoint-from-rectified-v5 [keypoint args...]
  bash scripts/wsl_ml.sh eval-mobilenetv2-geometry-cascade [geometry cascade eval args...]
  bash scripts/wsl_ml.sh eval-mobilenetv2-geometry-cascade-strict-reader [geometry cascade strict-reader args...]
  bash scripts/wsl_ml.sh eval-mobilenetv2-obb-cascade-strict-reader [obb cascade strict-reader args...]
  bash scripts/wsl_ml.sh sweep-obb-localizer-strict-reader-crop-scale [obb crop-scale sweep args...]
  bash scripts/wsl_ml.sh eval-rectified-scalar [rectified eval args...]
  bash scripts/wsl_ml.sh eval-rectified-captures [capture eval args...]
  bash scripts/wsl_ml.sh sweep-rectified-scalar-crop-scale [sweep args...]
  bash scripts/wsl_ml.sh calibrate-obb-scalar-firmware [calibration args...]
  bash scripts/wsl_ml.sh export-prod-v0-2 [prod v0.2 export args...]
  bash scripts/wsl_ml.sh package-prod-v0-2 [prod v0.2 package args...]
  bash scripts/wsl_ml.sh export-prod-v0-3-obb [prod v0.3 obb export args...]
  bash scripts/wsl_ml.sh package-prod-v0-3-obb [prod v0.3 obb package args...]
  bash scripts/wsl_ml.sh export-rectifier [rectifier export args...]
  bash scripts/wsl_ml.sh package-rectifier [rectifier package args...]
  bash scripts/wsl_ml.sh export-rectified-scalar [rectified scalar export args...]
  bash scripts/wsl_ml.sh package-rectified-scalar [rectified scalar package args...]
  bash scripts/wsl_ml.sh train-geometry-cascade-localizer [geometry cascade args...]
  bash scripts/wsl_ml.sh eval-cascade [cascade args...]
  bash scripts/wsl_ml.sh fit-search [fit search args...]
  bash scripts/wsl_ml.sh export [export args...]
  bash scripts/wsl_ml.sh pytest [pytest args...]
EOF
}

cmd="${1:-}"
if [[ -n "${cmd}" ]]; then
  shift
fi

case "${cmd}" in
  setup)
    # Install the project and dev dependencies in the active WSL Poetry env.
    exec "${POETRY_BIN}" install --with dev
    ;;
  gpu-check)
    # Confirm that the WSL environment can see the GPU without relying on the
    # TensorFlow device probe, which can stall on some WSL stacks.
    if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
      timeout 10s /usr/lib/wsl/lib/nvidia-smi -L
    else
      echo "The WSL NVIDIA shim was not found at /usr/lib/wsl/lib/nvidia-smi." >&2
      exit 1
    fi
    exec "${POETRY_BIN}" run python -c "import tensorflow as tf; print(tf.__version__)"
    ;;
  baseline)
    # Run the classical polar spoke-voting baseline on the labelled dataset.
    exec "${POETRY_BIN}" run python scripts/run_classical_baseline.py "$@"
    ;;
  baseline-manifest)
    # Run the classical baseline on an arbitrary image/value manifest.
    exec "${POETRY_BIN}" run python scripts/eval_classical_baseline_on_manifest.py "$@"
    ;;
  single-image)
    # Run the classical polar baseline on one specific camera frame.
    exec "${POETRY_BIN}" run python scripts/run_single_image_baseline.py "$@"
    ;;
  train)
    # Run the CNN training pipeline on the same WSL Poetry environment.
    exec "${POETRY_BIN}" run python scripts/run_training.py "$@"
    ;;
  train-tiny)
    # Run the compressed MobileNetV2 training preset for STM32N6 deployment.
    exec "${POETRY_BIN}" run python scripts/run_training.py \
      --model-family mobilenet_v2_tiny \
      --mobilenet-alpha 0.35 \
      --mobilenet-head-units 64 \
      --mobilenet-head-dropout 0.15 \
      "$@"
    ;;
  train-compact-interval)
    # Run the compact CNN coarse-to-fine interval model on the board-style mix.
    exec "${POETRY_BIN}" run python scripts/run_training.py \
      --model-family compact_interval \
      --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
      --hard-case-repeat 6 \
      --edge-focus-strength 1.0 \
      --interval-bin-width 5.0 \
      --epochs 60 \
      "$@"
    ;;
  train-compact-geometry)
    # Run the compact CNN geometry-first model on the board-style mix.
    exec "${POETRY_BIN}" run python scripts/run_training.py \
      --model-family compact_geometry \
      --hard-case-manifest data/hard_cases_plus_board30_valid_with_new5.csv \
      --hard-case-repeat 6 \
      --edge-focus-strength 1.0 \
      --keypoint-heatmap-size 28 \
      --keypoint-heatmap-loss-weight 1.0 \
      --keypoint-coord-loss-weight 1.0 \
      --geometry-value-loss-weight 1.0 \
      --epochs 60 \
      "$@"
    ;;
  train-compact-geometry-longterm)
    # Train the compact CNN localizer with the pinned board-style split.
    exec bash scripts/run_compact_geometry_longterm.sh "$@"
    ;;
  train-compact-geometry-cascade-localizer)
    # Fine-tune the compact CNN geometry localizer used by the cascade.
    exec bash scripts/run_compact_geometry_cascade_localizer.sh "$@"
    ;;
  train-compact-geometry-cascade-localizer-longterm)
    # Fine-tune the compact CNN geometry localizer with the pinned board-style split.
    exec bash scripts/run_compact_geometry_cascade_localizer_longterm.sh "$@"
    ;;
  train-mobilenetv2-geometry-cascade-localizer-longterm)
    # Fine-tune the explicit MobileNetV2 geometry localizer with the pinned board-style split.
    exec bash scripts/run_mobilenetv2_geometry_cascade_localizer_longterm.sh "$@"
    ;;
  train-mobilenetv2-geometry-uncertainty)
    # Train the uncertainty-aware MobileNetV2 geometry reader on the board mix.
    exec bash scripts/run_mobilenetv2_geometry_uncertainty_full_range.sh "$@"
    ;;
  train-mobilenetv2-geometry-longterm)
    # Train the long-term MobileNetV2 geometry model with explicit val/test manifests.
    exec bash scripts/run_mobilenetv2_geometry_longterm.sh "$@"
    ;;
  train-mobilenetv2-direction-longterm)
    # Train the long-term MobileNetV2 direction model with the same pinned split.
    exec bash scripts/run_mobilenetv2_direction_longterm.sh "$@"
    ;;
  train-mobilenetv2-obb-longterm)
    # Train the long-term MobileNetV2 OBB localizer on the labeled dataset.
    exec bash scripts/run_mobilenetv2_obb_longterm.sh "$@"
    ;;
  train-mobilenetv2-rectifier)
    # Train the rectifier-first MobileNetV2 model on the labeled dataset.
    exec bash scripts/run_mobilenetv2_rectifier_full.sh "$@"
    ;;
  train-mobilenetv2-rectifier-finetune)
    # Warm-start the rectifier-first MobileNetV2 model from the current best checkpoint.
    exec bash scripts/run_mobilenetv2_rectifier_finetune.sh "$@"
    ;;
  train-mobilenetv2-rectifier-hardcase-finetune)
    # Fine-tune the rectifier-first MobileNetV2 model on the expanded hard-case mix.
    exec bash scripts/run_mobilenetv2_rectifier_hardcase_finetune_v3.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar)
    # Fine-tune the scalar reader on rectifier-generated crops.
    exec bash scripts/run_mobilenetv2_rectified_scalar_finetune.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-linear)
    # Fine-tune the scalar reader with a linear output head on rectified crops.
    exec bash scripts/run_mobilenetv2_rectified_scalar_linear_v10.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-tailbalanced)
    # Train a tail-balanced rectified scalar model with explicit sample weights.
    exec bash scripts/run_mobilenetv2_rectified_scalar_tailbalanced_v11.sh "$@"
    ;;
  train-mobilenetv2-synthetic-pretrain)
    # Pretrain a MobileNetV2 scalar model on synthetic gauge renders.
    exec bash scripts/run_mobilenetv2_synthetic_pretrain_v11.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-from-synth)
    # Fine-tune the rectified scalar model from the synthetic pretrain checkpoint.
    exec bash scripts/run_mobilenetv2_rectified_scalar_from_synth_v12.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-from-hard-synth-previewaug)
    # Fine-tune the rectified scalar model from the hard synthetic checkpoint with preview-heavy augmentation.
    exec bash scripts/run_mobilenetv2_rectified_scalar_from_hard_synth_previewaug_v14.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-full-labelled)
    # Fine-tune the rectified scalar model on the full CVAT-labelled raw pool.
    exec bash scripts/run_mobilenetv2_rectified_scalar_full_labelled_v18.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-from-geometry)
    # Fine-tune the strict rectified scalar model using geometry-derived crops.
    exec bash scripts/run_mobilenetv2_rectified_scalar_from_geometry_v19.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-all-data)
    # Train the V5-style rectified scalar model on synthetic + full real data.
    exec bash scripts/run_mobilenetv2_rectified_scalar_all_data_v21.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-combined)
    # Train the strict v5 scalar model on a weighted real + synthetic combined manifest.
    exec bash scripts/run_mobilenetv2_rectified_scalar_combined_v22.sh "$@"
    ;;
  train-mobilenetv2-sweep-distribution)
    # Train the sweep-distribution reader on the widest deduped real + synthetic pool.
    exec bash scripts/run_mobilenetv2_sweep_distribution_v35.sh "$@"
    ;;
  train-mobilenetv2-polar-sweep-distribution)
    # Train the polar sweep-distribution reader on the broad real + synthetic pool.
    exec bash scripts/run_mobilenetv2_polar_sweep_distribution_v36.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-all-sources)
    # Train the strict v5 scalar model on the widest deduped real + synthetic pool.
    exec bash scripts/run_mobilenetv2_rectified_scalar_all_sources_v24.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-hardtail-specialist)
    # Train a hard-tail specialist rectified scalar model from the strict v5 backbone.
    exec bash scripts/run_mobilenetv2_rectified_scalar_hardtail_specialist_v25.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-curriculum)
    # Train the strict v5 scalar model with a two-stage cold-tail curriculum.
    exec bash scripts/run_mobilenetv2_rectified_scalar_curriculum_v23.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-raw-tail)
    # Fine-tune the rectified scalar model on the strict pool plus raw tail rows with valid boxes.
    exec bash scripts/run_mobilenetv2_rectified_scalar_raw_tail_v17.sh "$@"
    ;;
  train-mobilenetv2-geometry-raw-cvat)
    # Train a geometry-first model directly on the raw CVAT-labelled gauge images.
    exec bash scripts/run_mobilenetv2_geometry_raw_cvat_v18.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-pure)
    # Fine-tune the rectified scalar reader with the cleaner pure-model recipe.
    exec bash scripts/run_mobilenetv2_rectified_scalar_pure_finetune_v3.sh "$@"
    ;;
  train-mobilenetv2-rectified-scalar-interval)
    # Fine-tune the rectified scalar reader with the interval auxiliary head.
    exec bash scripts/run_mobilenetv2_rectified_scalar_interval_v9.sh "$@"
    ;;
  train-mobilenetv2-dual-resolution-interval)
    # Train the new dual-resolution MobileNetV2 interval model on all sources.
    exec bash scripts/run_mobilenetv2_dual_resolution_interval_v26.sh "$@"
    ;;
  train-mobilenetv2-polar-dualview)
    # Train the new polar-dualview MobileNetV2 model on all sources.
    exec bash scripts/run_mobilenetv2_polar_dualview_v27.sh "$@"
    ;;
  train-mobilenetv2-polar-only)
    # Train the simpler single-input polar MobileNetV2 model on all sources.
    exec bash scripts/run_mobilenetv2_polar_only_v28.sh "$@"
    ;;
  train-mobilenetv2-geometry-from-rectified-v5)
    # Fine-tune a geometry-first model from the rectified v5 scalar backbone.
    exec bash scripts/run_mobilenetv2_geometry_from_rectified_v5.sh "$@"
    ;;
  train-mobilenetv2-geometry-literature)
    # Train the literature-backed geometry-first MobileNetV2 model.
    exec bash scripts/run_mobilenetv2_geometry_literature_v29.sh "$@"
    ;;
  train-mobilenetv2-geometry-localizer-only)
    # Train the geometry-only localizer used as stage one in the two-stage cascade.
    exec bash scripts/run_mobilenetv2_geometry_localizer_only_v30.sh "$@"
    ;;
  train-mobilenetv2-obb-localizer)
    # Train the OBB localizer used as the first stage in the two-stage cascade.
    exec bash scripts/run_mobilenetv2_obb_localizer_v31.sh "$@"
    ;;
  train-mobilenetv2-bluraware-obb-geometry)
    # Train the blur-aware OBB-geometry model with raw + sharpened views.
    exec bash scripts/run_mobilenetv2_bluraware_obb_geometry_v34.sh "$@"
    ;;
  train-mobilenetv2-keypoint-from-rectified-v5)
    # Fine-tune a lighter keypoint-aware model from the rectified v5 mix.
    exec bash scripts/run_mobilenetv2_keypoint_from_rectified_v5.sh "$@"
    ;;
  eval-mobilenetv2-geometry-cascade)
    # Evaluate the raw-CVAT geometry model as the cascade front-end.
    exec bash scripts/run_mobilenetv2_geometry_cascade_eval_v20.sh "$@"
    ;;
  eval-rectified-scalar)
    # Evaluate the rectifier + scalar-reader chain on a labeled manifest.
    exec bash scripts/run_rectified_scalar_eval.sh "$@"
    ;;
  eval-rectified-captures)
    # Evaluate the rectifier + scalar-reader chain on raw board captures.
    exec bash scripts/run_rectified_scalar_capture_eval.sh "$@"
    ;;
  eval-obb-scalar-board-probe)
    # Evaluate the OBB localizer + scalar-reader chain on the board probe set.
    exec bash scripts/run_obb_scalar_eval_board_probe.sh "$@"
    ;;
  sweep-rectified-scalar-crop-scale)
    # Sweep rectifier crop expansion factors against the board-style manifest.
    exec bash scripts/sweep_rectified_scalar_crop_scale.sh "$@"
    ;;
  calibrate-obb-scalar-firmware)
    # Fit deploy-time calibration for the prodv0.3 OBB + scalar cascade.
    exec bash scripts/run_calibrate_obb_scalar_firmware.sh "$@"
    ;;
  export-prod-v0-2)
    # Export the current prod-v0.2 scalar candidate to board-ready TFLite artifacts.
    exec bash scripts/run_board_export_prod_model_v0_2.sh "$@"
    ;;
  package-prod-v0-2)
    # Package the prod-v0.2 scalar candidate and refresh the canonical xSPI2 blob.
    exec bash scripts/run_board_package_prod_model_v0_2_raw_int8.sh "$@"
    ;;
  export-prod-v0-3-obb)
    # Export the prod-v0.3 OBB localizer to board-ready TFLite artifacts.
    exec bash scripts/run_board_export_prod_model_v0_3_obb.sh "$@"
    ;;
  package-prod-v0-3-obb)
    # Package the prod-v0.3 OBB localizer and refresh the canonical xSPI2 blob.
    exec bash scripts/run_board_package_prod_model_v0_3_obb_raw_int8.sh "$@"
    ;;
  export-rectifier)
    # Export the rectifier stage to board-ready TFLite artifacts.
    exec bash scripts/run_board_export_rectifier.sh "$@"
    ;;
  package-rectifier)
    # Package the rectifier stage and refresh the canonical rectifier xSPI2 blob.
    exec bash scripts/run_board_package_rectifier_raw_int8.sh "$@"
    ;;
  export-rectified-scalar)
    # Export the rectified scalar stage to board-ready TFLite artifacts.
    exec bash scripts/run_board_export_rectified_scalar.sh "$@"
    ;;
  package-rectified-scalar)
    # Package the rectified scalar stage and refresh the canonical xSPI2 blob.
    exec bash scripts/run_board_package_rectified_scalar_raw_int8.sh "$@"
    ;;
  train-geometry-cascade-localizer)
    # Fine-tune the geometry localizer used by the keypoint-gated cascade.
    exec bash scripts/run_mobilenetv2_geometry_cascade_localizer.sh "$@"
    ;;
  eval-cascade)
    # Evaluate the keypoint-gated cascade against a labeled manifest.
    exec "${POETRY_BIN}" run python scripts/eval_keypoint_reader_cascade_on_manifest.py "$@"
    ;;
  eval-mobilenetv2-geometry-cascade-strict-reader)
    # Evaluate the geometry localizer with the strict v5 scalar reader.
    exec bash scripts/run_geometry_localizer_strict_reader_cascade_v30.sh "$@"
    ;;
  eval-mobilenetv2-obb-geometry-cascade-strict-reader)
    # Evaluate the new OBB-geometry localizer with the strict v5 scalar reader.
    exec bash scripts/run_mobilenetv2_obb_geometry_cascade_eval_v32.sh "$@"
    ;;
  eval-mobilenetv2-obb-cascade-strict-reader)
    # Evaluate the OBB localizer with the strict scalar reader.
    exec bash scripts/run_obb_localizer_strict_reader_cascade_v31.sh "$@"
    ;;
  eval-mobilenetv2-bluraware-obb-geometry-self)
    # Evaluate the blur-aware OBB-geometry localizer using its own value head.
    exec bash scripts/run_mobilenetv2_bluraware_obb_geometry_self_cascade_v34.sh "$@"
    ;;
  sweep-obb-localizer-strict-reader-crop-scale)
    # Sweep OBB crop scales against the hard-case manifest.
    exec bash scripts/sweep_obb_localizer_strict_reader_crop_scale_v32.sh "$@"
    ;;
  fit-search)
    # Probe candidate MobileNetV2 widths against the STM32N6 relocatable fit.
    exec bash scripts/run_mobilenetv2_fit_search.sh "$@"
    ;;
  export)
    # Export the calibrated CNN to TFLite + board metadata.
    exec "${POETRY_BIN}" run python scripts/export_board_artifacts.py "$@"
    ;;
  pytest)
    # Run the ML test suite from the same environment used for training.
    exec "${POETRY_BIN}" run pytest "$@"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
esac
