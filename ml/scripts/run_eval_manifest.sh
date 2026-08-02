#!/usr/bin/env bash
if [[ "${WSL_GUARDED:-0}" != "1" ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/run_wsl_guarded.sh" env WSL_GUARDED=1 bash "${BASH_SOURCE[0]}" "$@"
fi
export PATH="/home/rishi_latchmepersad/.local/bin:$PATH"
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
CUDA_VISIBLE_DEVICES=-1 poetry run python scripts/eval_keras_on_manifest.py \
  --model artifacts/training/scalar_full_finetune_closeup14c/model.keras \
  --manifest data/hard_cases_plus_board30_valid_with_new5_closeup14c.csv
