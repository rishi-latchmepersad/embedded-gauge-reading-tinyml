#!/usr/bin/env bash
if [[ "${WSL_GUARDED:-0}" != "1" ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/run_wsl_guarded.sh" env WSL_GUARDED=1 bash "${BASH_SOURCE[0]}" "$@"
fi
export PATH=/home/rishi_latchmepersad/.local/bin:$PATH
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
poetry run python -u scripts/debug_classical_cv.py
