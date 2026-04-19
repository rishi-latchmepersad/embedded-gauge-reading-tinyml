#!/usr/bin/env bash
export PATH=/home/rishi_latchmepersad/.local/bin:$PATH
cd /mnt/d/Projects/embedded-gauge-reading-tinyml/ml
poetry run python -u scripts/debug_classical_cv.py
