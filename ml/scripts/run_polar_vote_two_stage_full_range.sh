#!/usr/bin/env bash
set -euo pipefail

# Train the polar-vote backbone with a coarse-to-fine two-stage head.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_SUFFIX="${RUN_SUFFIX:-two_stage_full_range}"
export STRUCTURE_MODE=two_stage
export TARGET_MODE="${TARGET_MODE:-sweep}"
export COARSE_BINS="${COARSE_BINS:-16}"
export FINE_BINS="${FINE_BINS:-14}"

exec bash "${ROOT_DIR}/scripts/run_polar_vote_full_range_v8.sh"
