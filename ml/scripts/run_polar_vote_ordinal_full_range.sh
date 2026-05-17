#!/usr/bin/env bash
set -euo pipefail

# Train the polar-vote backbone with an ordinal threshold head.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RUN_SUFFIX="${RUN_SUFFIX:-ordinal_full_range}"
export STRUCTURE_MODE=ordinal
export TARGET_MODE="${TARGET_MODE:-sweep}"

exec bash "${ROOT_DIR}/scripts/run_polar_vote_full_range_v8.sh"
