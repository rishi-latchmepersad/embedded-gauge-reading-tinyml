#!/usr/bin/env bash
# Run one ML command with host-memory and process-concurrency guardrails.
#
# Usage from ml/:
#   scripts/run_wsl_guarded.sh poetry run python scripts/train_...py --batch-size 4
#
# The guard is intentionally outside Python: it also protects conversion,
# evaluation, and shell-launched jobs that do not configure TensorFlow.

set -euo pipefail

if (($# == 0)); then
    printf 'usage: %s COMMAND [ARG... ]\n' "$0" >&2
    exit 64
fi

# Keep enough RAM available for WSL itself, the shell, and the active editor.
# Override only after checking the machine's actual memory size.
MIN_AVAILABLE_MB="${WSL_MIN_AVAILABLE_MB:-2048}"
CHECK_INTERVAL_SECONDS="${WSL_MEMORY_CHECK_INTERVAL_SECONDS:-5}"
GPU_MEMORY_LIMIT_MB="${TF_GPU_MEMORY_LIMIT_MB:-12000}"
LOCK_FILE="${WSL_TRAINING_LOCK_FILE:-/tmp/embedded-gauge-reading-tinyml-training.lock}"

case "$MIN_AVAILABLE_MB" in (*[!0-9]*|'') printf 'WSL_MIN_AVAILABLE_MB must be an integer\n' >&2; exit 64;; esac
case "$CHECK_INTERVAL_SECONDS" in (*[!0-9]*|'') printf 'WSL_MEMORY_CHECK_INTERVAL_SECONDS must be an integer\n' >&2; exit 64;; esac
case "$GPU_MEMORY_LIMIT_MB" in (*[!0-9]*|'') printf 'TF_GPU_MEMORY_LIMIT_MB must be an integer\n' >&2; exit 64;; esac

# flock makes accidental parallel launches fail instead of multiplying RAM use.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf 'another guarded ML job is already running (lock: %s)\n' "$LOCK_FILE" >&2
    exit 75
fi

available_memory_mb() {
    # MemAvailable is the useful signal here; free memory alone excludes cache
    # that Linux can reclaim, while MemAvailable estimates reclaimable headroom.
    awk '/^MemAvailable:/ { printf "%d", $2 / 1024; exit }' /proc/meminfo
}

available_mb="$(available_memory_mb)"
if ((available_mb < MIN_AVAILABLE_MB)); then
    printf 'refusing to start: only %s MiB available, need %s MiB\n' \
        "$available_mb" "$MIN_AVAILABLE_MB" >&2
    exit 75
fi

# Bound CPU-side TensorFlow and image-loader fan-out. AUTOTUNE can otherwise
# create enough workers and prefetched batches to exhaust WSL host RAM.
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-2}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export TF_GPU_MEMORY_LIMIT_MB="$GPU_MEMORY_LIMIT_MB"

printf 'guarded ML job: %s\n' "$*"
printf 'memory floor: %s MiB; GPU budget: %s MiB; CPU threads: %s/%s\n' \
    "$MIN_AVAILABLE_MB" "$GPU_MEMORY_LIMIT_MB" "$TF_NUM_INTRAOP_THREADS" "$TF_NUM_INTEROP_THREADS"

# Start a separate process group so the watchdog can terminate all descendants
# (Python, TensorFlow helpers, and converter subprocesses) together.
setsid --wait "$@" &
child_pid=$!

cleanup() {
    # The child may already have exited; kill is deliberately best effort.
    kill -TERM -- "-$child_pid" 2>/dev/null || true
}
trap cleanup INT TERM

while kill -0 "$child_pid" 2>/dev/null; do
    available_mb="$(available_memory_mb)"
    if ((available_mb < MIN_AVAILABLE_MB)); then
        printf 'memory floor breached: %s MiB available; stopping job\n' "$available_mb" >&2
        kill -TERM -- "-$child_pid" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
        exit 137
    fi
    sleep "$CHECK_INTERVAL_SECONDS"
done

wait "$child_pid"
