#!/usr/bin/env bash
set -euo pipefail

# Train a learned polar voting head on the full real-data pool.
#
# This keeps the same full manifest mix as the earlier polar runs, but uses a
# vote-preserving polar architecture that keeps the angular axis explicit until
# the final soft-argmax. That is the closest learned analogue to the classical
# polar spoke accumulator we have not yet tried at full range.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/polar_vote_full_range_v4.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

cd "${ROOT_DIR}"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

TMP_DIR="../tmp"
MANIFEST_PATH="${TMP_DIR}/polar_vote_full_range_manifest_v4.csv"
BOARD_PROBE_TRAIN_MANIFEST="${TMP_DIR}/polar_vote_board_probe_train_v4.csv"
BOARD_PROBE_HOLDOUT_MANIFEST="${TMP_DIR}/polar_vote_board_probe_holdout_v4.csv"
HARD_CASES_MANIFEST="data/hard_cases_plus_board30_valid_with_new6.csv"
CAPTURED_MANIFEST="data/all_captured_images_manifest.csv"
FULL_LABELLED_MANIFEST="data/full_labelled_plus_board30_valid_with_new5.csv"
UNIFIED_MANIFEST="data/unified_training_manifest_v1.csv"
BOARD_PROBE_MANIFEST="data/board_rectified_probe_20260422.csv"
BOXES_CSV="data/rectified_crop_boxes_v5_all.csv"
OUTPUT_DIR="artifacts/training/polar_vote_full_range_v4"

echo "[WRAPPER] Starting polar vote full-range fine-tune v4." | tee "${LOG_FILE}"
echo "[WRAPPER] Manifest:       ${MANIFEST_PATH}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Board probe:    ${BOARD_PROBE_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Hard cases:     ${HARD_CASES_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Captures:       ${CAPTURED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Full labelled:  ${FULL_LABELLED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Unified pool:   ${UNIFIED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir:     ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building the deduplicated real-data manifest..." | tee -a "${LOG_FILE}"
HARD_CASES_MANIFEST_PATH="${HARD_CASES_MANIFEST}" \
CAPTURED_MANIFEST_PATH="${CAPTURED_MANIFEST}" \
FULL_LABELLED_MANIFEST_PATH="${FULL_LABELLED_MANIFEST}" \
UNIFIED_MANIFEST_PATH="${UNIFIED_MANIFEST}" \
BOARD_PROBE_MANIFEST_PATH="${BOARD_PROBE_MANIFEST}" \
MANIFEST_PATH="${MANIFEST_PATH}" \
BOARD_PROBE_TRAIN_MANIFEST_PATH="${BOARD_PROBE_TRAIN_MANIFEST}" \
BOARD_PROBE_HOLDOUT_MANIFEST_PATH="${BOARD_PROBE_HOLDOUT_MANIFEST}" \
"${POETRY_BIN}" run python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
"""Merge the real manifests into one weighted training pool."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SourceManifest:
    """One manifest source and its default sample weight."""

    path: Path
    default_weight: float


def _iter_rows(source: SourceManifest) -> tuple[int, list[dict[str, str]]]:
    """Load rows from one source manifest using UTF-8 BOM tolerant parsing."""
    rows: list[dict[str, str]] = []
    with source.path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {source.path}")
        for row in reader:
            image_path = str(row.get("image_path", "")).strip()
            value = str(row.get("value", "")).strip()
            if not image_path or not value:
                continue
            weight_raw = str(row.get("sample_weight", "")).strip()
            rows.append(
                {
                    "image_path": image_path.replace("\\", "/"),
                    "value": value,
                    "sample_weight": weight_raw or f"{source.default_weight:.3f}",
                }
            )
    return len(rows), rows


sources = [
    SourceManifest(Path(os.environ["HARD_CASES_MANIFEST_PATH"]), default_weight=1.75),
    SourceManifest(Path(os.environ["CAPTURED_MANIFEST_PATH"]), default_weight=0.90),
    SourceManifest(Path(os.environ["FULL_LABELLED_MANIFEST_PATH"]), default_weight=1.00),
    SourceManifest(Path(os.environ["UNIFIED_MANIFEST_PATH"]), default_weight=0.85),
]

manifest_path = Path(os.environ["MANIFEST_PATH"])
board_probe_train_path = Path(os.environ["BOARD_PROBE_TRAIN_MANIFEST_PATH"])
board_probe_holdout_path = Path(os.environ["BOARD_PROBE_HOLDOUT_MANIFEST_PATH"])
manifest_path.parent.mkdir(parents=True, exist_ok=True)
board_probe_train_path.parent.mkdir(parents=True, exist_ok=True)

board_probe_path = Path(os.environ["BOARD_PROBE_MANIFEST_PATH"])
board_probe_df = pd.read_csv(board_probe_path)
if "image_path" not in board_probe_df.columns or "value" not in board_probe_df.columns:
    raise ValueError(f"Board probe manifest is missing required columns: {board_probe_path}")

board_probe_df = board_probe_df.copy()
board_probe_df["image_path"] = board_probe_df["image_path"].astype(str).str.replace("\\", "/", regex=False)
board_probe_df["value"] = pd.to_numeric(board_probe_df["value"], errors="coerce")
board_probe_df = board_probe_df.dropna(subset=["image_path", "value"])
board_probe_df = board_probe_df.reset_index(drop=True)

if len(board_probe_df) < 2:
    raise ValueError(f"Board probe manifest too small to split: {board_probe_path}")

train_board_probe_df, holdout_board_probe_df = train_test_split(
    board_probe_df,
    test_size=0.25,
    random_state=21,
    stratify=board_probe_df["value"],
)

train_board_probe_df = train_board_probe_df.copy()
train_board_probe_df["sample_weight"] = 1.00
holdout_board_probe_df = holdout_board_probe_df.copy()
holdout_board_probe_df["sample_weight"] = 1.00

train_board_probe_df.to_csv(board_probe_train_path, index=False)
holdout_board_probe_df.to_csv(board_probe_holdout_path, index=False)
print(
    f"[WRAPPER] Board probe split: train={len(train_board_probe_df)} "
    f"holdout={len(holdout_board_probe_df)}"
)

seen_paths: set[str] = set()
total_written = 0
with manifest_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "sample_weight"])
    writer.writeheader()

    for source in sources:
        count, rows = _iter_rows(source)
        kept = 0
        for row in rows:
            image_path = row["image_path"]
            if image_path in seen_paths:
                continue
            seen_paths.add(image_path)
            writer.writerow(row)
            kept += 1
        total_written += kept
        print(
            f"[WRAPPER] Source {source.path.name}: loaded={count} kept={kept} "
            f"default_weight={source.default_weight:.2f}"
        )

    count, rows = _iter_rows(SourceManifest(board_probe_train_path, default_weight=1.00))
    kept = 0
    for row in rows:
        image_path = row["image_path"]
        if image_path in seen_paths:
            continue
        seen_paths.add(image_path)
        writer.writerow(row)
        kept += 1
    total_written += kept
    print(
        f"[WRAPPER] Source {board_probe_train_path.name}: loaded={count} kept={kept} "
        f"default_weight=1.00"
    )

    print(f"[WRAPPER] Combined manifest written to {manifest_path} ({total_written} rows).")
PY

echo "[WRAPPER] Training the polar vote reader..." | tee -a "${LOG_FILE}"
VENV_DIR="$("${POETRY_BIN}" env info -p)"
# Source the Poetry venv so the WSL GPU library paths are restored before TF starts.
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

NVIDIA_LIB_DIRS=("${VENV_DIR}"/lib/python*/site-packages/nvidia/*/lib)
if (( ${#NVIDIA_LIB_DIRS[@]} > 0 )); then
  NVIDIA_LIB_PATH="$(IFS=:; echo "${NVIDIA_LIB_DIRS[*]}")"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${NVIDIA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

python -u scripts/train_polar_angle_classifier_manifest.py \
  --manifest-path "${MANIFEST_PATH}" \
  --crop-boxes "${BOXES_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --bins 224 \
  --sigma-bins 4.0 \
  --polar-size 224 \
  --head-units 96 \
  --base-filters 24 \
  --dropout 0.15 \
  --representation vote \
  --seed 21 \
  --extra-eval-manifest "${BOARD_PROBE_HOLDOUT_MANIFEST}" \
  --extra-eval-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
  2>&1 | tee -a "${LOG_FILE}"
