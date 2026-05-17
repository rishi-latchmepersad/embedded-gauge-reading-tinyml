#!/usr/bin/env bash
set -euo pipefail

# Train a sharpened polar voting head on the full real-data pool.
#
# This v8 pass keeps the angular axis explicit, but it tightens the loss and
# rebalances the manifest across the whole temperature range so cold and hot
# tails both matter. This variant folds in every hard-case manifest we have,
# because the earlier mixed pool still under-served the cold tail.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
RUN_SUFFIX="${RUN_SUFFIX:-v8_allhard}"
TARGET_MODE="${TARGET_MODE:-sweep}"
STRUCTURE_MODE="${STRUCTURE_MODE:-vote}"
COARSE_BINS="${COARSE_BINS:-16}"
FINE_BINS="${FINE_BINS:-14}"
LOG_FILE="${LOG_DIR}/polar_vote_full_range_${RUN_SUFFIX}.log"
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
MANIFEST_PATH="${TMP_DIR}/polar_vote_full_range_manifest_${RUN_SUFFIX}.csv"
BOARD_PROBE_TRAIN_MANIFEST="${TMP_DIR}/polar_vote_board_probe_train_${RUN_SUFFIX}.csv"
BOARD_PROBE_HOLDOUT_MANIFEST="${TMP_DIR}/polar_vote_board_probe_holdout_${RUN_SUFFIX}.csv"
HARD_CASES_MANIFEST="data/hard_cases_plus_board30_valid_with_new6.csv"
HARD_CASES_MANIFEST_PRIMARY="data/hard_cases_extreme_weighted_v4.csv"
HARD_CASES_MANIFEST_SECONDARY="data/hard_cases.csv"
HARD_CASES_MANIFEST_TAIL="data/hard_cases_remaining_focus.csv"
CAPTURED_MANIFEST="data/all_captured_images_manifest.csv"
FULL_LABELLED_MANIFEST="data/full_labelled_plus_board30_valid_with_new5.csv"
UNIFIED_MANIFEST="data/unified_training_manifest_v1.csv"
BOARD_PROBE_MANIFEST="data/board_rectified_probe_20260422.csv"
BOXES_CSV="data/rectified_crop_boxes_v5_all.csv"
OUTPUT_DIR="artifacts/training/polar_vote_full_range_${RUN_SUFFIX}"

echo "[WRAPPER] Starting polar vote full-range fine-tune ${RUN_SUFFIX}." | tee "${LOG_FILE}"
echo "[WRAPPER] Manifest:       ${MANIFEST_PATH}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Board probe:    ${BOARD_PROBE_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Hard cases:     ${HARD_CASES_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Hard cases 2:   ${HARD_CASES_MANIFEST_PRIMARY}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Hard cases 3:   ${HARD_CASES_MANIFEST_SECONDARY}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Hard cases 4:   ${HARD_CASES_MANIFEST_TAIL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Captures:       ${CAPTURED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Full labelled:  ${FULL_LABELLED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Unified pool:   ${UNIFIED_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Output dir:     ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Structure mode: ${STRUCTURE_MODE}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Coarse bins:    ${COARSE_BINS}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Fine bins:      ${FINE_BINS}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building the range-balanced real-data manifest..." | tee -a "${LOG_FILE}"
HARD_CASES_MANIFEST_PATH="${HARD_CASES_MANIFEST}" \
HARD_CASES_MANIFEST_PRIMARY_PATH="${HARD_CASES_MANIFEST_PRIMARY}" \
HARD_CASES_MANIFEST_SECONDARY_PATH="${HARD_CASES_MANIFEST_SECONDARY}" \
HARD_CASES_MANIFEST_TAIL_PATH="${HARD_CASES_MANIFEST_TAIL}" \
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
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    SourceManifest(Path(os.environ["HARD_CASES_MANIFEST_PRIMARY_PATH"]), default_weight=2.50),
    SourceManifest(Path(os.environ["HARD_CASES_MANIFEST_PATH"]), default_weight=2.00),
    SourceManifest(Path(os.environ["HARD_CASES_MANIFEST_SECONDARY_PATH"]), default_weight=2.25),
    SourceManifest(Path(os.environ["HARD_CASES_MANIFEST_TAIL_PATH"]), default_weight=2.75),
    SourceManifest(Path(os.environ["CAPTURED_MANIFEST_PATH"]), default_weight=0.75),
    SourceManifest(Path(os.environ["FULL_LABELLED_MANIFEST_PATH"]), default_weight=0.90),
    SourceManifest(Path(os.environ["UNIFIED_MANIFEST_PATH"]), default_weight=0.70),
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
all_rows: list[dict[str, str]] = []

for source in sources:
    count, rows = _iter_rows(source)
    kept = 0
    for row in rows:
        image_path = row["image_path"]
        if image_path in seen_paths:
            continue
        seen_paths.add(image_path)
        all_rows.append(row)
        kept += 1
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
    all_rows.append(row)
    kept += 1
print(
    f"[WRAPPER] Source {board_probe_train_path.name}: loaded={count} kept={kept} "
    f"default_weight=1.00"
)

if not all_rows:
    raise ValueError("Combined manifest is empty.")

df = pd.DataFrame(all_rows)
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce")
df = df.dropna(subset=["value", "sample_weight"]).reset_index(drop=True)

value_min = float(df["value"].min())
value_max = float(df["value"].max())
bin_lo = math.floor(value_min / 10.0) * 10.0
bin_hi = math.ceil(value_max / 10.0) * 10.0 + 10.0
bin_edges = np.arange(bin_lo, bin_hi, 10.0, dtype=np.float32)
temp_bins = pd.cut(df["value"], bins=bin_edges, include_lowest=True)
bin_counts = temp_bins.value_counts().sort_index()
median_count = float(bin_counts.median())
bin_weights = temp_bins.map(
    lambda b: float((median_count / float(bin_counts[b])) ** 0.5)
)
df["sample_weight"] = df["sample_weight"] * bin_weights.astype(np.float32)
df["sample_weight"] = df["sample_weight"].clip(lower=0.5, upper=3.0)
df["sample_weight"] = df["sample_weight"] / float(df["sample_weight"].mean())

with manifest_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "sample_weight"])
    writer.writeheader()
    for _, row in df.iterrows():
        writer.writerow(
            {
                "image_path": row["image_path"],
                "value": f"{float(row['value']):.6f}",
                "sample_weight": f"{float(row['sample_weight']):.6f}",
            }
        )

total_written = len(df)
print(f"[WRAPPER] Combined manifest written to {manifest_path} ({total_written} rows).")
print("[WRAPPER] Temperature bin counts:")
for bin_label, count in bin_counts.items():
    print(f"[WRAPPER]   {bin_label}: {int(count)}")
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
  --epochs 60 \
  --batch-size 16 \
  --learning-rate 1.5e-4 \
  --bins 224 \
  --sigma-bins 1.5 \
  --polar-size 224 \
  --head-units 96 \
  --base-filters 24 \
  --dropout 0.15 \
  --label-smoothing 0.0 \
  --representation vote \
  --structure-mode "${STRUCTURE_MODE}" \
  --target-mode "${TARGET_MODE}" \
  --coarse-bins "${COARSE_BINS}" \
  --fine-bins "${FINE_BINS}" \
  --seed 21 \
  --extra-eval-manifest "${BOARD_PROBE_HOLDOUT_MANIFEST}" \
  --extra-eval-manifest data/hard_cases_plus_board30_valid_with_new6.csv \
  2>&1 | tee -a "${LOG_FILE}"
