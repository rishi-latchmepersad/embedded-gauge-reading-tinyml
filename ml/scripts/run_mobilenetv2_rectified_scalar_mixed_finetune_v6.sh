#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the tiny rectified scalar model on a broader weighted mix.
#
# This keeps the alpha=0.35 / head=64 backbone that worked on the strict
# rectified baseline, but adds hard-case and board30 coverage with source
# weights so the model sees the cold and hot ends instead of collapsing to the
# middle band.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_mixed_v6.log"
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

TRAIN_MANIFEST="data/rectified_scalar_mixed_train_v6.csv"
HOLDOUT_MANIFEST="data/hard_cases_plus_board30_valid_with_new5.csv"
BOXES_CSV="data/rectified_crop_boxes_v5_all.csv"
INIT_MODEL="artifacts/training/mobilenetv2_rectified_scalar_strict_v5/model.keras"

echo "[WRAPPER] Starting mixed rectified scalar fine-tune v6." | tee "${LOG_FILE}"
echo "[WRAPPER] Train manifest: ${TRAIN_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Holdout eval:   ${HOLDOUT_MANIFEST}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Boxes CSV:      ${BOXES_CSV}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Init model:     ${INIT_MODEL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:       ${LOG_FILE}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Building weighted mixed rectified training manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u - <<'PY'
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

root = Path.cwd()
out_manifest = root / "data" / "rectified_scalar_mixed_train_v6.csv"
holdout_manifest = root / "data" / "hard_cases_plus_board30_valid_with_new5.csv"

sources: list[tuple[str, float, Path]] = [
    ("hard_cases", 4.0, root / "data" / "hard_cases.csv"),
    ("hard_cases_board30", 3.0, root / "data" / "hard_cases_plus_board30.csv"),
    ("strict_rectified", 1.0, root / "data" / "rectified_scalar_strict_train_v5.csv"),
    ("full_board30_valid", 0.75, root / "data" / "full_labelled_plus_board30_valid_with_new5.csv"),
]

with holdout_manifest.open("r", encoding="utf-8", newline="") as handle:
    holdout_paths = {
        (row["image_path"].replace("\\", "/")).strip()
        for row in csv.DictReader(handle)
    }

seen: set[str] = set()
rows: list[dict[str, str]] = []
for source_name, source_weight, source_path in sources:
    if not source_path.exists():
        print(f"[WRAPPER] Skipping missing source: {source_path}")
        continue
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_path" not in reader.fieldnames or "value" not in reader.fieldnames:
            raise SystemExit(f"[WRAPPER] Bad manifest schema: {source_path}")
        for row in reader:
            image_path = (row["image_path"].replace("\\", "/")).strip()
            if image_path in holdout_paths:
                continue
            if image_path in seen:
                continue
            seen.add(image_path)
            rows.append(
                {
                    "image_path": image_path,
                    "value": str(row["value"]).strip(),
                    "sample_weight": f"{source_weight:.3f}",
                    "source_tag": source_name,
                }
            )

value_counts = Counter(row["value"] for row in rows)
if len(rows) < 200:
    raise SystemExit(f"[WRAPPER] Refusing to train on only {len(rows)} mixed rows")
if len(value_counts) < 12:
    raise SystemExit(
        f"[WRAPPER] Refusing collapsed mixed pool with only {len(value_counts)} unique labels"
    )

out_manifest.parent.mkdir(parents=True, exist_ok=True)
with out_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=["image_path", "value", "sample_weight", "source_tag"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"[WRAPPER] Wrote {len(rows)} mixed rows to {out_manifest}")
print(f"[WRAPPER] Unique labels: {len(value_counts)}")
print(f"[WRAPPER] Top labels: {value_counts.most_common(10)}")
PY

echo "[WRAPPER] Launching training..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir artifacts/training/mobilenetv2_rectified_scalar_mixed_v6 \
  --manifest-path "${TRAIN_MANIFEST}" \
  --precomputed-crop-boxes "${BOXES_CSV}" \
  --init-model "${INIT_MODEL}" \
  --batch-size 4 \
  --epochs 20 \
  --warmup-epochs 6 \
  --learning-rate 1e-4 \
  --fine-tune-lr 1e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --seed 21 \
  "$@" \
  2>&1 | tee -a "${LOG_FILE}"
