#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the rectified scalar model on the strict captured pool plus raw
# tail rows that already have valid crop boxes.
#
# This keeps the best preview-heavy base, but feeds the network real cold/hot
# raw captures so it can see the missing extremes instead of only preview-style
# examples.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/training_logs"
LOG_FILE="${LOG_DIR}/mobilenetv2_rectified_scalar_raw_tail_v17.log"
POETRY_BIN="${POETRY_BIN:-${HOME}/.local/bin/poetry}"
BASE_MODEL_SRC="${ROOT_DIR}/artifacts/training/mobilenetv2_rectified_scalar_from_hard_synth_previewaug_v14/model.keras"
BASE_MODEL_LOCAL="${HOME}/ml_eval_cache/mobilenetv2_rectified_scalar_from_hard_synth_previewaug_v14.model.keras"

if [[ ! -x "${POETRY_BIN}" ]]; then
  POETRY_BIN="$(command -v poetry || true)"
fi

if [[ -z "${POETRY_BIN}" ]]; then
  echo "[WRAPPER] Poetry was not found in WSL." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${BASE_MODEL_LOCAL}")"
cp -f "${BASE_MODEL_SRC}" "${BASE_MODEL_LOCAL}"
: > "${LOG_FILE}"

cd "${ROOT_DIR}"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

echo "[WRAPPER] Starting raw-tail rectified fine-tune v17." | tee "${LOG_FILE}"
echo "[WRAPPER] Base model: ${BASE_MODEL_LOCAL}" | tee -a "${LOG_FILE}"
echo "[WRAPPER] Log file:   ${LOG_FILE}" | tee -a "${LOG_FILE}"

echo "[WRAPPER] Rebuilding strict + raw-tail manifest..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u - <<'PY'
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

root = Path.cwd()
strict_manifest = root / "data" / "rectified_scalar_strict_train_v5.csv"
full_manifest = root / "data" / "full_scalar_manifest_v1.csv"
held_out_manifest = root / "data" / "hard_cases_plus_board30_valid_with_new5.csv"
crop_boxes_path = root / "data" / "rectified_crop_boxes_v5_all.csv"
out_manifest = root / "data" / "rectified_scalar_raw_tail_v17.csv"


def _normalize(path_str: str) -> str:
    return path_str.replace("\\", "/").strip()


held_out_rows: set[str] = set()
with held_out_manifest.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        held_out_rows.add(_normalize(row["image_path"]))

crop_box_paths: set[str] = set()
with crop_boxes_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        crop_box_paths.add(_normalize(row["image_path"]))

kept_rows: list[dict[str, str]] = []
with strict_manifest.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        image_path = _normalize(row["image_path"])
        if image_path in held_out_rows:
            continue
        kept_rows.append(
            {
                "image_path": image_path,
                "value": row["value"],
                "sample_weight": "1.0",
            }
        )

tail_rows: list[dict[str, str]] = []
with full_manifest.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        image_path = _normalize(row["image_path"])
        if image_path in held_out_rows:
            continue
        if image_path not in crop_box_paths:
            continue
        try:
            value = float(row["value"])
        except Exception:
            continue
        if value <= 10.0 or value >= 40.0:
            # Repeat the true cold/hot rows so they matter during fine-tuning.
            repeat = 4 if value <= 10.0 else 5
            for _ in range(repeat):
                tail_rows.append(
                    {
                        "image_path": image_path,
                        "value": row["value"],
                        "sample_weight": "4.0" if value <= 10.0 else "5.0",
                    }
                )

all_rows = kept_rows + tail_rows
if len(all_rows) < 100:
    raise SystemExit(f"[WRAPPER] Refusing to train on only {len(all_rows)} rows")

value_counts = Counter(row["value"] for row in all_rows)
if len(value_counts) < 6:
    raise SystemExit(
        f"[WRAPPER] Refusing collapsed rectified pool with only {len(value_counts)} unique labels"
    )

out_manifest.parent.mkdir(parents=True, exist_ok=True)
with out_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["image_path", "value", "sample_weight"])
    writer.writeheader()
    writer.writerows(all_rows)

print(
    f"[WRAPPER] Wrote {len(all_rows)} rows to {out_manifest} "
    f"(strict={len(kept_rows)}, tail={len(tail_rows)})"
)
PY

echo "[WRAPPER] Launching fine-tune..." | tee -a "${LOG_FILE}"
"${POETRY_BIN}" run python -u scripts/train_all_data_baseline.py \
  --output-dir "artifacts/training/mobilenetv2_rectified_scalar_raw_tail_v17" \
  --manifest-path "data/rectified_scalar_raw_tail_v17.csv" \
  --precomputed-crop-boxes "data/rectified_crop_boxes_v5_all.csv" \
  --init-model "${BASE_MODEL_LOCAL}" \
  --linear-output \
  --augment-mode hard_preview \
  --batch-size 4 \
  --epochs 16 \
  --warmup-epochs 6 \
  --learning-rate 5e-5 \
  --fine-tune-lr 2e-6 \
  --mobilenet-unfreeze-last-n 12 \
  --mobilenet-freeze-batchnorm \
  --alpha 0.35 \
  --head-units 64 \
  --dropout 0.15 \
  --seed 21 \
  2>&1 | tee -a "${LOG_FILE}"
