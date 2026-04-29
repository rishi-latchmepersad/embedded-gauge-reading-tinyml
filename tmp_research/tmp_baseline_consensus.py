from __future__ import annotations

from pathlib import Path
import csv
import statistics
from typing import Callable

import cv2
import numpy as np

from embedded_gauge_reading_tinyml.baseline_classical_cv import (
    detect_needle_unit_vector,
    needle_vector_to_value,
)
from embedded_gauge_reading_tinyml.gauge.processing import load_gauge_specs
from embedded_gauge_reading_tinyml.single_image_baseline import _auto_geometry_candidates

spec = load_gauge_specs()["littlegood_home_temp_gauge_c"]
repo_root = Path("..")
manifests = [
    Path("data/hard_cases.csv"),
    Path("data/hard_cases_remaining_focus.csv"),
    Path("data/board_weak_focus.csv"),
]


def resolve_image_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def candidate_values(image_bgr: np.ndarray):
    candidates = []
    for cand in _auto_geometry_candidates(image_bgr):
        det = detect_needle_unit_vector(
            image_bgr,
            center_xy=cand.center_xy,
            dial_radius_px=cand.dial_radius_px,
            gauge_spec=spec,
        )
        if det is None:
            continue
        val = needle_vector_to_value(det.unit_dx, det.unit_dy, spec)
        quality = det.confidence * max(det.peak_ratio - 1.0, 0.0)
        candidates.append((val, quality, cand.label, det.confidence, det.peak_ratio))
    return candidates


def _weighted_mean(cands) -> float:
    """Return the quality-weighted mean candidate value."""
    values = np.asarray([item[0] for item in cands], dtype=np.float32)
    weights = np.asarray([max(item[1], 1e-6) for item in cands], dtype=np.float32)
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_median(cands) -> float:
    """Return the quality-weighted median candidate value."""
    pairs = sorted(((float(item[0]), float(max(item[1], 1e-6))) for item in cands), key=lambda item: item[0])
    total_weight = sum(weight for _value, weight in pairs)
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= 0.5 * total_weight:
            return value
    return pairs[-1][0]


def _trimmed_mean(cands, *, trim_fraction: float = 0.25) -> float:
    """Return the quality-weighted mean after trimming the weakest candidates."""
    ordered = sorted(cands, key=lambda item: item[1], reverse=True)
    keep_count = max(1, int(round(len(ordered) * (1.0 - trim_fraction))))
    keep = ordered[:keep_count]
    return _weighted_mean(keep)


def _density_mode(cands, *, sigma_c: float = 4.0) -> float:
    """Return the candidate value with the strongest quality-weighted KDE support."""
    values = np.asarray([item[0] for item in cands], dtype=np.float32)
    weights = np.asarray([max(item[1], 1e-6) for item in cands], dtype=np.float32)
    scores = []
    for index, value in enumerate(values):
        kernel = np.exp(-0.5 * np.square((values - value) / sigma_c))
        scores.append(float(weights[index] * np.sum(weights * kernel)))
    return float(values[int(np.argmax(np.asarray(scores, dtype=np.float32)))])


def strategy_best_quality(cands):
    return max(cands, key=lambda item: item[1])[0]


def strategy_median_all(cands):
    return float(statistics.median([item[0] for item in cands]))


def strategy_mean_all(cands):
    return float(sum(item[0] for item in cands) / len(cands))


def strategy_median_top3(cands):
    top = sorted(cands, key=lambda item: item[1], reverse=True)[:3]
    return float(statistics.median([item[0] for item in top]))


def strategy_mean_top3(cands):
    top = sorted(cands, key=lambda item: item[1], reverse=True)[:3]
    return float(sum(item[0] for item in top) / len(top))


def strategy_weighted_mean(cands):
    return _weighted_mean(cands)


def strategy_weighted_median(cands):
    return _weighted_median(cands)


def strategy_trimmed_mean(cands):
    return _trimmed_mean(cands, trim_fraction=0.25)


def strategy_density_mode(cands):
    return _density_mode(cands, sigma_c=4.0)


def strategy_consensus(cands):
    # Find the median temperature, then pick the strongest candidate near it.
    temps = [item[0] for item in cands]
    median_temp = float(statistics.median(temps))
    max_quality = max(item[1] for item in cands)
    near = [item for item in cands if abs(item[0] - median_temp) <= 12.0 and item[1] >= 0.25 * max_quality]
    if near:
        return max(near, key=lambda item: (item[1], -abs(item[0] - median_temp)))[0]
    return max(cands, key=lambda item: (item[1], -abs(item[0] - median_temp)))[0]


strategies: dict[str, Callable[[list[tuple[float, float, str, float, float]]], float]] = {
    "best_quality": strategy_best_quality,
    "median_all": strategy_median_all,
    "mean_all": strategy_mean_all,
    "median_top3": strategy_median_top3,
    "mean_top3": strategy_mean_top3,
    "weighted_mean": strategy_weighted_mean,
    "weighted_median": strategy_weighted_median,
    "trimmed_mean": strategy_trimmed_mean,
    "density_mode": strategy_density_mode,
    "consensus": strategy_consensus,
}

for manifest_path in manifests:
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    print(f"\n=== {manifest_path.name} ===")
    for name, fn in strategies.items():
        errors = []
        success = 0
        for row in rows:
            image_path = resolve_image_path(row["image_path"])
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            cands = candidate_values(img)
            if not cands:
                continue
            pred = fn(cands)
            true = float(row["value"])
            errors.append(abs(pred - true))
            success += 1
        mae = float(np.mean(errors)) if errors else float("nan")
        print(f"{name:14s} success={success:3d}/{len(rows):3d} mae={mae:8.4f}")
