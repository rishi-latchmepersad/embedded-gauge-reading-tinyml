"""Run the classical polar gauge baseline and save artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

# Add `ml/src` to sys.path so this script works even before `poetry install`.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.baseline_runner import (
    ClassicalBaselineRunConfig,
    run_classical_baseline,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a reproducible baseline run."""
    parser = argparse.ArgumentParser(
        description="Run the classical polar gauge baseline."
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default="littlegood_home_temp_gauge_c",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for faster smoke runs.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run folder name. Defaults to timestamp.",
    )
    parser.add_argument(
        "--labelled-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "labelled",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate the classical baseline and print a compact run summary."""
    args = parse_args()

    config = ClassicalBaselineRunConfig(
        gauge_id=args.gauge_id,
        max_samples=args.max_samples,
        run_name=args.run_name,
        artifacts_dir=args.artifacts_dir,
        labelled_dir=args.labelled_dir,
        raw_dir=args.raw_dir,
    )

    run_result = run_classical_baseline(config)

    # Print a short summary so terminal runs stay easy to scan.
    print(f"Run directory: {run_result.run_dir}")
    print(f"Gauge spec: {asdict(run_result.spec)}")
    print(f"Label summary: {run_result.label_summary}")
    print(
        "Baseline metrics: "
        f"attempted={run_result.result.attempted_samples}, "
        f"successful={run_result.result.successful_samples}, "
        f"failed={run_result.result.failed_samples}, "
        f"mae={run_result.result.mae:.4f}, "
        f"rmse={run_result.result.rmse:.4f}"
    )


if __name__ == "__main__":
    main()
