"""Compare classical-baseline geometry strategies across the hard-case manifests."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedded_gauge_reading_tinyml.baseline_manifest_eval import (  # noqa: E402
    GeometryEvaluationConfig,
    ManifestEvaluationResult,
    evaluate_manifest,
)
from embedded_gauge_reading_tinyml.baseline_report import write_failure_report  # noqa: E402
from embedded_gauge_reading_tinyml.baseline_classical_cv import (  # noqa: E402
    ClassicalPrediction,
)
from embedded_gauge_reading_tinyml.gauge.processing import GaugeSpec, load_gauge_specs

DEFAULT_MANIFESTS: tuple[Path, Path] = (
    PROJECT_ROOT / "data" / "hard_cases.csv",
    PROJECT_ROOT / "data" / "hard_cases_plus_board30_valid_with_new5.csv",
)


@dataclass(frozen=True)
class StrategySpec:
    """One detector strategy candidate to compare across manifests."""

    name: str
    config: GeometryEvaluationConfig


@dataclass(frozen=True)
class StrategyRunRow:
    """Per-manifest metrics for one strategy."""

    strategy_name: str
    manifest_name: str
    attempted_samples: int
    successful_samples: int
    failed_samples: int
    mae: float
    rmse: float
    cases_over_5c: int


@dataclass(frozen=True)
class StrategyComparisonRow:
    """Aggregated metrics for one strategy across all manifests."""

    strategy_name: str
    combined_attempted_samples: int
    combined_successful_samples: int
    combined_failed_samples: int
    combined_mae: float
    combined_rmse: float
    combined_cases_over_5c: int
    per_manifest_rows: list[StrategyRunRow]


@dataclass(frozen=True)
class StrategyAggregateMetrics:
    """Combined metrics across all manifests for one strategy."""

    attempted_samples: int
    successful_samples: int
    failed_samples: int
    mae: float
    rmse: float
    cases_over_5c: int
    predictions: list[ClassicalPrediction]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the strategy sweep."""
    parser = argparse.ArgumentParser(
        description="Sweep classical geometry strategies across the hard-case manifests."
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Optional output directory for per-strategy failure reports.",
    )
    parser.add_argument(
        "--gauge-id",
        type=str,
        default="littlegood_home_temp_gauge_c",
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="*",
        default=[3.0, 4.0, 5.0],
        help="Confidence thresholds to try for the Hough-plus-center fallback strategy.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help="Override the default manifests. Can be passed multiple times.",
    )
    return parser.parse_args()


def _build_strategy_specs(confidence_thresholds: list[float]) -> list[StrategySpec]:
    """Build the default set of strategy candidates to compare."""
    strategies: list[StrategySpec] = [
        StrategySpec(
            name="hough_only",
            config=GeometryEvaluationConfig(mode="hough_only"),
        ),
        StrategySpec(
            name="center_only",
            config=GeometryEvaluationConfig(mode="center_only"),
        ),
    ]
    for threshold in confidence_thresholds:
        strategies.append(
            StrategySpec(
                name=f"hough_then_center_t{threshold:g}",
                config=GeometryEvaluationConfig(
                    mode="hough_then_center",
                    confidence_threshold=threshold,
                ),
            )
        )
    return strategies


def _collect_metrics(
    results: list[ManifestEvaluationResult],
) -> StrategyAggregateMetrics:
    """Aggregate metrics from multiple manifest evaluations."""
    all_predictions: list[ClassicalPrediction] = []
    attempted_samples = 0
    successful_samples = 0
    failed_samples = 0
    cases_over_5c = 0
    for item in results:
        attempted_samples += item.attempted_samples
        successful_samples += item.result.successful_samples
        failed_samples += item.result.failed_samples
        all_predictions.extend(item.result.predictions)
        cases_over_5c += sum(pred.abs_error > 5.0 for pred in item.result.predictions)

    if not all_predictions:
        return StrategyAggregateMetrics(
            attempted_samples=attempted_samples,
            successful_samples=successful_samples,
            failed_samples=failed_samples,
            mae=float("nan"),
            rmse=float("nan"),
            cases_over_5c=cases_over_5c,
            predictions=[],
        )

    error_array: np.ndarray = np.asarray(
        [prediction.abs_error for prediction in all_predictions],
        dtype=np.float32,
    )
    return StrategyAggregateMetrics(
        attempted_samples=attempted_samples,
        successful_samples=successful_samples,
        failed_samples=failed_samples,
        mae=float(np.mean(error_array)),
        rmse=float(np.sqrt(np.mean(np.square(error_array)))),
        cases_over_5c=cases_over_5c,
        predictions=all_predictions,
    )


def _write_csv(output_path: Path, rows: list[StrategyRunRow]) -> None:
    """Write the strategy comparison table to CSV."""
    fieldnames = [
        "strategy_name",
        "manifest_name",
        "attempted_samples",
        "successful_samples",
        "failed_samples",
        "mae",
        "rmse",
        "cases_over_5c",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    """Run a strategy sweep across the hard-case manifests and print the ranking."""
    args = _parse_args()
    manifests = args.manifest if args.manifest is not None else list(DEFAULT_MANIFESTS)

    specs: dict[str, GaugeSpec] = load_gauge_specs()
    gauge_id: str = args.gauge_id
    if gauge_id not in specs:
        raise ValueError(f"Unknown gauge_id '{gauge_id}'. Available: {list(specs)}")
    spec: GaugeSpec = specs[gauge_id]

    strategies = _build_strategy_specs(args.confidence_thresholds)
    run_rows: list[StrategyRunRow] = []
    comparison_rows: list[StrategyComparisonRow] = []

    for strategy in strategies:
        per_manifest_results: list[ManifestEvaluationResult] = []
        per_manifest_rows: list[StrategyRunRow] = []

        for manifest_path in manifests:
            result = evaluate_manifest(
                manifest_path,
                spec,
                config=strategy.config,
                repo_root=PROJECT_ROOT.parent,
            )
            per_manifest_results.append(result)
            run_row = StrategyRunRow(
                strategy_name=strategy.name,
                manifest_name=manifest_path.name,
                attempted_samples=result.attempted_samples,
                successful_samples=result.result.successful_samples,
                failed_samples=result.result.failed_samples,
                mae=result.result.mae,
                rmse=result.result.rmse,
                cases_over_5c=sum(
                    prediction.abs_error > 5.0 for prediction in result.result.predictions
                ),
            )
            run_rows.append(run_row)
            per_manifest_rows.append(run_row)

        aggregate = _collect_metrics(per_manifest_results)
        comparison_rows.append(
            StrategyComparisonRow(
                strategy_name=strategy.name,
                combined_attempted_samples=aggregate.attempted_samples,
                combined_successful_samples=aggregate.successful_samples,
                combined_failed_samples=aggregate.failed_samples,
                combined_mae=aggregate.mae,
                combined_rmse=aggregate.rmse,
                combined_cases_over_5c=aggregate.cases_over_5c,
                per_manifest_rows=per_manifest_rows,
            )
        )

        if args.report_root is not None:
            strategy_dir = args.report_root / strategy.name
            for manifest_result in per_manifest_results:
                report_dir = strategy_dir / manifest_result.manifest_path.stem
                write_failure_report(
                    report_dir,
                    manifest_result.result.predictions,
                    attempted_samples=manifest_result.attempted_samples,
                    top_n=10,
                    value_bucket_size=10.0,
                )

    comparison_rows.sort(
        key=lambda row: (
            row.combined_failed_samples,
            row.combined_mae,
            row.combined_cases_over_5c,
        )
    )
    best = comparison_rows[0]
    print("=== Strategy Sweep ===")
    for row in comparison_rows:
        success_rate: float = (
            float(row.combined_successful_samples) / float(row.combined_attempted_samples)
            if row.combined_attempted_samples > 0
            else float("nan")
        )
        print(
            f"{row.strategy_name}: mae={row.combined_mae:.4f} "
            f"rmse={row.combined_rmse:.4f} cases_over_5c={row.combined_cases_over_5c} "
            f"successful={row.combined_successful_samples}/{row.combined_attempted_samples} "
            f"failed={row.combined_failed_samples} success_rate={success_rate:.3f}"
        )
    print(
        f"best={best.strategy_name} mae={best.combined_mae:.4f} "
        f"rmse={best.combined_rmse:.4f} cases_over_5c={best.combined_cases_over_5c}"
    )

    if args.report_root is not None:
        args.report_root.mkdir(parents=True, exist_ok=True)
        csv_path = args.report_root / "strategy_sweep.csv"
        _write_csv(csv_path, run_rows)
        json_path = args.report_root / "strategy_sweep.json"
        json_payload = {
            "gauge_id": gauge_id,
            "best_strategy": asdict(best),
            "per_strategy": [asdict(row) for row in comparison_rows],
        }
        json_path.write_text(
            json.dumps(json_payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"[SWEEP] Wrote {csv_path}")
        print(f"[SWEEP] Wrote {json_path}")


if __name__ == "__main__":
    main()
