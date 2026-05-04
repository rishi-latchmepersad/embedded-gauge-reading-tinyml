"""Monitor training progress for the hardcase interval model.

Usage:
    python scripts/monitor_training.py [run_dir]

If run_dir is not specified, finds the most recent hardcase_interval run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def find_latest_run(artifacts_dir: Path) -> Path | None:
    """Find the most recent hardcase_interval training run."""
    runs = sorted(
        [d for d in artifacts_dir.glob("hardcase_interval_*") if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def load_metrics(run_dir: Path) -> dict | None:
    """Load metrics.json if it exists."""
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def load_history(run_dir: Path) -> dict | None:
    """Load history.json if it exists."""
    history_path = run_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return None


def load_config(run_dir: Path) -> dict | None:
    """Load config.json if it exists."""
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def monitor(run_dir: Path, watch: bool = False) -> None:
    """Print training status for the given run directory."""
    config = load_config(run_dir)
    history = load_history(run_dir)
    metrics = load_metrics(run_dir)

    print("=" * 60)
    print(f"Monitoring: {run_dir.name}")
    print("=" * 60)

    if config:
        print(f"Model family: {config.get('model_family', 'N/A')}")
        print(f"Epochs:       {config.get('epochs', 'N/A')}")
        print(f"Batch size:   {config.get('batch_size', 'N/A')}")
        print(f"Learning rate: {config.get('learning_rate', 'N/A')}")
        print(f"Device:       {config.get('device', 'N/A')}")
        print()

    if history:
        epochs_done = len(history.get("loss", []))
        total_epochs = config.get("epochs", epochs_done) if config else epochs_done
        print(f"Progress: {epochs_done}/{total_epochs} epochs")

        if epochs_done > 0:
            latest = epochs_done - 1
            print(f"\nLatest epoch ({latest + 1}):")
            print(f"  loss:        {history['loss'][latest]:.4f}")
            print(f"  val_loss:    {history.get('val_loss', [None])[latest]:.4f}")
            print(
                f"  mae:         {history.get('gauge_value_mae', history.get('mae', [None]))[latest]:.4f}"
            )
            print(
                f"  val_mae:     {history.get('val_gauge_value_mae', history.get('val_mae', [None]))[latest]:.4f}"
            )

            # Show best validation MAE so far
            val_mae_key = (
                "val_gauge_value_mae" if "val_gauge_value_mae" in history else "val_mae"
            )
            if val_mae_key in history and history[val_mae_key]:
                best_epoch = min(
                    range(len(history[val_mae_key])),
                    key=lambda i: history[val_mae_key][i],
                )
                print(
                    f"\nBest val MAE: {history[val_mae_key][best_epoch]:.4f} at epoch {best_epoch + 1}"
                )

    if metrics:
        print(
            f"\nFinal test MAE: {metrics.get('mae', metrics.get('gauge_value_mae', 'N/A'))}"
        )
        print(
            f"Final test RMSE: {metrics.get('rmse', metrics.get('gauge_value_rmse', 'N/A'))}"
        )

    if not history and not metrics:
        print("No history or metrics found yet. Training may have just started.")

    print("=" * 60)

    if watch:
        print("\nWatching for updates (Ctrl+C to stop)...")
        last_mtime = 0
        try:
            while True:
                history_path = run_dir / "history.json"
                if history_path.exists():
                    mtime = history_path.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        print(f"\n[{time.strftime('%H:%M:%S')}] Update detected:")
                        monitor(run_dir, watch=False)
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped watching.")


def main() -> None:
    """Parse args and monitor training."""
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts" / "training"
        run_dir = find_latest_run(artifacts_dir)
        if run_dir is None:
            print("No hardcase_interval runs found in artifacts/training/")
            sys.exit(1)

    watch = "--watch" in sys.argv or "-w" in sys.argv
    monitor(run_dir, watch=watch)


if __name__ == "__main__":
    main()
