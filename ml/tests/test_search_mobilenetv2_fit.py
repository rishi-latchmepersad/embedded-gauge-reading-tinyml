"""Tests for the MobileNetV2 fit-search helper."""

from __future__ import annotations

from importlib import util
from pathlib import Path
from types import ModuleType


def _load_fit_search_module() -> ModuleType:
    """Load the fit-search script as a module for direct helper tests."""
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "search_mobilenetv2_fit.py"
    )
    spec = util.spec_from_file_location("search_mobilenetv2_fit", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_network_csv_reads_total_row(tmp_path: Path) -> None:
    """The fit-search parser should extract per-pool total bytes."""
    module = _load_fit_search_module()
    network_csv = tmp_path / "network.csv"
    network_csv.write_text(
        "epochs,hyperRAM (r),hyperRAM (w),hyperRAM (r+w),octoFlash (r+w),cpuRAM2 (r+w)\n"
        "Total,0,0,0,123,456\n",
        encoding="utf-8",
    )

    totals = module._parse_network_csv(network_csv)

    assert totals["hyperRAM"] == 0
    assert totals["octoFlash"] == 123
    assert totals["cpuRAM2"] == 456
