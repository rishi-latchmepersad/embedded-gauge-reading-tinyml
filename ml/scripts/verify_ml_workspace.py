#!/usr/bin/env python3
"""Verify the source inputs and optional artifacts for an ML workspace."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib
from typing import Any


ML_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ML_ROOT / "workspace_manifest.toml"


def _load_manifest() -> dict[str, Any]:
    """Load the checked-in workspace manifest."""

    with MANIFEST_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _check_entries(entries: list[dict[str, Any]], *, required: bool) -> int:
    """Print the state of manifest entries and return the missing count."""

    missing = 0
    for entry in entries:
        relative_path = Path(str(entry["path"]))
        path = ML_ROOT / relative_path
        present = path.exists()
        label = "OK" if present else ("MISSING" if required else "OPTIONAL")
        print(f"[{label}] {relative_path} - {entry['purpose']}")
        if required and not present:
            missing += 1
    return missing


def main() -> int:
    """Validate required inputs and optionally require model artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="fail when optional model and ST Edge AI artifacts are absent",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    workspace = manifest["workspace"]
    print(
        f"workspace schema={workspace['schema_version']} "
        f"board={workspace['active_board_contract']} "
        f"candidate={workspace['research_candidate']}"
    )
    missing = _check_entries(manifest["required_inputs"], required=True)
    missing += _check_entries(
        manifest["optional_artifacts"], required=args.require_artifacts
    )
    if missing:
        print(f"workspace verification failed: {missing} required entries missing")
        return 1
    print("workspace verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
