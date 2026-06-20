#!/usr/bin/env python3
"""Entry point for the center-detector + SimCC KD/QAT training pipeline.

This thin wrapper keeps a discoverable script name for the new center-based
pipeline while reusing the implementation in the legacy-named training module.
"""

from __future__ import annotations

from train_qat_obb_simcc_combined import main as _train_main


def main() -> None:
    """Run the center-detector training pipeline."""

    _train_main()


if __name__ == "__main__":
    main()
