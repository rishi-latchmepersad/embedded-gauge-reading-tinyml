"""Ensure the src/ layout is importable when pytest runs without editable install."""

from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR: Path = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
