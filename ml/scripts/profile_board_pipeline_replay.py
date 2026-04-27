"""Profile the board-pipeline replay on one laptop capture.

This small helper prints import, model-load, and inference timings so we can
see whether the laptop is spending time in startup, preprocessing, or one of
the TFLite stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Final

import numpy as np

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parent
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _parse_args() -> argparse.Namespace:
    """Parse the single-capture profiling CLI."""
    parser = argparse.ArgumentParser(
        description="Profile the STM32 board-pipeline replay on one capture."
    )
    parser.add_argument(
        "--capture-path",
        type=Path,
        default=REPO_ROOT / "captured_images" / "capture_2026-04-24_22-30-21.png",
        help="Capture to replay.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip the firmware affine calibration step.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the profile and print stage timings."""
    args = _parse_args()
    print("[DBG] importing board_pipeline...", flush=True)
    import_start = time.perf_counter()
    from embedded_gauge_reading_tinyml.board_pipeline import (  # noqa: E402
        DEFAULT_OBB_MODEL,
        DEFAULT_RECTIFIER_MODEL,
        DEFAULT_SCALAR_MODEL,
        InferenceBurstHistory,
        load_model_session,
        predict_board_pipeline_on_capture,
    )

    print(
        f"[DBG] board_pipeline import_s={time.perf_counter() - import_start:.3f}",
        flush=True,
    )
    print("[DBG] loading models...", flush=True)
    model_start = time.perf_counter()
    obb_session = load_model_session(DEFAULT_OBB_MODEL, "auto")
    rectifier_session = load_model_session(DEFAULT_RECTIFIER_MODEL, "auto")
    scalar_session = load_model_session(DEFAULT_SCALAR_MODEL, "auto")
    print(
        f"[DBG] models loaded_s={time.perf_counter() - model_start:.3f}",
        flush=True,
    )
    history = InferenceBurstHistory()

    def trace(message: str) -> None:
        """Emit a trace line for each replay stage."""
        print(f"[DBG] {message}", flush=True)

    print(f"[DBG] replaying {args.capture_path}", flush=True)
    replay_start = time.perf_counter()
    result = predict_board_pipeline_on_capture(
        args.capture_path,
        obb_session=obb_session,
        rectifier_session=rectifier_session,
        scalar_session=scalar_session,
        history=history,
        progress=trace,
        use_calibration=not args.no_calibration,
    )
    print(
        f"[DBG] replay_s={time.perf_counter() - replay_start:.3f} "
        f"reported={result.reported_prediction:.3f} selected={result.selected_stage}",
        flush=True,
    )


if __name__ == "__main__":
    main()
