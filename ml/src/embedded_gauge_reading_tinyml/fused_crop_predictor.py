"""Shared helpers for crop-fusion experiments.

This module keeps the path resolution logic in one place so offline scripts can
load manifests that mix `ml/data/raw/...`, `ml/data/captured_images/...`, and
already-resolved absolute paths without tripping over the caller's working
directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

ML_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]


def resolve_dataset_image_path(
    raw_path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
    ml_root: Path = ML_ROOT,
) -> Path:
    """Resolve a dataset image path against the repo layout.

    The manifests in this repository are not fully uniform. Some rows point to
    `ml/data/raw/...`, some to `ml/data/captured_images/...`, and some scripts
    are run from the `ml/` directory while others run from the repository root.
    This helper tries the sensible absolute forms first and returns the first
    existing file.
    """
    normalized = str(raw_path).replace("\\", "/").strip()
    path = Path(normalized)

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                repo_root / path,
                ml_root / path,
                Path.cwd() / path,
            ]
        )

        # Allow manifests that already include an `ml/` prefix to resolve from
        # the repository root without double-prefixing the path.
        if path.parts and path.parts[0] == "ml":
            candidates.append(repo_root / path)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    if candidates:
        return candidates[0].resolve(strict=False)
    return path.resolve(strict=False)
