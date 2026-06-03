"""ctypes wrapper for the firmware's EstimateDialCenterFromRimVotes.

Loads the compiled C shared library (rim_estimator.so) which contains
the exact C functions extracted from app_baseline_runtime.c.  This gives
us ground-truth-quality centre labels on YUV422 board capture frames
without having to port the firmware to Python (where subtle float
differences accumulate and produce noisy pseudo-labels).
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Resolve the shared library path relative to this file.
_LIB_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent.parent / "c" / "rim_estimator.so"
)

if not _LIB_PATH.exists():
    raise FileNotFoundError(
        f"rim_estimator.so not found at {_LIB_PATH}.  "
        "Compile with: cd ml/c && gcc -shared -fPIC -O2 -lm -o rim_estimator.so rim_estimator.c"
    )

_lib: ctypes.CDLL = ctypes.CDLL(str(_LIB_PATH))

# int rim_estimator_find_center(
#     const uint8_t *frame, size_t frame_size, size_t row_stride,
#     size_t fw, size_t fh, float dial_radius,
#     float *cx_out, float *cy_out, float *q_out) -> bool
_lib.rim_estimator_find_center.restype = ctypes.c_bool
_lib.rim_estimator_find_center.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # frame bytes
    ctypes.c_size_t,                 # frame_size
    ctypes.c_size_t,                 # row_stride
    ctypes.c_size_t,                 # fw
    ctypes.c_size_t,                 # fh
    ctypes.c_float,                  # dial_radius
    ctypes.POINTER(ctypes.c_float),  # cx_out
    ctypes.POINTER(ctypes.c_float),  # cy_out
    ctypes.POINTER(ctypes.c_float),  # q_out
]


def find_rim_center(
    yuv422: NDArray[np.uint8],
    dial_radius_px: float = 68.9,
) -> tuple[float, float, bool]:
    """Run the firmware's rim-centre estimator on a YUV422 byte buffer.

    Parameters
    ----------
    yuv422 : np.uint8 array of shape (height, width, 2) — YUV422 packed frame.
             The array must be contiguous in memory (C order).
    dial_radius_px : float — dial radius in pixels (default 68.9 for 224×224).

    Returns
    -------
    (cx, cy, found) — centre in pixel coords and whether detection succeeded.
    """
    fh, fw, bpp = yuv422.shape  # bpp == 2 for YUV422
    assert bpp == 2, f"Expected YUV422 (3D array with 2 channels), got shape {yuv422.shape}"
    frame_size = yuv422.nbytes
    row_stride = fw * 2  # bytes per row

    # Ensure contiguous C-order memory for the pointer.
    buf = np.ascontiguousarray(yuv422.ravel())
    frame_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    cx_out = ctypes.c_float()
    cy_out = ctypes.c_float()
    q_out = ctypes.c_float()

    found = _lib.rim_estimator_find_center(
        frame_ptr,
        ctypes.c_size_t(frame_size),
        ctypes.c_size_t(row_stride),
        ctypes.c_size_t(fw),
        ctypes.c_size_t(fh),
        ctypes.c_float(dial_radius_px),
        ctypes.byref(cx_out),
        ctypes.byref(cy_out),
        ctypes.byref(q_out),
    )

    return float(cx_out.value), float(cy_out.value), bool(found)
