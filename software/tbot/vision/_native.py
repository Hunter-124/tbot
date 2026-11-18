"""Loader for the native measurement kernel (tbot/vision/_measure.c), called through ctypes.

The kernel is optional: `kernel()` returns None when the shared library has not been built
(scripts/build_native.sh), does not load, or is from another version of the C file, and the detector then
runs its Python loop, which is the reference and gives the same detections. TBOT_NATIVE=0 in the
environment (or detector.native = false in the config) forces the Python path.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ABI = 1          # TBOT_MEASURE_ABI in _measure.c
OUT_COLS = 12    # reject code, u, v, area, diameter, contrast, snr, score, bbox x, y, w, h
ENV = "TBOT_NATIVE"

# Array arguments go over as plain addresses (c_void_p): ndarray.ctypes.data_as() costs several microseconds
# a pointer, as much as the kernel takes to measure a candidate.
_ptr = ctypes.c_void_p


class MeasureParams(ctypes.Structure):
    """tbot_measure_params in _measure.c, field for field."""

    _fields_ = [
        ("h", ctypes.c_int32), ("w", ctypes.c_int32),
        ("gray", _ptr), ("bg", _ptr), ("gsm", _ptr), ("contrast16", _ptr),   # uint8 x3, int16
        ("labels", _ptr), ("stats", _ptr), ("mask", _ptr),                   # int32 x2, uint8
        ("sigma", ctypes.c_double), ("matched_sigma", ctypes.c_double), ("grow_low", ctypes.c_double),
        ("has_grow", ctypes.c_int32), ("grown_done", ctypes.c_int32),
        ("k_sigma", ctypes.c_double), ("min_contrast", ctypes.c_double),
        ("max_peak_fraction", ctypes.c_double), ("max_aspect", ctypes.c_double),
        ("min_area_px", ctypes.c_int32), ("max_area_px", ctypes.c_int32), ("moat_half", ctypes.c_int32),
        ("adaptive_min_half", ctypes.c_int32), ("grow_radius", ctypes.c_int32),
        ("local_ring_px", ctypes.c_int32), ("adaptive_iters", ctypes.c_int32),
        ("window_sigmas", ctypes.c_double), ("alpha_half", ctypes.c_double),
        ("local_k_sigma", ctypes.c_double), ("meas_seed_sigmas", ctypes.c_double),
        ("moment_floor_sigmas", ctypes.c_double), ("pixel_box_var", ctypes.c_double),
        ("adaptive_alpha", ctypes.c_double), ("min_weight_var", ctypes.c_double),
        ("adaptive_tol", ctypes.c_double), ("adaptive_damping", ctypes.c_double),
        ("diameter_divisor", ctypes.c_double), ("moment_m00_min", ctypes.c_double),
    ]


def library_path() -> Path:
    """Where scripts/build_native.sh puts the library: next to the C file."""
    ext = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    return Path(__file__).with_name("_measure" + ext)


def disabled_by_env() -> bool:
    return os.environ.get(ENV, "").strip().lower() in ("0", "false", "no", "off", "python")


_lib: ctypes.CDLL | None = None
_tried = False
status = "not loaded"   # why the kernel is or is not in use (tbot bench prints it)


def _load() -> ctypes.CDLL | None:
    global status
    if disabled_by_env():
        status = f"off ({ENV}={os.environ.get(ENV)})"
        return None
    path = library_path()
    if not path.is_file():
        status = f"not built ({path.name} missing: scripts/build_native.sh)"
        return None
    try:
        lib = ctypes.CDLL(str(path))
        lib.tbot_measure_abi.restype = ctypes.c_int
        lib.tbot_measure_params_size.restype = ctypes.c_int
        abi, size = int(lib.tbot_measure_abi()), int(lib.tbot_measure_params_size())
    except (OSError, AttributeError) as exc:
        status = f"failed to load {path.name}: {exc}"
        log.warning("native measurement kernel: %s; using the Python path", status)
        return None
    if abi != ABI or size != ctypes.sizeof(MeasureParams):
        status = f"stale {path.name} (ABI {abi}, struct {size} bytes): rebuild with scripts/build_native.sh"
        log.warning("native measurement kernel: %s; using the Python path", status)
        return None
    lib.tbot_measure.restype = ctypes.c_int
    lib.tbot_measure.argtypes = [ctypes.POINTER(MeasureParams), _ptr, ctypes.c_int32, _ptr]
    status = f"loaded ({path.name})"
    return lib


def kernel() -> ctypes.CDLL | None:
    """The loaded library, or None. Tried once per process."""
    global _lib, _tried
    if not _tried:
        _lib = _load()
        _tried = True
    return _lib


def describe(enabled: bool = True) -> str:
    """One line for reports (tbot bench): the kernel is in use, or why it is not."""
    if not enabled:
        return "off (detector.native = false)"
    kernel()
    return status


def reload() -> ctypes.CDLL | None:
    """Forget the cached result and look again (tests; after a rebuild or an environment change)."""
    global _tried
    _tried = False
    return kernel()


def ptr(a: np.ndarray | None, dtype) -> int | None:
    """Address of a C-contiguous array of exactly `dtype`; None (a null pointer) for None. Raises TypeError
    for anything else (the caller then takes the Python path rather than copy or reinterpret)."""
    if a is None:
        return None
    if not isinstance(a, np.ndarray) or a.dtype != dtype or not a.flags.c_contiguous:
        raise TypeError("not a C-contiguous array of the expected type")
    return a.ctypes.data
