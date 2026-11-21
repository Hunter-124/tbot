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


SCENERY_ABI = 3      # TBOT_SCENERY_ABI in _measure.c
SCENERY_MAX_OLD = 4  # SCENERY_MAX_OLD (= clutter.COMPARE_FRAMES)
SCENERY_LISTED = 24  # SCENERY_LISTED (= clutter.STATIC_LISTED)
RAYS_GIVEN, RAYS_FISHEYE, RAYS_PINHOLE, RAYS_PINHOLE_IDEAL = -1, 0, 1, 2
ARR_TYPES = {np.dtype(np.float64): 0, np.dtype(np.int64): 1, np.dtype(np.int32): 2}


class SceneryParams(ctypes.Structure):
    """tbot_scenery_params in _measure.c, field for field."""

    _fields_ = [
        ("n", ctypes.c_int32), ("model", ctypes.c_int32),
        ("uv", _ptr), ("rays", _ptr),
        ("fx", ctypes.c_double), ("fy", ctypes.c_double), ("cx", ctypes.c_double), ("cy", ctypes.c_double),
        ("skew", ctypes.c_double), ("k", ctypes.c_double * 8),
        ("newton_iters", ctypes.c_int32), ("undistort_iters", ctypes.c_int32), ("newton_tol", ctypes.c_double),
        ("R_now", _ptr),
        ("n_old", ctypes.c_int32), ("old_m", ctypes.c_int32 * SCENERY_MAX_OLD),
        ("old_R", _ptr * SCENERY_MAX_OLD), ("old_dirs", _ptr * SCENERY_MAX_OLD),
        ("tol0", ctypes.c_double), ("turn_tol", ctypes.c_double), ("lever", ctypes.c_double),
        ("lam_max", ctypes.c_double), ("view", ctypes.c_double * 4),
        ("areas", _ptr), ("widths", _ptr), ("heights", _ptr),
        ("areas_t", ctypes.c_int32), ("widths_t", ctypes.c_int32), ("heights_t", ctypes.c_int32),
        ("size_tol", ctypes.c_double),
        ("protect", _ptr), ("n_protect", ctypes.c_int32), ("protect_cos", ctypes.c_double),
        ("focus", _ptr), ("focus_cos", _ptr), ("n_focus", ctypes.c_int32), ("others_max", ctypes.c_int32),
        ("drift_on", ctypes.c_int32), ("ifov", ctypes.c_double), ("t_now", ctypes.c_int64),
        ("old_t", ctypes.c_int64 * SCENERY_MAX_OLD),
        ("drift_bin", ctypes.c_double), ("drift_max", ctypes.c_double), ("drift_peak_ratio", ctypes.c_double),
        ("drift_gain", ctypes.c_double), ("drift_box", ctypes.c_int32), ("drift_min_votes", ctypes.c_int32),
        ("drift_x", ctypes.c_double), ("drift_y", ctypes.c_double), ("drift_updates", ctypes.c_int32),
        ("drift_votes", ctypes.c_int32), ("drift_grid", _ptr), ("drift_ii", _ptr), ("drift_side", ctypes.c_int32),
        ("dirs", _ptr), ("keep", _ptr), ("flags", _ptr), ("static_px", _ptr),
        ("n_keep", ctypes.c_int32), ("n_static", ctypes.c_int32), ("n_new", ctypes.c_int32),
        ("n_focus_kept", ctypes.c_int32), ("n_listed", ctypes.c_int32),
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
    _load_scenery(lib)
    _load_track(lib)
    _load_front(lib)
    _load_motion(lib)
    return lib


scenery_status = "not loaded"   # the scenery filter's kernel (tbot bench prints it)
_scenery = None


def _load_scenery(lib: ctypes.CDLL) -> None:
    """The scenery filter's entry point in the same library; a library built before it has none."""
    global _scenery, scenery_status
    try:
        lib.tbot_scenery_abi.restype = ctypes.c_int
        lib.tbot_scenery_params_size.restype = ctypes.c_int
        abi, size = int(lib.tbot_scenery_abi()), int(lib.tbot_scenery_params_size())
    except AttributeError:
        scenery_status = "not in this build (rebuild with scripts/build_native.sh)"
        return
    if abi != SCENERY_ABI or size != ctypes.sizeof(SceneryParams):
        scenery_status = f"stale (ABI {abi}, struct {size} bytes): rebuild with scripts/build_native.sh"
        log.warning("native scenery filter: %s; using the Python path", scenery_status)
        return
    fn = lib.tbot_scenery
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p]
    _scenery = fn
    scenery_status = "loaded"


TRACK_ABI = 1   # TBOT_TRACK_ABI in _measure.c
track_status = "not loaded"
_track = None


def _load_track(lib: ctypes.CDLL) -> None:
    """The tracker's filter step (tbot_track_predict / tbot_track_update); a library built before it has none."""
    global _track, track_status
    try:
        lib.tbot_track_abi.restype = ctypes.c_int
        abi = int(lib.tbot_track_abi())
        predict, update = lib.tbot_track_predict, lib.tbot_track_update
    except AttributeError:
        track_status = "not in this build (rebuild with scripts/build_native.sh)"
        return
    if abi != TRACK_ABI:
        track_status = f"stale (ABI {abi}): rebuild with scripts/build_native.sh"
        log.warning("native tracker step: %s; using the Python path", track_status)
        return
    d = ctypes.c_double
    predict.restype = None
    predict.argtypes = [_ptr, _ptr, d, d, _ptr, _ptr]
    update.restype = ctypes.c_int
    update.argtypes = [_ptr, _ptr, _ptr, _ptr, _ptr, ctypes.c_int32, d, d, _ptr, _ptr]
    _track = (predict, update)
    track_status = "loaded"


FRONT_ABI = 1          # TBOT_FRONT_ABI in _measure.c
FRONT_CAP_RUNS = 40000  # runs of set pixels a frame (a tree line: a few thousand); more -> OpenCV labels the frame
FRONT_CAP_COMPS = 8192
front_status = "not loaded"
_front = None


class FrontState(ctypes.Structure):
    """tbot_front in _measure.c, field for field."""

    _fields_ = [
        ("h", ctypes.c_int32), ("w", ctypes.c_int32),
        ("cap_runs", ctypes.c_int32), ("cap_comps", ctypes.c_int32),
        ("runs", _ptr), ("row_start", _ptr), ("labels", _ptr), ("stats", _ptr), ("cents", _ptr),
        ("n_runs", ctypes.c_int32), ("roi", ctypes.c_int32 * 4),
    ]


class Front:
    """The buffers of tbot_front_label for one frame size: label image, component table, runs."""

    def __init__(self, h: int, w: int) -> None:
        self.shape = (int(h), int(w))
        self.runs = np.zeros((FRONT_CAP_RUNS, 4), np.int32)
        self.row_start = np.zeros(int(h) + 1, np.int32)
        self.labels = np.zeros(self.shape, np.int32)
        self.stats = np.zeros((FRONT_CAP_COMPS, 5), np.int32)
        self.cents = np.zeros((FRONT_CAP_COMPS, 2), np.float64)
        st = self.state = FrontState()
        st.h, st.w = self.shape
        st.cap_runs, st.cap_comps = FRONT_CAP_RUNS, FRONT_CAP_COMPS
        st.runs, st.row_start = self.runs.ctypes.data, self.row_start.ctypes.data
        st.labels, st.stats, st.cents = self.labels.ctypes.data, self.stats.ctypes.data, self.cents.ctypes.data
        self.ref = ctypes.byref(st)


def _load_front(lib: ctypes.CDLL) -> None:
    """Labelling, the stands-alone test and the noise estimate; a library built before them has none."""
    global _front, front_status
    try:
        lib.tbot_front_abi.restype = ctypes.c_int
        lib.tbot_front_size.restype = ctypes.c_int
        abi, size = int(lib.tbot_front_abi()), int(lib.tbot_front_size())
        label, alone, noise = lib.tbot_front_label, lib.tbot_front_alone, lib.tbot_noise_sigma
        box, cells = lib.tbot_front_box, lib.tbot_front_cells
    except AttributeError:
        front_status = "not in this build (rebuild with scripts/build_native.sh)"
        return
    if abi != FRONT_ABI or size != ctypes.sizeof(FrontState):
        front_status = f"stale (ABI {abi}, struct {size} bytes): rebuild with scripts/build_native.sh"
        log.warning("native detector front end: %s; using OpenCV", front_status)
        return
    i32, d = ctypes.c_int32, ctypes.c_double
    label.restype = ctypes.c_int
    label.argtypes = [ctypes.c_void_p, _ptr]
    alone.restype = None
    alone.argtypes = [ctypes.c_void_p, _ptr, i32, i32, i32, d, _ptr]
    noise.restype = d
    noise.argtypes = [_ptr, _ptr, _ptr, _ptr, i32, i32, i32, i32, i32, i32, i32, d, d, d]
    box.restype = ctypes.c_int
    box.argtypes = [_ptr, i32, i32, i32, i32, _ptr]
    cells.restype = ctypes.c_int
    cells.argtypes = [_ptr, i32, i32, i32, i32, i32, _ptr]
    _front = (label, alone, noise, box, cells)
    front_status = "loaded"


MOTION_ABI = 5        # TBOT_MOTION_ABI in _measure.c
MOTION_DET_COLS = 15  # MO_DET_COLS: u, v, area, diameter, contrast, snr, score, bbox x, y, w, h, stab u, v, mass, minor
motion_status = "not loaded"
_motion = None


class MotionReg(ctypes.Structure):
    """tbot_motion_reg in _measure.c, field for field."""

    _fields_ = [
        ("hc", ctypes.c_int32), ("wc", ctypes.c_int32), ("hf", ctypes.c_int32), ("wf", ctypes.c_int32),
        ("block", ctypes.c_int32), ("r_c", ctypes.c_int32),
        ("gr", ctypes.c_int32), ("gc", ctypes.c_int32), ("field_r", ctypes.c_int32), ("use_field", ctypes.c_int32),
        ("np_chunk", ctypes.c_int32), ("need_x", ctypes.c_int32), ("need_y", ctypes.c_int32), ("pad_", ctypes.c_int32),
        ("cur_c", _ptr), ("cur_f", _ptr), ("tex_f", _ptr), ("ref_c", _ptr), ("ref_f", _ptr),
        ("srcv_c", _ptr), ("srcv_f", _ptr), ("curv_c", _ptr), ("curv_f", _ptr),
        ("wrc", _ptr), ("vrc", _ptr), ("wrf", _ptr), ("vrf", _ptr),
        ("dx", _ptr), ("dy", _ptr), ("rx", _ptr), ("ry", _ptr),
        ("ys", _ptr), ("xs", _ptr), ("field", _ptr), ("field_ok", _ptr),
        ("lc", ctypes.c_double * 9), ("lf", ctypes.c_double * 9), ("lm", ctypes.c_double * 9),
        ("min_overlap_frac", ctypes.c_double), ("black_dn", ctypes.c_double), ("gain_min", ctypes.c_double),
        ("gain_max", ctypes.c_double), ("reg_min_response", ctypes.c_double), ("fit_std_min", ctypes.c_double),
        ("cell_texture_frac", ctypes.c_double),
        ("early", ctypes.c_int32), ("edge", ctypes.c_int32), ("curv", ctypes.c_int32),
        ("n_ovl", ctypes.c_int32), ("n_tex", ctypes.c_int32), ("g0", ctypes.c_int32 * 2),
        ("depth", ctypes.c_double), ("delta", ctypes.c_double * 2),
        ("gq", ctypes.c_int32 * 2), ("a", ctypes.c_double), ("b", ctypes.c_double),
    ]


class MotionKParams(ctypes.Structure):
    """tbot_motion_kparams in _measure.c, field for field."""

    _fields_ = [
        ("h", ctypes.c_int32), ("w", ctypes.c_int32), ("block", ctypes.c_int32), ("env", ctypes.c_int32),
        ("cell", ctypes.c_int32), ("row_step", ctypes.c_int32), ("cfar_box", ctypes.c_int32), ("cfar_guard", ctypes.c_int32),
        ("ga", ctypes.c_int64), ("oa", ctypes.c_int64), ("gb", ctypes.c_int64), ("ob", ctypes.c_int64),
        ("cur", _ptr), ("ref_a", _ptr), ("ref_b", _ptr), ("src_valid", _ptr), ("mask", _ptr),
        ("dxa", _ptr), ("dya", _ptr), ("dxb", _ptr), ("dyb", _ptr),
        ("wa", _ptr), ("wb", _ptr), ("va", _ptr), ("vb", _ptr), ("D", _ptr), ("binary", _ptr), ("thr", _ptr),
        ("mD", _ptr), ("SD", _ptr), ("ND", _ptr), ("SA", _ptr), ("NA", _ptr), ("scratch", _ptr),
        ("t_min_dn", ctypes.c_double), ("noise_k", ctypes.c_double), ("cfar_gain", ctypes.c_double),
        ("texture_gain", ctypes.c_double), ("abs_to_sigma", ctypes.c_double), ("noise_quantile", ctypes.c_double),
        ("noise_min_fill", ctypes.c_double), ("sigma_n", ctypes.c_double),
    ]


class MotionMeas(ctypes.Structure):
    """tbot_motion_meas in _measure.c, field for field."""

    _fields_ = [
        ("h", ctypes.c_int32), ("w", ctypes.c_int32), ("cell", ctypes.c_int32), ("scale", ctypes.c_int32),
        ("min_area", ctypes.c_int32), ("max_area", ctypes.c_int32), ("max_cand", ctypes.c_int32),
        ("cap_runs", ctypes.c_int32),
        ("binary", _ptr), ("D", _ptr), ("cur", _ptr), ("wa", _ptr), ("mD", _ptr),
        ("joined", _ptr), ("runs", _ptr), ("out", _ptr),
        ("S", ctypes.c_double * 9), ("pixel_box_var", ctypes.c_double), ("score_mass", ctypes.c_double),
        ("n_labels", ctypes.c_int32), ("n_cand", ctypes.c_int32), ("n_out", ctypes.c_int32),
        ("roi", ctypes.c_int32 * 4),
    ]


def _load_motion(lib: ctypes.CDLL) -> None:
    """The motion detector's kernels (tbot_motion_*); a library built before them has none."""
    global _motion, motion_status
    try:
        lib.tbot_motion_abi.restype = ctypes.c_int
        abi = int(lib.tbot_motion_abi())
        sizes = []
        for nm in ("tbot_motion_reg_size", "tbot_motion_kparams_size", "tbot_motion_meas_size"):
            f = getattr(lib, nm)
            f.restype = ctypes.c_int
            sizes.append(int(f()))
    except AttributeError:
        motion_status = "not in this build (rebuild with scripts/build_native.sh)"
        return
    want = [ctypes.sizeof(MotionReg), ctypes.sizeof(MotionKParams), ctypes.sizeof(MotionMeas)]
    if abi != MOTION_ABI or sizes != want:
        motion_status = f"stale (ABI {abi}, structs {sizes} bytes, want {want}): rebuild with scripts/build_native.sh"
        log.warning("native motion kernel: %s; using the Python path", motion_status)
        return
    i32, i64, d, p = ctypes.c_int32, ctypes.c_int64, ctypes.c_double, _ptr
    sig = {
        "box_down2": (None, [p, i32, i32, p]),
        "texture": (None, [p, i32, i32, i32, p, p, p]),
        "prep": (None, [p, i32, i32, i32, i32, p, p, p, p, p, i32, p, p]),
        "block_offsets": (None, [p, i32, i32, i32, i32, p, p, p, p, p, p, p, i32]),
        "sum": (d, [p, i64, i32]),
        "warp": (None, [p, p, i32, i32, p, p, i32, p, p]),
        "gain": (None, [p, i64, i64, i64, p]),
        "envelope": (None, [p, p, p, p, p, p, i32, i32, i32, p, p]),
        "cell_sums": (None, [p, i32, i32, i32, i32, p]),
        "absdiff_sums": (None, [p, p, p, i32, i32, i32, i32, p, p]),
        "noise": (d, [p, p, i32, i32, i32, d, d, d]),
        "threshold": (ctypes.c_int, [p, p, p, p, i32, i32, d, d, d, d, d, d, i32, i32, p, p]),
        "compare": (None, [p, i32, i32, p, i32, i32, p]),
        "sad": (None, [p, p, p, p, i32, i32, i32, i32, i32, i32, i32, i32, i32, p, p]),
        "sad_cells": (None, [p, p, p, p, i32, i32, i32, p, i32, p, i32, p, p]),
        "photometric": (None, [p, p, p, i32, i32, d, d, d, d, p, p]),
        "reg1": (ctypes.c_int, [p]),
        "reg2": (ctypes.c_int, [p, d]),
        "field_extra": (None, [p, i32, i32, i32, i32, p, p, p, p, p, p, d, p, p]),
        "offsets": (None, [p, i32, i32, i32, p, i32, i32, p, p, p, p, p, p, d, p, p, p, p, p, p]),
        "kernel": (ctypes.c_int, [p]),
        "measure": (ctypes.c_int, [p]),
    }
    fns = {}
    try:
        for nm, (res, args) in sig.items():
            f = getattr(lib, "tbot_motion_" + nm)
            f.restype = res
            f.argtypes = args
            fns[nm] = f
    except AttributeError as exc:
        motion_status = f"incomplete build ({exc}): rebuild with scripts/build_native.sh"
        return
    chunk = _numpy_reduce_chunk(fns["sum"])
    if chunk is None:
        motion_status = "off (this NumPy's float64 summation order is not one the kernel knows)"
        log.warning("native motion kernel: %s; using the Python path", motion_status)
        return
    fns["np_chunk"] = chunk
    _motion = type("MotionKernel", (), fns)
    motion_status = "loaded"


def _numpy_reduce_chunk(c_sum) -> int | None:
    """How this NumPy sums a contiguous float64 array (ndarray.sum / mean): pairwise over the whole array
    (NumPy >= 2.3: 0) or over runs of 8192 elements (its buffer size before; 8192). None when neither model
    reproduces it bit for bit on the probe arrays."""
    rng = np.random.default_rng(12345)
    probes = [rng.standard_normal(n) * 10.0 ** rng.integers(-4, 5, n) for n in (5, 300, 9000, 20011, 40000)]
    for chunk in (0, 8192):
        if all(float(c_sum(a.ctypes.data, a.size, chunk)) == float(a.sum()) for a in probes):
            return chunk
    return None


def motion_kernel():
    """The motion detector's C entry points (attributes prep, reg1, reg2, offsets, kernel, measure and the
    primitives: ctypes functions) from the loaded library, or None (not built, stale, switched off)."""
    if kernel() is None:
        return None
    return _motion


def front_kernel():
    """(tbot_front_label, tbot_front_alone, tbot_noise_sigma, tbot_front_box, tbot_front_cells) from the loaded
    library, or None."""
    if kernel() is None:
        return None
    return _front


def track_kernel():
    """(tbot_track_predict, tbot_track_update) from the loaded library, or None."""
    if kernel() is None:
        return None
    return _track


def scenery_kernel():
    """tbot_scenery from the loaded library, or None (not built, stale, switched off)."""
    if kernel() is None:
        return None
    return _scenery


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
    return f"{status}; motion kernel {motion_status}"


def reload() -> ctypes.CDLL | None:
    """Forget the cached result and look again (tests; after a rebuild or an environment change)."""
    global _tried, _scenery, scenery_status, _track, track_status, _front, front_status, _motion, motion_status
    _tried = False
    _motion = None
    motion_status = "not loaded"
    _front = None
    front_status = "not loaded"
    _scenery = None
    scenery_status = "not loaded"
    _track = None
    track_status = "not loaded"
    return kernel()


def ptr(a: np.ndarray | None, dtype) -> int | None:
    """Address of a C-contiguous array of exactly `dtype`; None (a null pointer) for None. Raises TypeError
    for anything else (the caller then takes the Python path rather than copy or reinterpret)."""
    if a is None:
        return None
    if not isinstance(a, np.ndarray) or a.dtype != dtype or not a.flags.c_contiguous:
        raise TypeError("not a C-contiguous array of the expected type")
    return a.ctypes.data
