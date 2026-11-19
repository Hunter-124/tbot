"""The native measurement kernel (tbot/vision/_measure.c) gives the detections of the Python loop it replaces.

The Python loop in ClayDetector._measure is the reference. The kernel repeats it statement by statement, so
the same candidates are accepted and rejected and the integer results (area, bounding box, peak contrast)
are equal; the moments differ by rounding only (expf against NumPy's float32 exp, summation order).

Everything here is skipped when the shared library is not built (scripts/build_native.sh; no compiler, or
TBOT_NATIVE=0), which is the state the rest of the suite must also pass in.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tbot.camera.synthetic import SyntheticSource
from tbot.config import Calibration, load_config
from tbot.sim.scenario import SCENARIOS
from tbot.types import Frame
from tbot.vision import _native
from tbot.vision.detector import ClayDetector
from test_detector import add_disc, add_ellipse, frame, sky

pytestmark = pytest.mark.skipif(_native.kernel() is None, reason=f"native kernel: {_native.status}")

W, H = 640, 400
PRESSES = Path(__file__).resolve().parents[1] / "field_data" / "2026-11-17-range" / "presses"
POS_TOL_PX = 1e-3
DIAMETER_RTOL = 1e-3


def _pair(sets=(), mask=None):
    """(native detector, Python detector) on the same configuration."""
    out = []
    for native in ("true", "false"):
        cfg = load_config(sets=[*sets, f"detector.native={native}"])
        out.append(ClayDetector(cfg, mask))
    return out


def _same(a, b, where):
    assert len(a) == len(b), f"{where}: {len(a)} native detections, {len(b)} Python"
    for x, y in zip(a, b):
        assert x.t_ns == y.t_ns and x.bbox == y.bbox and x.area_px == y.area_px, where
        assert x.contrast == y.contrast, where
        # A blob cut by the frame edge can send the moment fit far outside the frame (both kernels alike: found
        # 2026-11-19 on trap_dead_straight, a centroid at u = -171). Such a centroid is an extrapolation, and the
        # two kernels then agree to the same RELATIVE accuracy, not to a thousandth of a pixel.
        off = max(-y.u, y.u - W, -y.v, y.v - H, 0.0)
        tol = POS_TOL_PX * (1.0 + off / 50.0)
        assert abs(x.u - y.u) < tol and abs(x.v - y.v) < tol, where
        assert x.diameter_px == pytest.approx(y.diameter_px, rel=DIAMETER_RTOL), where
        assert x.snr == pytest.approx(y.snr, rel=1e-9), where
        assert x.score == pytest.approx(y.score, rel=1e-3), where
        for v in (x.u, x.v, x.area_px, x.diameter_px, x.contrast, x.snr, x.score):
            assert type(v) is float, where


def _run(frames, sets=(), mask=None, candidate_filter=None):
    """Both detectors over the same frames; returns (frames, detections compared, candidates seen)."""
    nat, ref = _pair(sets, mask)
    nat.candidate_filter = ref.candidate_filter = candidate_filter
    n_frames = n_dets = n_cand = 0
    for k, fr in enumerate(frames):
        a, b = nat.detect(fr), ref.detect(fr)
        assert nat.stats == ref.stats, f"frame {k}"
        _same(a, b, f"frame {k}")
        n_frames += 1
        n_dets += len(b)
        n_cand += int(ref.stats.get("candidates", 0))
    return n_frames, n_dets, n_cand


# --------------------------------------------------------------------------------- built scenes
def _scene(rng, n=14, noise=2.0, clouds=6.0, specks=12):
    """Sky with clays of both polarities, edge-on clays, clays at the image border, and hot pixels."""
    img = sky(rng, noise=noise, clouds=clouds)
    for _ in range(n):
        cx, cy = rng.uniform(-2, W + 2), rng.uniform(-2, H + 2)
        d = rng.uniform(1.5, 26.0)
        c = rng.choice([-1.0, 1.0]) * rng.uniform(12.0, 70.0)
        if rng.random() < 0.4:
            add_ellipse(img, cx, cy, d, max(1.0, d * rng.uniform(0.2, 0.6)), c, angle_deg=rng.uniform(0, 180))
        else:
            add_disc(img, cx, cy, d, c)
    for _ in range(specks):
        img[rng.integers(0, H), rng.integers(0, W)] += rng.choice([-1.0, 1.0]) * rng.uniform(30.0, 90.0)
    return img


@pytest.mark.parametrize("sets", [
    (),
    ("detector.polarity=dark",),
    ("detector.polarity=bright",),
    ("detector.downscale=1",),
    ("detector.downscale=2",),
    ("detector.downscale=1", "detector.bg_mode=morph"),
    ("detector.grow_k_sigma=0.0",),
    ("detector.matched_sigma_px=0.0",),                      # plain mode: the int16 residual
    ("detector.matched_sigma_px=0.0", "detector.downscale=1"),
    ("detector.max_peak_fraction=1.0", "detector.min_area_px=4"),
    ("detector.min_area_px=3", "detector.clutter_ring_max=0.0", "detector.downscale=1"),
    ("detector.min_contrast=4.0", "detector.k_sigma=4.0", "detector.max_detections=64"),
], ids=lambda s: ",".join(x.split(".", 1)[1] for x in s) or "default")
def test_built_scenes_match(sets):
    rng = np.random.default_rng(11)
    frames = [frame(_scene(rng, noise=float(rng.uniform(0.5, 4.0))), seq=k, t_ns=1_000_000_000 + k * 10_000_000)
              for k in range(12)]
    n_frames, n_dets, n_cand = _run(frames, sets)
    assert n_dets >= 20 and n_cand > n_dets   # both accepted and rejected candidates were compared


def test_masked_and_cluttered_frames_match():
    rng = np.random.default_rng(5)
    mask = np.full((H, W), 255, np.uint8)
    mask[:, :90] = 0
    mask[300:, 200:420] = 0
    frames = []
    for k in range(8):
        img = _scene(rng, n=20, specks=60)
        img[330:, :] += rng.normal(0.0, 18.0, (H - 330, W)).astype(np.float32)   # a busy band: many candidates
        frames.append(frame(img, seq=k, t_ns=1_000_000_000 + k * 10_000_000))
    for sets in ((), ("detector.downscale=1", "detector.local_noise_cell=0", "detector.clutter_ring_max=0.0")):
        n_frames, n_dets, n_cand = _run(frames, sets, mask=mask)
        assert n_dets >= 10
        # a candidate filter (run.py's scenery filter): the survivors arrive in its order and are limited
        _run(frames, sets, mask=mask, candidate_filter=lambda uv, areas, ws, hs: np.argsort(-areas, kind="stable"))


def test_colour_frames_match():
    """Colour sources take the plain path, with hue-selected blobs that may have no luma contrast."""
    rng = np.random.default_rng(2)
    frames = []
    for k in range(6):
        gray = np.clip(_scene(rng), 0, 255).astype(np.uint8)
        bgr = np.repeat(gray[:, :, None], 3, axis=2)
        for _ in range(5):   # orange patches at the brightness of the sky around them
            x, y, r = int(rng.integers(20, W - 20)), int(rng.integers(20, H - 20)), int(rng.integers(2, 7))
            v = int(gray[y, x])
            bgr[y - r:y + r, x - r:x + r] = (max(0, v - 90), v, min(255, v + 60))
        frames.append(Frame(image=bgr, seq=k, t_exposure_ns=1_000_000_000 + k * 10_000_000,
                            t_capture_ns=1_005_000_000 + k * 10_000_000, exposure_us=500.0, gain=1.0))
    for sets in ((), ("detector.color_combine=and",), ("detector.mode=mono",)):
        n_frames, n_dets, _ = _run(frames, sets)
        assert n_dets >= 6


def test_tiny_frames_match():
    """Windows clipped on every side: frames barely larger than the measurement window."""
    rng = np.random.default_rng(9)
    for h, w in ((24, 40), (48, 48), (17, 90)):
        frames = []
        for k in range(6):
            img = sky(rng, h=h, w=w, noise=1.5)
            add_disc(img, rng.uniform(0, w), rng.uniform(0, h), rng.uniform(2.0, 9.0), -50.0)
            add_disc(img, rng.uniform(0, w), rng.uniform(0, h), rng.uniform(2.0, 6.0), 45.0)
            frames.append(frame(img, seq=k))
        _run(frames, ("detector.downscale=1", "detector.local_noise_cell=0"))


# --------------------------------------------------------------------------------- simulator scenarios
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_synthetic_scenarios_match(scenario):
    cfg = load_config()
    src = SyntheticSource(cfg, Calibration(dir=Path(".")), scenario=scenario, seed=3)
    src.start()
    try:
        def frames():
            while True:
                fr = src.read(timeout_s=2.0)
                if fr is None:
                    return
                # the source may reuse its frame buffer: each detector must see this frame's pixels
                yield Frame(image=fr.image.copy(), seq=fr.seq, t_exposure_ns=fr.t_exposure_ns,
                            t_capture_ns=fr.t_capture_ns, exposure_us=fr.exposure_us, gain=fr.gain)
        n_frames, n_dets, _ = _run(frames())
    finally:
        src.stop()
    assert n_frames > 50
    if scenario != "no_clay":
        assert n_dets > 20


# --------------------------------------------------------------------------------- recorded frames
def _press_dirs():
    if not PRESSES.is_dir():
        return []
    return sorted(d for d in PRESSES.iterdir() if (d / "frames.bin").is_file() and (d / "meta.json").is_file())


def _recorded_frames(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    h, w, ch = int(meta["height"]), int(meta["width"]), int(meta.get("channels", 1))
    size = (d / "frames.bin").stat().st_size
    if size < h * w * ch:   # a git LFS pointer, not the frames
        return None
    shape = (size // (h * w * ch), h, w) + ((ch,) if ch > 1 else ())
    data = np.memmap(d / "frames.bin", dtype=np.uint8, mode="r", shape=shape)
    with open(d / "frames.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    n = min(len(rows), shape[0])
    return [Frame(image=np.array(data[k]), seq=int(rows[k]["seq"]), t_exposure_ns=int(rows[k]["t_exposure_ns"]),
                  t_capture_ns=int(rows[k]["t_capture_ns"]), exposure_us=float(rows[k]["exposure_us"]),
                  gain=float(rows[k]["gain"])) for k in range(n)]


@pytest.mark.parametrize("press", _press_dirs() or [None], ids=lambda d: d.name if d else "absent")
def test_recorded_range_frames_match(press):
    if press is None:
        pytest.skip("no recorded presses (field_data is git LFS)")
    frames = _recorded_frames(press)
    if frames is None:
        pytest.skip("frames.bin is a git LFS pointer")
    n_frames, _, n_cand = _run(frames)
    assert n_frames > 0
    # ... and again at full resolution without the clutter screens, where far more candidates are measured
    _run(frames[::4], ("detector.downscale=1", "detector.clutter_ring_max=0.0", "detector.local_noise_cell=0",
                       "detector.min_size_frac=0.0", "detector.max_detections=64"))


# --------------------------------------------------------------------------------- the switch
def test_python_path_can_be_forced(monkeypatch):
    rng = np.random.default_rng(1)
    fr = frame(_scene(rng))
    calls = []
    lib = _native.kernel()

    class Spy:
        def tbot_measure(self, *a):
            calls.append(1)
            return lib.tbot_measure(*a)

    monkeypatch.setattr(_native, "kernel", lambda: Spy())
    nat, ref = _pair()
    assert nat.detect(fr) and calls
    calls.clear()
    assert ref.detect(fr) and not calls            # detector.native = false
    monkeypatch.undo()
    monkeypatch.setenv(_native.ENV, "0")
    try:
        assert _native.reload() is None and "off" in _native.status   # TBOT_NATIVE=0
    finally:
        monkeypatch.undo()
        assert _native.reload() is not None


def test_kernel_refuses_what_it_has_no_buffers_for():
    """A window larger than the kernel's buffers is a refusal (the Python loop then runs), never an overrun."""
    import ctypes

    lib = _native.kernel()
    p = _native.MeasureParams()
    p.h, p.w, p.moat_half = 4, 4, 41
    out = np.zeros((1, _native.OUT_COLS))
    lids = np.ones(1, np.int32)
    assert lib.tbot_measure(ctypes.byref(p), lids.ctypes.data, 1, out.ctypes.data) == -1
