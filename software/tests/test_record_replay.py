"""Session writer/replay round trips, the RAM ring recorder, CvSource video playback and the
Picamera2 pure helpers (no hardware, no GUI, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tbot.app.record import RingRecorder, SessionWriter
from tbot.camera.cv_source import CvSource
from tbot.camera.picam import exposure_mid_ns, extract_y_plane, select_sensor_mode
from tbot.camera.replay import ReplaySource
from tbot.config import load_config
from tbot.types import Frame, ImuSample

PERIOD_NS = 10_000_000  # 100 fps
T0_NS = 1_000_000_000
CAPTURE_LAG_NS = 3_000_000


def make_frames(n: int, height: int, width: int, channels: int, rng) -> list[Frame]:
    shape = (height, width) if channels == 1 else (height, width, channels)
    frames = []
    for i in range(n):
        t_exposure = T0_NS + i * PERIOD_NS
        frames.append(
            Frame(
                image=rng.integers(0, 256, shape, dtype=np.uint8),
                seq=i,
                t_exposure_ns=t_exposure,
                t_capture_ns=t_exposure + CAPTURE_LAG_NS,
                exposure_us=500.0 + i * 0.5,
                gain=1.0 + i * 0.125,
            )
        )
    return frames


def make_imu(t_start_ns: int, t_end_ns: int, step_ns: int, rng) -> list[ImuSample]:
    samples = []
    t = t_start_ns
    while t <= t_end_ns:
        samples.append(
            ImuSample(
                t_ns=t,
                gyro=rng.normal(0.0, 1.0, 3),
                accel=rng.normal(0.0, 9.81, 3),
                temp_c=32.5,
            )
        )
        t += step_ns
    return samples


# ------------------------------------------------------------------------- writer -> replay
@pytest.mark.parametrize("channels", [1, 3])
def test_session_round_trip(tmp_path: Path, channels: int) -> None:
    rng = np.random.default_rng(7)
    frames = make_frames(8, 24, 32, channels, rng)
    # IMU at 4x the frame rate, starting before the first frame and running past the last one.
    samples = make_imu(T0_NS - PERIOD_NS, frames[-1].t_capture_ns + 2 * PERIOD_NS, PERIOD_NS // 4, rng)
    events = [
        {"t_ns": frames[2].t_capture_ns, "type": "arm", "armed": True},
        {"t_ns": frames[5].t_capture_ns, "type": "fire", "track_id": 3},
    ]

    writer = SessionWriter(tmp_path / "sess", 32, 24, channels, 100.0, meta_extra={"source": "unit-test"})
    pending = list(samples)
    for frame in frames:
        due = [s for s in pending if s.t_ns <= frame.t_capture_ns]
        pending = pending[len(due):]
        writer.add_imu(due)
        for event in events:
            if event["t_ns"] == frame.t_capture_ns:
                writer.add_event(event)
        writer.add_frame(frame)
    writer.add_imu(pending)
    path = writer.close()
    assert writer.dropped == 0
    assert writer.frames_written == len(frames)

    meta = json.loads((path / "meta.json").read_text())
    assert meta["frames"] == len(frames)
    assert (meta["width"], meta["height"], meta["channels"]) == (32, 24, channels)
    assert meta["complete"] is True and meta["version"] == 1 and meta["source"] == "unit-test"

    source = ReplaySource(path)
    assert (source.width, source.height) == (32, 24)
    assert source.is_color == (channels == 3)
    assert source.nominal_fps == 100.0
    assert source.n_frames == len(frames)
    assert source.events == events

    imu = source.imu()
    delivered: list[ImuSample] = []
    previous_capture = -1
    for original in frames:
        got = source.read()
        assert got is not None
        assert np.array_equal(got.image, original.image)
        assert got.image.dtype == np.uint8
        assert (got.seq, got.t_exposure_ns, got.t_capture_ns) == (
            original.seq, original.t_exposure_ns, original.t_capture_ns)
        assert got.exposure_us == original.exposure_us and got.gain == original.gain
        assert source.clock.now_ns() >= got.t_capture_ns

        batch = imu.read()
        assert [s.t_ns for s in batch] == [
            s.t_ns for s in samples if previous_capture < s.t_ns <= got.t_capture_ns]
        for s in batch:
            assert s.t_ns <= got.t_capture_ns
        delivered += batch
        previous_capture = got.t_capture_ns

    assert source.read() is None
    assert imu.read() == []  # nothing new once the frames are exhausted
    expected = [s for s in samples if s.t_ns <= frames[-1].t_capture_ns]
    assert [s.t_ns for s in delivered] == [s.t_ns for s in expected]
    for got, want in zip(delivered, expected):
        assert np.allclose(got.gyro, want.gyro) and np.allclose(got.accel, want.accel)
        assert got.temp_c == want.temp_c
    assert source.dropped == 0


def test_writer_rejects_bad_input(tmp_path: Path) -> None:
    writer = SessionWriter(tmp_path / "s", 8, 4, 1, 30.0)
    try:
        with pytest.raises(ValueError):
            writer.add_frame(Frame(image=np.zeros((4, 8, 3), np.uint8), seq=0, t_exposure_ns=1, t_capture_ns=2))
        with pytest.raises(ValueError):
            writer.add_frame(Frame(image=np.zeros((4, 8), np.uint16), seq=0, t_exposure_ns=1, t_capture_ns=2))
        with pytest.raises(ValueError):
            writer.add_event({"type": "arm"})
    finally:
        writer.close()
    with pytest.raises(RuntimeError):
        writer.add_event({"t_ns": 1, "type": "arm"})
    with pytest.raises(ValueError):
        SessionWriter(tmp_path / "bad", 8, 4, 2, 30.0)


def test_replay_trusts_what_is_on_disk(tmp_path: Path) -> None:
    """A session cut short mid-write replays the frames that actually made it to frames.bin."""
    rng = np.random.default_rng(3)
    frames = make_frames(5, 6, 8, 1, rng)
    writer = SessionWriter(tmp_path / "s", 8, 6, 1, 50.0)
    for frame in frames:
        writer.add_frame(frame)
    path = writer.close()

    binary = path / "frames.bin"
    binary.write_bytes(binary.read_bytes()[: 3 * 6 * 8 + 10])  # a torn 4th frame
    source = ReplaySource(path)
    assert source.n_frames == 3
    assert [f.seq for f in iter(lambda: source.read(), None)] == [0, 1, 2]


def test_replay_realtime_skips_late_frames(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    frames = make_frames(6, 6, 8, 1, rng)
    writer = SessionWriter(tmp_path / "s", 8, 6, 1, 100.0)
    for frame in frames:
        writer.add_frame(frame)
    path = writer.close()

    source = ReplaySource(path, realtime=True)
    first = source.read()
    assert first is not None and first.seq == 0
    source.clock.advance(3 * PERIOD_NS)  # pretend processing took three frame periods
    late = source.read()
    assert late is not None and late.seq >= 3
    assert source.dropped == late.seq - 1
    plain = ReplaySource(path, realtime=False)
    assert [f.seq for f in iter(lambda: plain.read(), None)] == [f.seq for f in frames]
    assert plain.dropped == 0


# ------------------------------------------------------------------------------ ring recorder
def ring_config(tmp_path: Path, **overrides):
    sets = [f"record.dir={tmp_path}"] + [f"record.{k}={v}" for k, v in overrides.items()]
    return load_config(sets=sets)


def test_ring_recorder_window(tmp_path: Path) -> None:
    cfg = ring_config(tmp_path, preroll_s=0.2, postroll_s=0.1, max_ram_mb=64)
    rng = np.random.default_rng(5)
    frames = make_frames(60, 12, 16, 1, rng)
    recorder = RingRecorder(cfg, 16, 12, 1, 100.0)

    trigger_frame = frames[40]
    t_trigger = trigger_frame.t_capture_ns
    lo, hi = t_trigger - 200_000_000, t_trigger + 100_000_000
    samples = make_imu(T0_NS, frames[-1].t_capture_ns, PERIOD_NS, rng)
    by_time = {s.t_ns: s for s in samples}

    path = None
    for frame in frames:
        sample = by_time.get(frame.t_exposure_ns)  # pipeline order: IMU drain, then the frame
        if sample is not None:
            recorder.push_imu([sample])
        recorder.push(frame)
        if frame is trigger_frame:
            recorder.event({"t_ns": t_trigger, "type": "arm"})
            path = recorder.trigger(t_trigger, tag="arm")
    assert path is not None and path.parent == tmp_path and path.name.endswith("-arm")
    recorder.close()
    assert recorder.sessions == [path]

    expected = [f for f in frames if lo <= f.t_capture_ns <= hi]
    assert len(expected) == 31  # 20 preroll + trigger + 10 postroll frames at 100 fps
    source = ReplaySource(path)
    assert source.n_frames == len(expected)
    replayed = list(iter(lambda: source.read(), None))
    assert [f.seq for f in replayed] == [f.seq for f in expected]
    for got, want in zip(replayed, expected):
        assert np.array_equal(got.image, want.image)
    assert lo <= replayed[0].t_capture_ns and replayed[-1].t_capture_ns <= hi
    assert source.events == [{"t_ns": t_trigger, "type": "arm"}]
    # the dump carries the IMU samples of the same window, delivered in step with its frames
    paced = ReplaySource(path)
    imu = paced.imu()
    dumped: list[ImuSample] = []
    while paced.read() is not None:
        dumped += imu.read()
    # samples sit at exposure time, 3 ms before capture, so the window holds one fewer than frames
    windowed = [s for s in samples if lo <= s.t_ns <= hi]
    assert len(windowed) == len(expected) - 1
    assert [s.t_ns for s in dumped] == [s.t_ns for s in windowed]


def test_ring_recorder_ram_cap_evicts_oldest(tmp_path: Path) -> None:
    cfg = ring_config(tmp_path, preroll_s=60.0, postroll_s=0.1, max_ram_mb=1)
    rng = np.random.default_rng(9)
    frames = make_frames(40, 256, 256, 1, rng)  # 64 KiB each -> 16 fit in 1 MiB
    recorder = RingRecorder(cfg, 256, 256, 1, 100.0)
    for frame in frames:
        recorder.push(frame)
    assert recorder.ram_bytes <= 1024 * 1024
    assert recorder.n_frames == 16
    assert recorder.evicted == len(frames) - 16

    path = recorder.trigger(frames[-1].t_capture_ns, tag="cap")
    recorder.close()
    source = ReplaySource(path)
    assert [f.seq for f in iter(lambda: source.read(), None)] == [f.seq for f in frames[-16:]]


# ----------------------------------------------------------------------------------- CvSource
def write_video(path: Path, n: int, width: int, height: int, fps: float) -> None:
    fourcc = (cv2.VideoWriter_fourcc(*"MJPG") if hasattr(cv2, "VideoWriter_fourcc")
              else cv2.VideoWriter.fourcc(*"MJPG"))
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "no MJPG encoder available"
    for i in range(n):
        image = np.zeros((height, width, 3), np.uint8)
        image[:, :, 0] = 20 * i                      # blue ramp over time
        image[:, :, 2] = 200                         # strong red everywhere
        image[: height // 2, :, 1] = 150             # green in the top half
        writer.write(image)
    writer.release()


def read_all(source: CvSource) -> list[Frame]:
    source.start()
    try:
        return list(iter(lambda: source.read(0.5), None))
    finally:
        source.stop()


def test_cv_source_video_playback(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    write_video(path, 10, 64, 48, 30.0)
    cfg = load_config(sets=["camera.emulate_mono=none", "latency.exposure_to_user_ms=8.0"])
    source = CvSource(cfg, f"video:{path}")
    assert source.is_color and (source.width, source.height) == (64, 48)
    assert source.nominal_fps == pytest.approx(30.0)

    frames = read_all(source)
    assert len(frames) == 10
    assert source.dropped == 0
    assert [f.seq for f in frames] == list(range(10))
    assert all(f.image.shape == (48, 64, 3) for f in frames)
    assert frames[0].t_exposure_ns == T0_NS
    steps = np.diff([f.t_exposure_ns for f in frames])
    assert np.all(steps > 0)
    assert np.allclose(steps, 1e9 / 30.0, rtol=0.0, atol=1e6)
    assert all(f.t_capture_ns - f.t_exposure_ns == 8_000_000 for f in frames)
    assert source.clock.now_ns() >= frames[-1].t_capture_ns

    luma = read_all(CvSource(load_config(sets=["camera.emulate_mono=luma"]), f"video:{path}"))
    red = read_all(CvSource(load_config(sets=["camera.emulate_mono=red"]), f"video:{path}"))
    assert len(luma) == len(red) == 10
    for colour, gray, r in zip(frames, luma, red):
        assert gray.image.shape == (48, 64) and not gray.is_color
        assert r.image.shape == (48, 64) and not r.is_color
        assert np.array_equal(r.image, colour.image[:, :, 2])
        assert np.array_equal(gray.image, cv2.cvtColor(colour.image, cv2.COLOR_BGR2GRAY))
        assert r.t_exposure_ns == colour.t_exposure_ns


def test_cv_source_errors(tmp_path: Path) -> None:
    cfg = load_config()
    with pytest.raises(FileNotFoundError):
        CvSource(cfg, f"video:{tmp_path / 'missing.avi'}")
    with pytest.raises(ValueError):
        CvSource(cfg, "cv:usb0")
    with pytest.raises(ValueError):
        CvSource(cfg, "video:")
    with pytest.raises(ValueError):
        CvSource(load_config(sets=["camera.emulate_mono=sepia"]), "video:x.avi")


# ------------------------------------------------------------------------------ picam helpers
OV9281_MODES = [
    {"size": (1280, 800), "format": "R8", "fps": 144.0},
    {"size": (1280, 720), "format": "R8", "fps": 160.0},
    {"size": (640, 400), "format": "R8", "fps": 309.0},
]


@pytest.mark.parametrize(
    "request_size,expected",
    [
        ((640, 400), (640, 400)),     # exact (the binned default)
        ((1280, 800), (1280, 800)),   # exact
        ((320, 240), (640, 400)),     # smallest covering mode
        ((1280, 400), (1280, 720)),   # smallest covering mode, not the biggest
        ((1920, 1080), (1280, 800)),  # nothing covers it: the largest mode
    ],
)
def test_select_sensor_mode(request_size, expected) -> None:
    mode = select_sensor_mode(OV9281_MODES, *request_size)
    assert mode["size"] == expected


def test_select_sensor_mode_prefers_faster_on_ties_and_needs_modes() -> None:
    modes = [{"size": (640, 400), "fps": 60.0}, {"size": (640, 400), "fps": 210.0}]
    assert select_sensor_mode(modes, 640, 400)["fps"] == 210.0
    with pytest.raises(ValueError):
        select_sensor_mode([], 640, 400)


def test_extract_y_plane_handles_stride_padding() -> None:
    width, height, stride = 640, 400, 704  # libcamera pads rows to a 64-byte multiple
    rng = np.random.default_rng(1)
    buffer = rng.integers(0, 256, (height * 3 // 2, stride), dtype=np.uint8)

    y = extract_y_plane(buffer, width, height)
    assert y.shape == (height, width) and y.dtype == np.uint8
    assert y.flags["C_CONTIGUOUS"]
    assert np.array_equal(y, buffer[:height, :width])
    buffer[:] = 0  # the request buffer is released right after extraction
    assert y.any()

    unpadded = rng.integers(0, 256, (height * 3 // 2, width), dtype=np.uint8)
    y2 = extract_y_plane(unpadded, width, height)
    assert np.array_equal(y2, unpadded[:height])
    unpadded[:] = 0
    assert y2.any()

    with pytest.raises(ValueError):
        extract_y_plane(np.zeros((height, width, 3), np.uint8), width, height)
    with pytest.raises(ValueError):
        extract_y_plane(np.zeros((height - 1, width), np.uint8), width, height)
    with pytest.raises(ValueError):
        extract_y_plane(np.zeros((height * 3 // 2, width - 1), np.uint8), width, height)


def test_exposure_mid_ns() -> None:
    # Mid-exposure is half an exposure before the start-of-readout stamp, plus the calibrated offset.
    assert exposure_mid_ns(1_000_000_000, 500.0, 0) == 1_000_000_000 - 250_000
    assert exposure_mid_ns(1_000_000_000, 500.0, 12_345) == 1_000_000_000 - 250_000 + 12_345
    assert exposure_mid_ns(1_000_000_000, 0.0, -1_000) == 999_999_000
    assert exposure_mid_ns(2_000_000_000, 1.0, 0) == 2_000_000_000 - 500  # rounds to the nearest ns
    assert isinstance(exposure_mid_ns(1, 3.0, 0), int)
