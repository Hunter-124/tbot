"""Hardware-free tests for HITL audio timing and click detection."""
from __future__ import annotations

import time
import wave

import numpy as np

from tbot.hitl.audio import SimMic, StreamClock, detect_onset, write_wav


def test_stream_clock_lower_envelope():
    rng = np.random.default_rng(8)
    fs = 48000 * (1 + 80e-6)
    epoch = 123456789000000
    clock = StreamClock(48000, window_s=0.5)
    for n in range(96, 48000 * 20, 96):
        jitter = rng.exponential(150_000)
        clock.add(n, round(epoch + n * 1e9 / fs + jitter))
    assert abs(clock.ppm - 80) < 3
    assert abs(clock.time_of(0) - epoch) < 100_000
    n = 123456
    assert abs(clock.index_at(clock.time_of(n)) - n) < 0.001


def test_stream_clock_chunk_boundary_and_fallback():
    clock = StreamClock(1000)
    clock.add(10, 1_010_000_000)
    assert clock.time_of(10) == 1_010_000_000
    assert clock.time_of(9) == 1_009_000_000
    assert clock.fs_fitted == 1000


def test_ring_wrap_and_inclusive_clip():
    from tbot.hitl.audio import MicBase

    mic = MicBase(1000, 0, 0.005, 'test')
    mic._append(np.arange(4, dtype=np.int16), 1_004_000_000)
    mic._append(np.arange(4, 9, dtype=np.int16), 1_009_000_000)
    samples, first, fs = mic.clip(1_000_000_000, 1_007_000_000)
    assert first == 1_004_000_000
    assert fs == 1000
    np.testing.assert_array_equal(samples * 32768, [4, 5, 6, 7])
    assert mic.clip(1_000_000_000, 1_001_000_000) is None


def test_detect_click_with_whine_and_noise():
    fs = 48000
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.002, fs // 2)
    t = np.arange(len(x)) / fs
    whine = (t >= 0.15) & (t < 0.35)
    x[whine] += 0.1 * np.sin(2 * np.pi * 240 * t[whine])
    onset = round(0.3 * fs)
    burst = rng.normal(0, 0.5, 144) * np.exp(-np.arange(144) / 35)
    burst[0] = 0.5
    x[onset:onset + len(burst)] += burst
    found = detect_onset(x, fs, 0, search_from_ns=100_000_000, noise_to_ns=100_000_000)
    assert found is not None
    assert abs(found['t_ns'] - 300_000_000) < 200_000
    noise = rng.normal(0, 0.002, fs // 2)
    assert detect_onset(noise, fs, 0, search_from_ns=100_000_000) is None


def test_sim_mic_clicks():
    mic = SimMic(latency_ms=3, true_latency_ms=3, seed=4, jitter_ms=0.2)
    mic.start()
    try:
        now = time.monotonic_ns()
        clicks = [now + n * 1_000_000 for n in (180, 420, 660)]
        for t in clicks:
            mic.schedule_click(t)
        mic.schedule_noise(now + 100_000_000, now + 750_000_000, 0.05)
        time.sleep(0.85)
        for t in clicks:
            clip = mic.clip(t - 80_000_000, t + 30_000_000)
            assert clip is not None
            samples, first, fs = clip
            result = detect_onset(samples, fs, first, search_from_ns=t - 20_000_000,
                                  search_to_ns=t + 20_000_000, noise_to_ns=t - 40_000_000)
            assert result is not None
            assert abs(result['t_ns'] - t) < 500_000
    finally:
        mic.stop()


def test_wav(tmp_path):
    path = tmp_path / 'test.wav'
    write_wav(path, np.array([-1, 0, 1]), 48000)
    with wave.open(str(path), 'rb') as f:
        assert f.getnchannels() == 1
        assert f.getframerate() == 48000
        assert np.frombuffer(f.readframes(3), dtype='<i2').tolist() == [-32768, 0, 32767]
