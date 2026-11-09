"""Microphone capture, sample-clock estimation and transient detection for HITL."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np


class StreamClock:
    """Fit the lower arrival-time envelope; n_end denotes the exclusive chunk boundary.

    Thus time_of(n_end - 1) is one sample period before the fitted boundary.
    Arrival latency remains in the fit and must be calibrated separately.
    """

    def __init__(self, fs_nominal: float, window_s: float = 0.5):
        if fs_nominal <= 0 or window_s <= 0:
            raise ValueError("sample rate and window must be positive")
        self.fs_nominal = float(fs_nominal)
        self.window_s = float(window_s)
        self._dt = 1e9 / self.fs_nominal
        self._bins: dict[int, tuple[float, int, int]] = {}
        self._origin: int | None = None
        self._dirty = True
        self._slope = self._dt
        self._intercept = 0.0

    def add(self, n_end: int, t_arrival_ns: int) -> None:
        if self._origin is None:
            self._origin = n_end
        key = int((n_end - self._origin) / (self.fs_nominal * self.window_s))
        residual = t_arrival_ns - n_end * self._dt
        previous = self._bins.get(key)
        if previous is None or residual < previous[0]:
            self._bins[key] = (residual, n_end, t_arrival_ns)
            self._dirty = True

    def _fit(self) -> None:
        if not self._dirty:
            return
        if not self._bins:
            raise RuntimeError("sample clock has no observations")
        keys = sorted(self._bins)
        if len(keys) >= 4:
            keys = keys[1:]  # the first window holds the capture start-up burst (prefilled periods)
        points = [self._bins[k] for k in keys]
        if len(points) < 2:
            self._slope = self._dt
            self._intercept = min(p[0] for p in points)
        else:
            # Center both axes before regression to retain nanosecond precision.
            x = np.asarray([p[1] for p in points], dtype=np.float64)
            y = np.asarray([p[2] for p in points], dtype=np.float64)
            keep = np.ones(len(x), dtype=bool)

            def line(sel: np.ndarray) -> None:
                xm, ym = x[sel].mean(), y[sel].mean()
                dx = x[sel] - xm
                denom = float(np.dot(dx, dx))
                self._slope = float(np.dot(dx, y[sel] - ym) / denom) if denom else self._dt
                self._intercept = float(ym - self._slope * xm)

            for _ in range(4):
                line(keep)
                # Drop windows whose minimum sits far off the line (a missed wake-up or a burst), then refit.
                res = y - (self._intercept + self._slope * x)
                med = float(np.median(res[keep]))
                mad = float(np.median(np.abs(res[keep] - med)))
                tol = max(4.4478 * mad, 200_000.0)  # 3 sigma, floored at 0.2 ms
                new_keep = np.abs(res - med) <= tol
                if new_keep.sum() < 2 or np.array_equal(new_keep, keep):
                    break
                keep = new_keep
            else:
                line(keep)
        self._dirty = False

    def time_of(self, n: float) -> int:
        self._fit()
        return round(self._intercept + self._slope * n)

    def index_at(self, t_ns: int) -> float:
        self._fit()
        return (t_ns - self._intercept) / self._slope

    @property
    def fs_fitted(self) -> float:
        self._fit()
        return 1e9 / self._slope

    @property
    def ppm(self) -> float:
        return (self.fs_fitted / self.fs_nominal - 1.0) * 1e6


class MicBase:
    """Thread-safe bounded sample storage shared by capture implementations."""

    def __init__(self, fs: int, latency_ms: float, keep_s: float, device: str):
        if fs <= 0 or keep_s <= 0:
            raise ValueError("sample rate and retention must be positive")
        self.fs = int(fs)
        self.latency_ns = round(latency_ms * 1e6)
        self.device = device
        self._buffer = np.empty(max(1, round(fs * keep_s)), dtype=np.int16)
        self._lock = threading.Lock()
        self._clock = StreamClock(fs)
        self._samples = 0
        self._chunks = 0
        self._overruns = 0
        self._jitter: list[float] = []
        self._previous_arrival: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _append(self, data: np.ndarray, arrival_ns: int) -> None:
        count = len(data)
        if not count:
            return
        with self._lock:
            end = self._samples + count
            if self._previous_arrival is not None:
                self._jitter.append(((arrival_ns - self._previous_arrival) - count * 1e9 / self.fs) / 1e6)
                if len(self._jitter) > 10000:
                    del self._jitter[:5000]
            self._previous_arrival = arrival_ns
            capacity = len(self._buffer)
            kept = min(count, capacity)
            start = (end - kept) % capacity
            first = min(kept, capacity - start)
            self._buffer[start:start + first] = data[-kept:-kept + first] if first < kept else data[-kept:]
            if first < kept:
                self._buffer[:kept - first] = data[-kept + first:]
            self._samples = end
            self._chunks += 1
            self._clock.add(end, arrival_ns)

    def clip(self, t0_ns: int, t1_ns: int) -> tuple[np.ndarray, int, float] | None:
        if t1_ns < t0_ns:
            raise ValueError("clip end precedes start")
        with self._lock:
            if not self._samples:
                return None
            # Request bounds are inclusive, expressed on the corrected capture clock.
            start = max(0, self._samples - len(self._buffer), math.ceil(self._clock.index_at(t0_ns + self.latency_ns)))
            end = min(self._samples, math.floor(self._clock.index_at(t1_ns + self.latency_ns)) + 1)
            if end <= start:
                return None
            pos = start % len(self._buffer)
            length = end - start
            first = min(length, len(self._buffer) - pos)
            samples = np.empty(length, dtype=np.float32)
            samples[:first] = self._buffer[pos:pos + first]
            if first < length:
                samples[first:] = self._buffer[:length - first]
            samples *= 1.0 / 32768.0
            return samples, self._clock.time_of(start) - self.latency_ns, self._clock.fs_fitted

    def stats(self) -> dict:
        with self._lock:
            fitted = self._clock.fs_fitted if self._samples else float(self.fs)
            jitter = np.abs(self._jitter)
            return {"fs_nominal": self.fs, "fs_fitted": fitted,
                    "ppm": (fitted / self.fs - 1) * 1e6, "chunks": self._chunks,
                    "samples": self._samples, "overruns": self._overruns,
                    "arrival_jitter_ms": {"p50": float(np.percentile(jitter, 50)) if len(jitter) else 0.0,
                                          "p95": float(np.percentile(jitter, 95)) if len(jitter) else 0.0},
                    "device": self.device}

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class ArecordMic(MicBase):
    def __init__(self, device: str = "default", fs: int = 48000, latency_ms: float = 0.0,
                 period_us: int = 2000, buffer_us: int = 20000, keep_s: float = 30.0):
        super().__init__(fs, latency_ms, keep_s, device)
        self.period_us, self.buffer_us = period_us, buffer_us
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._errors: list[str] = []

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("microphone already started")
        command = ["arecord", "-q", "-D", self.device, "-f", "S16_LE", "-c", "1", "-r", str(self.fs),
                   "-t", "raw", f"--period-time={self.period_us}", f"--buffer-time={self.buffer_us}"]
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise RuntimeError(f"Cannot start arecord: {exc}\n{list_devices()}") from exc
        self._stop.clear()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._thread.start()
        self._stderr_thread.start()
        try:
            self._process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            return
        self._thread.join(timeout=1)
        self._stderr_thread.join(timeout=1)
        detail = ''.join(self._errors).strip()
        self._process = None
        raise RuntimeError(f"arecord exited immediately for device {self.device!r}: {detail}\n{list_devices()}")

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        pending = b''
        while not self._stop.is_set():
            try:
                data = os.read(self._process.stdout.fileno(), max(2, round(self.fs * self.period_us / 1e6) * 2))
            except OSError:
                break
            arrival = time.monotonic_ns()
            if not data:
                break
            data = pending + data
            pending = data[len(data) & ~1:]
            data = data[:len(data) & ~1]
            self._append(np.frombuffer(data, dtype='<i2'), arrival)

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            text = line.decode(errors='replace')
            self._errors.append(text)
            if 'overrun' in text.lower():
                with self._lock:
                    self._overruns += 1

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if self._thread:
                self._thread.join()
            if self._stderr_thread:
                self._stderr_thread.join()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
            self._process = None


class SimMic(MicBase):
    def __init__(self, fs: int = 48000, latency_ms: float = 0.0, true_latency_ms: float = 3.0,
                 noise: float = 0.002, period_ms: float = 2.0, jitter_ms: float = 0.5,
                 seed: int = 0, keep_s: float = 30.0):
        super().__init__(fs, latency_ms, keep_s, 'sim')
        if period_ms <= 0 or noise < 0 or jitter_ms < 0 or true_latency_ms < 0:
            raise ValueError("invalid simulation timing or noise")
        self.true_latency_ms, self.noise, self.jitter_ms = true_latency_ms, noise, jitter_ms
        self._period = max(1, round(fs * period_ms / 1000))
        self._rng = np.random.default_rng(seed)
        self._clicks: list[tuple[int, float, np.ndarray]] = []
        self._noises: list[tuple[int, int, float]] = []
        self._events_lock = threading.Lock()
        self._seed = seed
        self._epoch_ns = 0

    def schedule_click(self, t_ns: int, amplitude: float = 0.5) -> None:
        # Event waveform uses a separate deterministic stream from background samples.
        with self._events_lock:
            rng = np.random.default_rng(np.random.SeedSequence([self._seed, len(self._clicks)]))
            count = max(1, round(self.fs * 0.004))
            waveform = rng.normal(size=count) * np.exp(-np.arange(count) / (self.fs * 0.0007))
            waveform[0] = 1.0
            self._clicks.append((t_ns, amplitude, waveform))

    def schedule_noise(self, t0_ns: int, t1_ns: int, amplitude: float = 0.05) -> None:
        if t1_ns < t0_ns:
            raise ValueError("noise end precedes start")
        with self._events_lock:
            self._noises.append((t0_ns, t1_ns, amplitude))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("microphone already started")
        self._stop.clear()
        self._epoch_ns = time.monotonic_ns() - round(self._samples * 1e9 / self.fs)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            start = self._samples
            indices = np.arange(start, start + self._period)
            times = self._epoch_ns + indices * (1e9 / self.fs)
            values = self._rng.normal(0, self.noise, self._period)
            with self._events_lock:
                for t_ns, amplitude, waveform in self._clicks:
                    offset = math.ceil((t_ns - self._epoch_ns) * self.fs / 1e9)
                    wi = indices - offset
                    mask = (wi >= 0) & (wi < len(waveform))
                    values[mask] += amplitude * waveform[wi[mask]]
                for t0, t1, amplitude in self._noises:
                    mask = (times >= t0) & (times < t1)
                    count = int(mask.sum())
                    if count:
                        phase = (times[mask] - t0) * (2 * np.pi * 240 / 1e9)
                        values[mask] += amplitude * (np.sin(phase) + 0.2 * self._rng.normal(size=count))
            arrival = int(times[-1] + (self.true_latency_ms + abs(self._rng.normal(0, self.jitter_ms))) * 1e6)
            if self._stop.wait(max(0.0, (arrival - time.monotonic_ns()) / 1e9)):
                break
            self._append(np.clip(np.rint(values * 32768), -32768, 32767).astype(np.int16), time.monotonic_ns())

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
            self._thread = None


def detect_onset(samples: np.ndarray, fs: float, t_first_ns: int, *, search_from_ns: int | None = None,
                 search_to_ns: int | None = None, noise_to_ns: int | None = None) -> dict | None:
    """Find the dominant high-frequency transient and return its threshold crossing."""
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if fs <= 0:
        raise ValueError("sample rate must be positive")
    if len(samples) < 2:
        return None
    diff = np.abs(np.diff(samples, prepend=samples[0]))
    # Trailing RMS retains the onset timing without moving energy backward in time.
    width = max(1, round(fs * 0.00025))
    cumulative = np.concatenate(([0.0], np.cumsum(diff * diff)))
    idx = np.arange(len(diff))
    envelope = np.sqrt((cumulative[idx + 1] - cumulative[np.maximum(0, idx + 1 - width)]) / np.minimum(idx + 1, width))
    def index(t: int | None, default: int) -> int:
        return default if t is None else max(0, min(len(samples), math.ceil((t - t_first_ns) * fs / 1e9)))
    lo = index(search_from_ns, 0)
    hi = index(search_to_ns, len(samples))
    noise_end = index(noise_to_ns, lo)
    noise_data = envelope[:noise_end]
    if not len(noise_data):
        noise_data = envelope
    median = float(np.median(noise_data))
    mad = float(np.median(np.abs(noise_data - median)))
    noise = median + 1.4826 * mad
    if hi <= lo:
        return None
    peak_index = lo + int(np.argmax(envelope[lo:hi]))
    peak = float(envelope[peak_index])
    snr = peak / max(noise, 1e-12)
    if snr < 6:
        return None
    threshold = noise + 0.2 * (peak - noise)
    onset = peak_index
    while onset > lo and envelope[onset - 1] >= threshold:
        onset -= 1
    return {"t_ns": t_first_ns + round(onset * 1e9 / fs), "peak": peak,
            "noise": noise, "snr": snr, "index": onset}


def write_wav(path: str | Path, samples: np.ndarray, fs: int) -> None:
    pcm = np.clip(np.rint(np.asarray(samples) * 32768), -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(fs)
        out.writeframes(pcm.tobytes())


def list_devices() -> str:
    if shutil.which('arecord') is None:
        return 'arecord not found; install alsa-utils to list capture devices.'
    outputs = []
    for flag in ('-l', '-L'):
        try:
            result = subprocess.run(['arecord', flag], capture_output=True, text=True, timeout=5)
            outputs.append(f"arecord {flag}:\n{result.stdout}{result.stderr}".strip())
        except (OSError, subprocess.TimeoutExpired) as exc:
            outputs.append(f"arecord {flag}: {exc}")
    return '\n'.join(outputs)
