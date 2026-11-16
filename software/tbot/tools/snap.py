"""`tbot snap` — camera mode: the arming button takes a still picture and saves it to the SD card.

For calibration photos away from the network. Nothing here touches the trigger actuator: no PWM channel is
opened, the arming button is only read. Run it instead of `tbot run`, never beside it (`tbotctl camera on`
makes it the power-up service).

Each press saves the SHARPEST of the next `--burst` frames (hand-held shake) as a lossless PNG at full sensor
resolution, numbered on from the highest number already on the card (the Pi has no clock without the
network, so file times are unreliable; the numbers are not). Every file is fsync'ed before the next press is
accepted: switching the battery off any time after the LED blink loses nothing.

LED (gpio.latency_led_pin, if fitted): one short blink every 3 s = ready; long flash = picture saved;
fast flicker = camera down (still retrying).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from tbot.camera import linkcheck, open_live_source
from tbot.config import add_common_args, load_from_args

__all__ = ["next_index", "sharpness", "pick_sharpest", "save_png", "main"]

NAME_RE = re.compile(r"^snap_(\d{4,})\.png$")
MIN_FREE_MB = 200


def next_index(folder: Path) -> int:
    """1 + the highest snap number in `folder` (1 when empty)."""
    best = 0
    if folder.is_dir():
        for p in folder.iterdir():
            m = NAME_RE.match(p.name)
            if m:
                best = max(best, int(m.group(1)))
    return best + 1


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian (higher = sharper)."""
    g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def pick_sharpest(frames: list) -> tuple[int, float]:
    scores = [sharpness(f.image) for f in frames]
    i = int(np.argmax(scores))
    return i, scores[i]


def _fsync_dir(folder: Path) -> None:
    try:
        fd = os.open(str(folder), os.O_RDONLY)
    except OSError:
        return  # Windows: directories cannot be opened
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_png(path: Path, image: np.ndarray) -> None:
    """Write atomically and durably (temp file, fsync, rename, fsync the directory)."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG encode failed for {path.name}")
    tmp = path.with_suffix(".png.part")
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _append_log(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Led:
    """Status LED on gpio.latency_led_pin; silently absent when there is no GPIO."""

    def __init__(self, gpio, pin: int | None) -> None:
        self._gpio, self._pin = gpio, pin
        self._lock = threading.Lock()
        if gpio is not None and pin is not None:
            try:
                gpio.setup_output(pin, 0)
            except Exception as exc:  # noqa: BLE001 - LED is optional
                print(f"tbot snap: no status LED on GPIO{pin}: {exc}", file=sys.stderr)
                self._gpio = None

    def pulse(self, seconds: float) -> None:
        if self._gpio is None:
            return
        with self._lock:
            self._gpio.write(self._pin, 1)
            time.sleep(seconds)
            self._gpio.write(self._pin, 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tbot snap", description=__doc__.split("\n\n")[0])
    add_common_args(ap)
    ap.add_argument("--out", default="snaps", help="folder for the pictures (default snaps/, relative to the working dir)")
    ap.add_argument("--burst", type=int, default=5, help="frames per press; the sharpest is kept (default 5)")
    ap.add_argument("--full-res", dest="full_res", action=argparse.BooleanOptionalAction, default=True,
                    help="capture at 1280x800 (default) instead of the configured size")
    ap.add_argument("--max-exposure-us", type=float, default=4000.0,
                    help="auto-exposure time cap for stills (default 4000; the live pipeline uses camera.ae_max_exposure_us)")
    ap.add_argument("--stream", action="store_true", help="serve the live view on the debug server too")
    ap.add_argument("--no-button", action="store_true", help="no GPIO: take a picture each time Enter is pressed")
    args = ap.parse_args(argv)
    if args.burst < 1:
        ap.error("--burst must be >= 1")

    cfg, calib = load_from_args(args)
    if args.full_res:
        cfg.camera.width, cfg.camera.height = 1280, 800
        cfg.camera.fps = min(float(cfg.camera.fps), 15.0)  # stills: 15 fps is plenty and keeps the Pi cool for hours
    if cfg.camera.auto_exposure:
        cfg.camera.ae_max_exposure_us = float(args.max_exposure_us)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "snaps.jsonl"

    # the button: a real GPIO on the Pi, never the silent sim fallback (that would make the button dead)
    gpio = None
    presses: queue.Queue[float] = queue.Queue()
    if not args.no_button:
        from tbot.hw import gpio as gpio_mod  # noqa: PLC0415
        from tbot.hw.arming import ArmingSwitch  # noqa: PLC0415

        gpio = gpio_mod.open_gpio(cfg)
        if isinstance(gpio, gpio_mod.SimGpio) and gpio_mod.is_raspberry_pi():
            print("tbot snap: lgpio is not usable, so the button cannot be read (sudo apt install python3-lgpio)",
                  file=sys.stderr)
            return 2
        button = ArmingSwitch(gpio, cfg)
        button.on_change(lambda armed, t_ns: presses.put(time.monotonic()) if armed else None)
    else:
        def _stdin() -> None:
            for _ in sys.stdin:
                presses.put(time.monotonic())
        threading.Thread(target=_stdin, daemon=True).start()
    led = Led(gpio, int(cfg.gpio.latency_led_pin) if gpio is not None else None)

    server = None
    if args.stream:
        from tbot.app.stream import DebugServer  # noqa: PLC0415
        server = DebugServer(cfg).start()

    source = open_live_source(cfg.camera.source, cfg, calib)
    source.start()
    idx = next_index(out)
    print(f"tbot snap: camera mode, {cfg.camera.width}x{cfg.camera.height}, auto-exposure "
          f"{'on (cap ' + format(cfg.camera.ae_max_exposure_us, 'g') + ' us)' if cfg.camera.auto_exposure else 'off'}; "
          f"press the {'Enter key' if args.no_button else 'arming button'} to save -> {out.resolve()}/snap_{idx:04d}.png"
          + (f"; view http://{server.host}:{server.port}/" if server else ""), flush=True)

    last_frame = None
    down_since: float | None = None
    next_beat = next_diag = 0.0
    saved = 0
    try:
        while True:
            frame = source.read(timeout_s=0.5)
            now = time.monotonic()
            if frame is None:
                down_since = down_since or now
                if now - down_since > 3.0 and now >= next_beat:
                    for _ in range(4):
                        led.pulse(0.05)
                        time.sleep(0.05)
                    next_beat = now + 1.0
                if now - down_since >= 5.0 and now >= next_diag and getattr(source, "streaming", True):
                    diag = linkcheck.diagnose(window_s=1.5, source=source)
                    print(f"tbot snap: camera down {now - down_since:.0f} s, retrying - {diag.layer}: {diag.message}",
                          file=sys.stderr, flush=True)
                    next_diag = time.monotonic() + 15.0
                while not presses.empty():  # a press while the camera is down is not queued up for later
                    presses.get_nowait()
                    print("tbot snap: button pressed but the camera is down: nothing saved", file=sys.stderr, flush=True)
                continue
            if down_since is not None:
                print(f"tbot snap: camera back after {now - down_since:.1f} s", flush=True)
                down_since = None
            last_frame = frame
            if now >= next_beat:
                threading.Thread(target=led.pulse, args=(0.03,), daemon=True).start()
                next_beat = now + 3.0
            if server is not None and server.due():
                bgr = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR) if frame.image.ndim == 2 else frame.image.copy()
                cv2.putText(bgr, f"CAMERA MODE  next snap_{idx:04d}  exp {frame.exposure_us:.0f} us  gain {frame.gain:.2f}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                server.publish(bgr, {"mode": "camera", "next": idx, "saved": saved,
                                     "exposure_us": frame.exposure_us, "gain": frame.gain})
            try:
                t_press = presses.get_nowait()
            except queue.Empty:
                continue
            while not presses.empty():  # bounce / impatient double press: one picture
                presses.get_nowait()
            free_mb = shutil.disk_usage(out).free / 1e6
            if free_mb < MIN_FREE_MB:
                print(f"tbot snap: only {free_mb:.0f} MB free on the card: not saving", file=sys.stderr, flush=True)
                continue
            burst = [frame]
            while len(burst) < args.burst:
                f = source.read(timeout_s=1.0)
                if f is None:
                    break
                burst.append(f)
            best, score = pick_sharpest(burst)
            pick = burst[best]
            path = out / f"snap_{idx:04d}.png"
            save_png(path, pick.image)
            med = float(np.median(pick.image[::4, ::4]))
            record = {"file": path.name, "index": idx, "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "uptime_s": round(time.monotonic(), 1), "width": int(pick.image.shape[1]),
                      "height": int(pick.image.shape[0]), "exposure_us": round(float(pick.exposure_us), 1),
                      "gain": round(float(pick.gain), 3), "median_dn": med,
                      "saturated_pct": round(100.0 * float(np.mean(pick.image >= 250)), 3),
                      "sharpness": round(score, 1), "burst": len(burst), "picked": best,
                      "press_to_save_s": round(time.monotonic() - t_press, 2)}
            _append_log(log_path, record)
            print(f"tbot snap: saved {path.name}  exp {pick.exposure_us:.0f} us gain {pick.gain:.2f} "
                  f"median {med:.0f} DN sharp {score:.0f}", flush=True)
            saved += 1
            idx += 1
            led.pulse(0.6)
            while not presses.empty():  # presses during the save were the same press
                presses.get_nowait()
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        if server is not None:
            server.stop()
        if gpio is not None:
            try:
                gpio.close()
            except Exception:  # noqa: BLE001
                pass
    print(f"tbot snap: {saved} picture(s) saved this session", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
