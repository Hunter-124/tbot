"""Background previews and clips for every recorded press, done only while the gun is idle.

Live mode records presses but does not render their clips on the gun (record.clip = off: a render is a second
OpenCV process for 10-20 s, a core the next clay may need). This worker catches up instead, under rules that keep
it out of the way of a shot:

* it starts work only while the arming button is UP (pipeline telemetry), the Pi is below HOT_C, no camera or
  servo tool is open, and the press finished more than SETTLE_S ago (the pipeline's own renderer in dry/test
  mode, the recorder still writing);
* the moment the button is pressed (or the Pi gets hot) the running render is FROZEN (SIGSTOP: it uses no CPU at
  all) and continued where it stopped once the button is up again (and the Pi has cooled below COOL_C). One frozen
  for MAX_FROZEN_S is killed: `tbot clip` writes to a .part file and renames it at the end, so a killed render
  leaves no half clip; it is simply done again later;
* one thing at a time, at the lowest CPU priority (nice 19).

Previews (thumb.jpg + preview.json, tbot.console.thumbs) come first: they take a fraction of a second each.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from tbot.console import sessions as sess
from tbot.console.system import tbot_command

SETTLE_S = 45.0       # a press younger than this is left alone (still being written / rendered by the pipeline)
HOT_C = 76.0          # the Pi throttles at 80 C
POLL_S = 0.3          # how often a running job checks the arming button
IDLE_S = 8.0          # pause between scans when there is nothing to do
MAX_FROZEN_S = 900.0  # a render frozen this long (a long day of shooting) is dropped and redone later
COOL_C = 73.0         # after a heat pause, work resumes only below this (hysteresis)
IDLE_EXTRA_C = 3.0    # pipeline stopped (nothing can fire): both limits are this much higher (79 / 76 C)


class AutoClips:
    def __init__(self, root: Path, sessions_root: Callable[[], Path], busy: Callable[[], str | None],
                 settings_path: Path):
        self.root = root
        self.sessions_root = sessions_root
        self.busy = busy                    # -> None when work may start/continue, else the reason not to
        self.settings_path = settings_path
        self.state = "idle"
        self.reason = ""
        self.current: str | None = None
        self.pending_thumbs = 0
        self.pending_clips = 0
        self.done_clips = 0
        self.done_thumbs = 0
        self.last_error: str | None = None
        self._failed: dict[str, float] = {}  # session -> time it failed (retried after an hour)
        self._wake = threading.Event()
        threading.Thread(target=self._loop, name="autoclips", daemon=True).start()

    # ---------------------------------------------------------------- settings
    @property
    def enabled(self) -> bool:
        try:
            return bool(json.loads(self.settings_path.read_text()).get("auto_clips", True))
        except (OSError, ValueError):
            return True

    def set_enabled(self, on: bool) -> None:
        try:
            d = json.loads(self.settings_path.read_text())
        except (OSError, ValueError):
            d = {}
        d["auto_clips"] = bool(on)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(d, indent=1))
        self._wake.set()

    def describe(self) -> dict:
        return {"enabled": self.enabled, "state": self.state, "reason": self.reason, "current": self.current,
                "pending_clips": self.pending_clips, "pending_thumbs": self.pending_thumbs,
                "done_clips": self.done_clips, "done_thumbs": self.done_thumbs, "last_error": self.last_error}

    def kick(self) -> None:
        self._wake.set()

    # ---------------------------------------------------------------- work
    def _scan(self) -> tuple[list[Path], list[Path]]:
        root = self.sessions_root()
        if not root.is_dir():
            return [], []
        now = time.time()
        dirs = [d for d in root.iterdir() if d.is_dir() and sess.SESSION_NAME.match(d.name)]
        dirs.sort(key=lambda d: d.name, reverse=True)  # newest press first
        settled = []
        for d in dirs:
            try:
                age = now - (d / "summary.json").stat().st_mtime
            except OSError:
                continue
            if age >= SETTLE_S and now - self._failed.get(d.name, 0.0) > 3600.0:
                settled.append(d)
        return [d for d in settled if sess.needs_thumb(d)], [d for d in settled if sess.needs_clip(d)]

    def _run(self, argv: list[str], what: str) -> bool:
        """Run argv at the lowest priority. Whenever busy() says so it is FROZEN (SIGSTOP: no CPU at all, from the
        next scheduler tick) and continued later where it stopped; frozen longer than MAX_FROZEN_S it is killed
        (it is redone from the start later). True = finished OK."""
        pre = (lambda: os.nice(19)) if hasattr(os, "nice") else None  # (not on Windows: --dev)
        p = subprocess.Popen(argv, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             preexec_fn=pre)  # noqa: PLW1509 - nice only
        out: list[str] = []
        threading.Thread(target=lambda: out.extend(l.decode("utf-8", "replace") for l in p.stdout),  # type: ignore[union-attr]
                         daemon=True).start()
        can_freeze = hasattr(signal, "SIGSTOP")
        frozen_since: float | None = None
        working = self.state
        while p.poll() is None:
            why = self.busy() or (None if self.enabled else "turned off")
            if why and (not can_freeze or why == "turned off"):
                p.kill()
                p.wait()
                self.state, self.reason = "paused", why
                return False
            if why and frozen_since is None:
                p.send_signal(signal.SIGSTOP)
                frozen_since = time.monotonic()
            if why:
                self.state, self.reason = "paused", why
                if time.monotonic() - frozen_since > MAX_FROZEN_S:
                    p.kill()
                    p.wait()
                    return False
            elif frozen_since is not None:
                p.send_signal(signal.SIGCONT)
                frozen_since = None
                self.state, self.reason = working, ""
            time.sleep(POLL_S)
        if p.returncode != 0:
            self.last_error = f"{what}: {(out[-1] if out else '').strip()}"
        return p.returncode == 0

    def _loop(self) -> None:
        time.sleep(5.0)
        while True:
            try:
                self._step()
            except Exception as exc:  # noqa: BLE001 - the worker must never die
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.state = "idle"
            self._wake.wait(IDLE_S)
            self._wake.clear()

    def _step(self) -> None:
        self.current = None
        if not self.enabled:
            self.state, self.reason = "off", ""
            return
        thumbs, clips = self._scan()
        self.pending_thumbs, self.pending_clips = len(thumbs), len(clips)
        if not thumbs and not clips:
            self.state, self.reason = "idle", ""
            return
        why = self.busy()
        if why:
            self.state, self.reason = "paused", why
            return
        if thumbs:
            batch = thumbs[:12]
            self.state, self.reason, self.current = "previews", "", batch[0].name
            if self._run([sys.executable, "-m", "tbot.console.thumbs", *map(str, batch)], "preview"):
                self.done_thumbs += len(batch)
            for d in batch:  # a session that still has no preview failed: do not retry it every few seconds
                if sess.needs_thumb(d):
                    self._failed[d.name] = time.time()
            self._wake.set()
            return
        d = clips[0]
        self.state, self.reason, self.current = "rendering", "", d.name
        if self._run(tbot_command("clip", str(d), "--threads", "1"), f"clip {d.name}"):
            self.done_clips += 1
        elif self.state != "paused":
            self._failed[d.name] = time.time()
        self.current = None
        self._wake.set()
