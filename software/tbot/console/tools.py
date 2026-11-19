"""Helper processes (camera, servo: one at a time, they need the hardware the pipeline holds) and background
jobs (`tbot doctor`, `tbot clip`, `tbot score`, `tbot calib-boresight` ...) with captured output."""

from __future__ import annotations

import collections
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


class Helper:
    """A JSON-lines child process: commands in on stdin, events out on stdout."""

    def __init__(self, kind: str, argv: list[str], cwd: Path, on_exit: Callable[["Helper"], None] | None = None):
        self.kind = kind
        self.argv = argv
        self.started = time.time()
        self.events: collections.deque = collections.deque(maxlen=400)
        self.seq = 0
        self.state: dict[str, Any] = {}
        self.log: collections.deque = collections.deque(maxlen=200)
        self.cond = threading.Condition()
        self.proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self._on_exit = on_exit
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def _push(self, ev: dict) -> None:
        with self.cond:
            self.seq += 1
            ev["seq"] = self.seq
            ev.setdefault("t", time.time())
            self.events.append(ev)
            t = ev.get("type")
            if t in ("stats", "state", "ready", "fatal", "ring", "ring_error", "waiting"):
                self.state[t] = ev
            self.cond.notify_all()

    def _read_out(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                self.log.append(line)
                continue
            if isinstance(ev, dict):
                self._push(ev)
        code = self.proc.wait()
        self._push({"type": "exit", "code": code})
        if self._on_exit:
            self._on_exit(self)

    def _read_err(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.log.append(line.rstrip())

    def send(self, obj: dict) -> None:
        if not self.alive or self.proc.stdin is None:
            raise RuntimeError(f"the {self.kind} helper is not running")
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def events_since(self, seq: int, wait_s: float = 0.0) -> list[dict]:
        with self.cond:
            if wait_s > 0 and self.seq <= seq:
                self.cond.wait(wait_s)
            return [e for e in self.events if e["seq"] > seq]

    def stop(self, timeout: float = 5.0) -> None:
        if self.alive:
            try:
                self.send({"cmd": "quit"})
            except (RuntimeError, OSError, ValueError):
                pass
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()

    def describe(self) -> dict:
        return {"kind": self.kind, "alive": self.alive, "started": self.started, "seq": self.seq,
                "state": self.state, "log": list(self.log)[-20:], "args": self.argv[5:]}


class Job:
    _ids = itertools.count(1)

    def __init__(self, kind: str, argv: list[str], cwd: Path, title: str,
                 on_done: Callable[["Job"], None] | None = None, meta: dict | None = None):
        self.id = next(self._ids)
        self.kind = kind
        self.title = title
        self.argv = argv
        self.meta = meta or {}
        self.started = time.time()
        self.ended: float | None = None
        self.code: int | None = None
        self.output: list[str] = []
        self.result: Any = None
        self._on_done = on_done
        self.proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.output.append(line.rstrip("\n"))
            if len(self.output) > 4000:
                del self.output[:1000]
        self.code = self.proc.wait()
        if self._on_done:
            try:
                self._on_done(self)
            except Exception as exc:  # noqa: BLE001
                self.output.append(f"[console] post-processing failed: {exc}")
        self.ended = time.time()

    @property
    def running(self) -> bool:
        return self.ended is None

    def cancel(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()

    def describe(self, full: bool = True) -> dict:
        d = {"id": self.id, "kind": self.kind, "title": self.title, "started": self.started, "ended": self.ended,
             "code": self.code, "running": self.running, "meta": self.meta, "result": self.result}
        if full:
            d["output"] = self.output[-1500:]
        return d


class JobManager:
    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.jobs: dict[int, Job] = {}
        self.lock = threading.Lock()

    def start(self, kind: str, argv: list[str], title: str, on_done=None, meta: dict | None = None) -> Job:
        job = Job(kind, argv, self.cwd, title, on_done, meta)
        with self.lock:
            self.jobs[job.id] = job
            if len(self.jobs) > 40:  # forget the oldest finished ones
                for jid in sorted(self.jobs)[:-40]:
                    if not self.jobs[jid].running:
                        del self.jobs[jid]
        return job

    def get(self, jid: int) -> Job | None:
        return self.jobs.get(jid)

    def running(self, kind: str | None = None) -> list[Job]:
        return [j for j in self.jobs.values() if j.running and (kind is None or j.kind == kind)]

    def list(self) -> list[dict]:
        return [j.describe(full=False) for j in sorted(self.jobs.values(), key=lambda j: -j.id)]


def python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-u", "-m", module, *args]
