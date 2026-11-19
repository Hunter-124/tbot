"""`tbot console`: the gun's web app and JSON API (see tbot/console/__init__.py, docs/CONSOLE.md).

    tbot console [--port 80] [--dev]

No authentication by design: the gun lives on a direct cable (or its own network) and every action that can
move the trigger still needs a human holding the arming button. LIVE mode needs the word LIVE typed, as
tbotctl does.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tbot import __version__
from tbot.console import API_VERSION, sessions as sess, tomledit
from tbot.console.autoclips import COOL_C, HOT_C, IDLE_EXTRA_C, AutoClips
from tbot.console.system import DevSystem, PiSystem, MODES, software_root, tbot_command
from tbot.console.tools import Helper, JobManager, python_module

log = logging.getLogger("tbot.console")
WEB = Path(__file__).resolve().parent / "web"
CALIB_FILES = ("intrinsics", "boresight", "imu", "latency", "detector", "bench")


def _finite(obj: Any) -> Any:
    """NaN / Infinity (calibration files have them) become null: browsers only parse strict JSON."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Console:
    def __init__(self, root: Path, dev: bool, pipeline_port: int = 8080, cam_port: int = 8081,
                 sessions_dir: str | None = None, dev_camera: str = "synthetic"):
        self.root = root
        self.dev_camera = dev_camera
        self.dev = dev
        self.system = DevSystem(root, pipeline_port) if dev else PiSystem(root)
        self.pipeline_port = pipeline_port
        self.cam_port = cam_port
        self.jobs = JobManager(root)
        self.helper: Helper | None = None
        self.helper_lock = threading.Lock()
        self._sessions_override = Path(sessions_dir).resolve() if sessions_dir else None
        self._pipe_cache: tuple[float, dict] = (0.0, {})
        self.hostname = socket.gethostname()
        self._tele_misses = 0
        self._hot = False
        self.autoclips = AutoClips(root, self.sessions_root, self._clip_busy, root / "config" / "console.json")

    def _clip_busy(self) -> str | None:
        """Why background clip work must not run right now (None = it may)."""
        if self.pipeline()["active"]:
            # Never on the gun while it can shoot (2026-11-19, the range: the gun stopped working between presses
            # and rendering was the suspect). The laptop app renders every press from its raw data instead.
            return "the pipeline is running: the laptop app makes the clips"
        h = self.helper
        if h is not None and h.alive:
            return "a calibration tool is open"
        if self.jobs.running("clip") or self.jobs.running("hwcheck"):
            return "another job is running"
        temp = self.system.health().get("cpu_temp_c")
        if temp is not None:  # hysteresis: stop at HOT_C, go on only once below COOL_C
            # with the pipeline stopped no shot depends on the CPU: work on up to just under the throttle point
            extra = 0.0 if self.pipeline()["active"] else IDLE_EXTRA_C
            self._hot = temp >= (COOL_C if self._hot else HOT_C) + extra
            if self._hot:
                return f"the Pi is hot ({temp:.0f} C)"
        if self.pipeline()["active"]:
            tele = self.pipeline_telemetry(timeout_s=1.5)
            if tele is None:
                # one slow answer (a busy Pi) is not a reason to throw away a render; a pipeline that has not
                # answered for a few seconds (starting up, or stuck) is
                self._tele_misses += 1
                return "the pipeline is not answering" if self._tele_misses >= 6 else None
            self._tele_misses = 0
            if tele.get("armed"):
                return "the arming button is held"
        return None

    # ------------------------------------------------------------------ config
    @property
    def default_path(self) -> Path:
        return self.root / "config" / "default.toml"

    @property
    def local_path(self) -> Path:
        env = os.environ.get("TBOT_LOCAL_CONFIG")
        if env:
            p = Path(env)
            return p if p.is_absolute() else self.root / p
        return self.root / "config" / "local.toml"

    def _local_text(self) -> str:
        try:
            return self.local_path.read_text()
        except OSError:
            return ""

    def schema(self) -> dict:
        return tomledit.parse_schema(self.default_path.read_text(), self._local_text())

    def effective(self, dotted: str) -> Any:
        for sec in self.schema()["sections"]:
            for k in sec["keys"]:
                if k["key"] == dotted:
                    return k["value"]
        raise KeyError(dotted)

    def sessions_root(self) -> Path:
        if self._sessions_override:
            return self._sessions_override
        try:
            d = Path(str(self.effective("record.dir")))
        except KeyError:
            d = Path("sessions")
        return d if d.is_absolute() else self.root / d

    def clip_fps(self) -> float:
        try:
            return float(self.effective("record.clip_fps"))
        except (KeyError, TypeError, ValueError):
            return 30.0

    def write_config(self, changes: dict[str, Any], reset: list[str]) -> dict:
        schema = self.schema()
        keys = {k["key"]: k for s in schema["sections"] for k in s["keys"]}
        coerced: dict[str, Any] = {}
        for dotted, value in changes.items():
            if dotted not in keys:
                raise ApiError(400, f"unknown setting {dotted}")
            try:
                coerced[dotted] = tomledit.coerce(value, keys[dotted]["default"], dotted)
            except ValueError as exc:
                raise ApiError(400, str(exc)) from None
            choices = keys[dotted].get("choices")
            if choices and coerced[dotted] not in choices:
                raise ApiError(400, f"{dotted} must be one of: {', '.join(choices)}")
        for dotted in reset:
            if dotted not in keys:
                raise ApiError(400, f"unknown setting {dotted}")
        old = self._local_text()
        header = "# Per-rig overrides of config/default.toml (edit freely; the TBOT Console keeps comments).\n"
        new = tomledit.patch_text(old or header, coerced, set(reset))
        self._validate(new)
        if old == new:
            return {"changed": False}
        if old:
            bdir = self.root / "config" / "backups"
            bdir.mkdir(parents=True, exist_ok=True)
            (bdir / time.strftime("local-%Y%m%d-%H%M%S.toml")).write_text(old)
            for stale in sorted(bdir.glob("local-*.toml"))[:-30]:
                stale.unlink(missing_ok=True)
        tmp = self.local_path.with_suffix(".toml.tmp")
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(new)
        tmp.replace(self.local_path)
        return {"changed": True, "restart_needed": self.pipeline()["active"]}

    def _validate(self, text: str) -> None:
        """The real loader decides (types, unknown keys): run it on the candidate in a child process."""
        fd, tmp = tempfile.mkstemp(suffix=".toml", prefix="tbot-console-")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            p = subprocess.run([sys.executable, "-c", "from tbot.config import load_config; load_config()"],
                               cwd=self.root, capture_output=True, text=True, timeout=60,
                               env={**os.environ, "TBOT_LOCAL_CONFIG": tmp})
        finally:
            os.unlink(tmp)
        if p.returncode != 0:
            msg = (p.stderr.strip().splitlines() or ["invalid configuration"])[-1]
            raise ApiError(400, msg)

    # ------------------------------------------------------------------ pipeline / helper
    def pipeline(self) -> dict:
        now = time.monotonic()
        t, cached = self._pipe_cache
        if now - t < 1.0 and cached:
            return cached
        p = self.system.pipeline()
        self._pipe_cache = (now, p)
        return p

    def _forget_pipeline(self) -> None:
        self._pipe_cache = (0.0, {})

    def pipeline_telemetry(self, timeout_s: float = 0.6) -> dict | None:
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.pipeline_port, timeout=timeout_s)
            c.request("GET", "/telemetry")
            r = c.getresponse()
            body = r.read()
            c.close()
            return json.loads(body) if r.status == 200 else None
        except (OSError, ValueError, http.client.HTTPException):
            return None

    def start_pipeline(self, mode: str) -> dict:
        if mode not in MODES:
            raise ApiError(400, f"mode must be one of {MODES}")
        self.stop_helper()
        code, out = self.system.start(mode)
        self._forget_pipeline()
        if code != 0:
            raise ApiError(500, out or "start failed")
        return {"output": out}

    def stop_pipeline(self) -> dict:
        code, out = self.system.stop()
        self._forget_pipeline()
        if code != 0:
            raise ApiError(500, out or "stop failed")
        return {"output": out}

    def stop_helper(self) -> None:
        with self.helper_lock:
            h, self.helper = self.helper, None
        if h is not None:
            h.stop()

    def start_helper(self, kind: str, opts: dict) -> Helper:
        if kind not in ("camera", "servo"):
            raise ApiError(400, "kind must be camera or servo")
        if self.pipeline().get("camera_mode"):
            raise ApiError(409, "camera mode (tbotctl camera on) holds the camera: tbotctl camera off first")
        if self.pipeline()["active"]:
            if not opts.get("stop_pipeline"):
                raise ApiError(409, "the pipeline holds the camera and the servo: stop it first")
            self.stop_pipeline()
            time.sleep(1.0)
        self.stop_helper()
        sets: list[str] = []
        if kind == "camera":
            w, h = int(opts.get("width", 1280)), int(opts.get("height", 800))
            sets += [f"camera.width={w}", f"camera.height={h}", f"camera.fps={float(opts.get('fps', 30))}",
                     "camera.process=false"]
            exp = float(opts.get("exposure_us") or 0)
            if exp > 0:
                sets += ["camera.auto_exposure=false", f"camera.exposure_us={exp}",
                         f"camera.analogue_gain={float(opts.get('gain') or 1.0)}"]
            if self.dev:
                sets += [f"camera.source={self.dev_camera}"]
            argv = python_module("tbot.console.camtool", "--port", str(self.cam_port))
        else:
            argv = python_module("tbot.console.servotool") + (["--sim"] if self.dev else [])
        for s in sets:
            argv += ["--set", s]
        helper = Helper(kind, argv, self.root)
        helper.opts = opts  # type: ignore[attr-defined]
        with self.helper_lock:
            self.helper = helper
        return helper

    # ------------------------------------------------------------------ calibration
    def calibration(self) -> dict:
        d = self.root / "calib"
        out: dict[str, Any] = {}
        for name in CALIB_FILES:
            p = d / f"{name}.json"
            try:
                out[name] = {"data": json.loads(p.read_text()), "mtime": p.stat().st_mtime}
            except (OSError, ValueError):
                out[name] = None
        try:
            servo = {k: self.effective(f"actuator.servo.{k}") for k in ("rest_us", "fire_us", "min_us", "max_us",
                                                                         "hold_ms", "break_us", "preload_frac")}
        except KeyError:
            servo = {}
        out["servo"] = servo
        try:
            out["mount"] = {"lens_tip_x_mm": self.effective("mount.lens_tip_x_mm"),
                            "muzzle_x_mm": self.effective("mount.muzzle_pos_mm")[0]}
        except KeyError:
            out["mount"] = None
        hist = d / "history"
        out["history"] = sorted((p.name for p in hist.glob("*.json")), reverse=True)[:30] if hist.is_dir() else []
        return out

    def backup_calib(self, name: str) -> None:
        src = self.root / "calib" / f"{name}.json"
        if src.exists():
            dst = self.root / "calib" / "history" / time.strftime(f"{name}-%Y%m%d-%H%M%S.json")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def boresight_job(self, u: float, v: float, range_m: float, check: bool, note: str = "") -> dict:
        if not (0.5 <= range_m <= 200):
            raise ApiError(400, "range must be 0.5-200 m from the muzzle")
        args = ["calib-boresight", "--pixel", f"{u:.2f},{v:.2f}", "--range-m", f"{range_m:.4f}"]
        if check:
            args.append("--check")
        else:
            self.backup_calib("boresight")
        job = self.jobs.start("boresight", tbot_command(*args),
                              ("Check" if check else "Solve") + f" boresight at {range_m:.2f} m",
                              meta={"pixel": [u, v], "range_m": range_m, "check": check, "note": note})
        return job.describe()

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        p = self.pipeline()
        h = self.helper
        tele = self.pipeline_telemetry() if p["active"] else None
        brief = None
        if tele:
            d = tele.get("decision") or {}
            brief = {k: tele.get(k) for k in ("fps", "armed", "shots", "live", "imu_ready", "exposure_us", "gain",
                                               "n_detections", "age_ms", "lost", "cam_restarts", "frames", "ae")}
            brief["tracks"] = len(tele.get("tracks") or [])
            brief["decision"] = {"reason": d.get("reason"), "p_hit": d.get("p_hit"), "fire": d.get("fire")}
            brief["cpu_ms"] = (tele.get("stage_ms") or {}).get("total")
        root = self.sessions_root()
        try:
            names = sorted((x.name for x in root.iterdir() if x.is_dir()), reverse=True) if root.is_dir() else []
        except OSError:
            names = []
        return {
            "product": "TBOT", "api": API_VERSION, "version": __version__, "hostname": self.hostname,
            "dev": self.dev, "time": time.time(), "pipeline": p, "pipeline_telemetry": brief,
            "health": self.system.health(),
            "helper": {"kind": h.kind, "alive": h.alive} if h is not None else None,
            "jobs": [j.describe(full=False) for j in self.jobs.running()],
            "sessions": {"count": len(names), "latest": names[0] if names else None},
            "autoclips": self.autoclips.describe(),
        }


# =========================================================================================== HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "tbot-console/1"
    protocol_version = "HTTP/1.0"

    @property
    def c(self) -> Console:
        return self.server.console  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        log.debug("%s %s", self.address_string(), fmt % args)

    # -- helpers
    def _send(self, code: int, ctype: str, body: bytes, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, "application/json", json.dumps(_finite(obj), default=str, allow_nan=False).encode())

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            data = json.loads(self.rfile.read(n))
        except ValueError:
            raise ApiError(400, "body is not JSON") from None
        if not isinstance(data, dict):
            raise ApiError(400, "body must be a JSON object")
        return data

    def _route(self, method: str) -> None:
        url = urllib.parse.urlsplit(self.path)
        path = url.path.rstrip("/") or "/"
        query = dict(urllib.parse.parse_qsl(url.query))
        try:
            if path.startswith("/api/"):
                self._api(method, path[5:], query)
            elif method in ("GET", "HEAD"):
                self._static(path)
            else:
                raise ApiError(405, "method not allowed")
        except ApiError as e:
            self._json({"error": str(e)}, e.status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:  # noqa: BLE001 - the server must survive any one request
            log.exception("request failed: %s %s", method, self.path)
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            except OSError:
                pass

    def do_GET(self) -> None:  # noqa: N802
        self._route("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._route("HEAD")

    def do_POST(self) -> None:  # noqa: N802
        self._route("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._route("DELETE")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # -- static
    def _static(self, path: str) -> None:
        rel = "index.html" if path == "/" else path.lstrip("/")
        f = (WEB / rel).resolve()
        if WEB.resolve() not in f.parents or not f.is_file():
            f = WEB / "index.html"  # single-page app: unknown paths get the app
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix in (".js", ".css", ".html", ".svg"):
            ctype += "; charset=utf-8"
        self._send(200, ctype, f.read_bytes())

    # -- api
    def _api(self, method: str, path: str, q: dict) -> None:
        c = self.c
        parts = path.split("/")
        head = parts[0]
        if head == "ping":
            return self._json({"product": "TBOT", "api": API_VERSION, "hostname": c.hostname, "dev": c.dev,
                               "time": time.time()})
        if head == "status":
            return self._json(c.status())
        if head == "telemetry":
            return self._json(c.pipeline_telemetry() or {})
        if head == "stream.mjpg":
            return self._proxy_stream()
        if head == "snapshot.jpg":
            return self._proxy_snapshot()
        if head == "pipeline" and method == "POST":
            b = self._body()
            action = b.get("action")
            if action == "start":
                mode = str(b.get("mode", "dry"))
                if mode == "live" and b.get("confirm") != "LIVE":
                    raise ApiError(400, "type LIVE to start live")
                return self._json(c.start_pipeline(mode))
            if action == "stop":
                return self._json(c.stop_pipeline())
            raise ApiError(400, "action must be start or stop")
        if head == "boot" and method == "POST":
            b = self._body()
            what = str(b.get("mode", ""))
            if what not in (*MODES, "on", "off"):
                raise ApiError(400, "mode must be dry, test, live, on or off")
            if what == "live" and b.get("confirm") != "LIVE":
                raise ApiError(400, "type LIVE to make the gun come up live at power-up")
            code, out = c.system.boot(what)
            c._forget_pipeline()
            if code:
                raise ApiError(500, out)
            return self._json({"output": out})
        if head == "system" and method == "POST":
            b = self._body()
            action = b.get("action")
            if action == "set_time":
                epoch = float(b.get("epoch_ms", 0)) / 1000.0
                if epoch < 1.7e9:
                    raise ApiError(400, "implausible time")
                code, out = c.system.set_time(epoch)
            elif action in ("reboot", "poweroff"):
                c.stop_helper()
                code, out = c.system.power(action)
            else:
                raise ApiError(400, "unknown action")
            if code:
                raise ApiError(500, out)
            return self._json({"output": out})
        if head == "config":
            if method == "GET":
                return self._json({**c.schema(), "local_path": str(c.local_path),
                                   "local_text": c._local_text()})
            if method == "POST":
                b = self._body()
                return self._json(c.write_config(dict(b.get("set") or {}), list(b.get("reset") or [])))
        if head == "calibration":
            return self._json(c.calibration())
        if head == "boresight" and method == "POST":
            b = self._body()
            try:
                u, v, rng = float(b["u"]), float(b["v"]), float(b["range_m"])
            except (KeyError, TypeError, ValueError):
                raise ApiError(400, "need u, v, range_m") from None
            return self._json(c.boresight_job(u, v, rng, bool(b.get("check")), str(b.get("note", ""))))
        if head == "servo" and len(parts) > 1 and parts[1] == "save" and method == "POST":
            return self._json(self._servo_save(self._body()))
        if head == "tool":
            return self._tool(method, parts[1:] if len(parts) > 1 else [""], q)
        if head == "jobs":
            return self._jobs(method, parts[1:], q)
        if head == "sessions":
            return self._sessions(method, parts[1:], q)
        if head == "autoclips":
            if method == "POST":
                c.autoclips.set_enabled(bool(self._body().get("enabled", True)))
            return self._json(c.autoclips.describe())
        if head == "logs":
            n = min(2000, int(q.get("n", 300)))
            f = c.system.log_file()
            lines: list[str] = []
            try:
                with open(f, "rb") as fh:
                    fh.seek(0, 2)
                    size = fh.tell()
                    fh.seek(max(0, size - 400_000))
                    lines = fh.read().decode("utf-8", "replace").splitlines()[-n:]
            except OSError:
                pass
            return self._json({"file": str(f), "lines": lines})
        raise ApiError(404, f"no such endpoint: {method} /api/{path}")

    def _servo_save(self, b: dict) -> dict:
        try:
            vals = {k: float(b[k]) for k in ("rest_us", "fire_us", "min_us", "max_us")}
        except (KeyError, TypeError, ValueError):
            raise ApiError(400, "need rest_us, fire_us, min_us, max_us") from None
        if not 0 < vals["min_us"] < vals["max_us"]:
            raise ApiError(400, "mark both ends of the travel first (lo < hi)")
        for k in ("rest_us", "fire_us"):
            if not vals["min_us"] <= vals[k] <= vals["max_us"]:
                raise ApiError(400, f"{k} {vals[k]:.0f} us is outside the travel {vals['min_us']:.0f}-"
                                    f"{vals['max_us']:.0f} us")
        if vals["rest_us"] == vals["fire_us"]:
            raise ApiError(400, "rest and fire are the same pulse")
        changes = {f"actuator.servo.{k}": v for k, v in vals.items()}
        if b.get("hold_ms") is not None:
            changes["actuator.servo.hold_ms"] = float(b["hold_ms"])
        if b.get("break_us") is not None:
            brk = float(b["break_us"])
            if brk and not min(vals["rest_us"], vals["fire_us"]) < brk <= max(vals["rest_us"], vals["fire_us"]):
                raise ApiError(400, f"break {brk:.0f} us must lie between rest and fire")
            changes["actuator.servo.break_us"] = brk
        if b.get("preload_frac") is not None:
            from tbot.hw.actuator import MAX_PRELOAD_FRAC

            frac = float(b["preload_frac"])
            if not 0.0 <= frac <= MAX_PRELOAD_FRAC:
                raise ApiError(400, f"pre-travel must be 0-{MAX_PRELOAD_FRAC * 100:.0f} % of the way to the break")
            changes["actuator.servo.preload_frac"] = frac
        return self.c.write_config(changes, [])

    def _tool(self, method: str, parts: list[str], q: dict) -> None:
        c = self.c
        what = parts[0]
        if what == "start" and method == "POST":
            b = self._body()
            h = c.start_helper(str(b.get("kind")), b)
            return self._json(h.describe())
        if what == "stop" and method == "POST":
            c.stop_helper()
            return self._json({"ok": True})
        h = c.helper
        if what in ("", "state"):
            return self._json(h.describe() if h else None)
        if h is None:
            raise ApiError(409, "no camera or servo tool is running")
        if what == "command" and method == "POST":
            try:
                h.send(self._body())
            except (RuntimeError, OSError) as e:
                raise ApiError(409, str(e)) from None
            return self._json({"ok": True})
        if what == "events":
            since = int(q.get("since", 0))
            wait = min(10.0, float(q.get("wait", 0)))
            evs = h.events_since(since, wait)
            # stats are frequent: only the newest one is worth sending
            stats = [e for e in evs if e.get("type") == "stats"]
            evs = [e for e in evs if e.get("type") != "stats"] + stats[-1:]
            evs.sort(key=lambda e: e["seq"])
            return self._json({"seq": h.seq, "alive": h.alive, "kind": h.kind, "events": evs})
        raise ApiError(404, "unknown tool endpoint")

    def _jobs(self, method: str, parts: list[str], q: dict) -> None:
        c = self.c
        if not parts or parts[0] == "":
            if method == "GET":
                return self._json(c.jobs.list())
            b = self._body()
            kind = b.get("kind")
            if kind == "doctor":
                j = c.jobs.start("doctor", tbot_command("doctor"), "Configuration & calibration check")
            elif kind == "hwcheck":
                if c.pipeline()["active"]:
                    raise ApiError(409, "the hardware check needs the camera: stop the pipeline first")
                c.stop_helper()
                j = c.jobs.start("hwcheck", tbot_command("hwcheck"), "Hardware check")
            elif kind == "latency_report":
                j = c.jobs.start("latency_report", tbot_command("latency-report"), "Latency budget")
            elif kind == "clip_missing":
                j = c.jobs.start("clip", tbot_command("clip", "--missing", "--root", str(c.sessions_root()),
                                                      "--nice", "15"), "Render missing clips")
            else:
                raise ApiError(400, "unknown job kind")
            return self._json(j.describe())
        try:
            job = c.jobs.get(int(parts[0]))
        except ValueError:
            job = None
        if job is None:
            raise ApiError(404, "no such job")
        if len(parts) > 1 and parts[1] == "cancel" and method == "POST":
            job.cancel()
        return self._json(job.describe())

    def _session_dir(self, name: str) -> Path:
        if not sess.SESSION_NAME.match(name):
            raise ApiError(404, "no such session")
        p = self.c.sessions_root() / name
        if not p.is_dir():
            raise ApiError(404, "no such session")
        return p

    def _sessions(self, method: str, parts: list[str], q: dict) -> None:
        c = self.c
        if not parts or parts[0] == "":
            return self._json({"root": str(c.sessions_root()),
                               "sessions": sess.list_sessions(c.sessions_root(), c.clip_fps())})
        p = self._session_dir(parts[0])
        if len(parts) == 1:
            if method == "DELETE":
                sess.delete(p)
                return self._json({"deleted": p.name})
            return self._json(sess.detail(p, c.clip_fps()))
        action = parts[1]
        if action == "notes" and method == "POST":
            try:
                return self._json(sess.save_notes(p, self._body()))
            except ValueError as e:
                raise ApiError(400, str(e)) from None
        if action == "score" and method == "POST":
            def done(job, p=p):
                text = "\n".join(job.output)
                i = text.find("[")
                if job.code == 0 and i >= 0:
                    try:
                        data = json.loads(text[i:])
                        (p / "score.json").write_text(json.dumps(data, indent=1))
                        job.result = data
                    except ValueError:
                        pass
            j = c.jobs.start("score", tbot_command("score", str(p), "--json", "--image"), f"Score {p.name}",
                             on_done=done, meta={"session": p.name})
            return self._json(j.describe())
        if action == "clip" and method == "POST":
            j = c.jobs.start("clip", tbot_command("clip", str(p), "--nice", "15"), f"Render clip {p.name}",
                             meta={"session": p.name})
            return self._json(j.describe())
        ctype = sess.servable(action)
        if ctype is None or not (p / action).is_file():
            raise ApiError(404, "no such file")
        return self._file(p / action, ctype, download=q.get("download") == "1")

    def _file(self, path: Path, ctype: str, download: bool = False) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", "").strip())
        partial = bool(m and (m.group(1) or m.group(2)))
        if partial:
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        # previews are requested as thumb.jpg?v=<mtime>: a new picture is a new URL, so they may be cached
        self.send_header("Cache-Control", "max-age=604800" if path.name == "thumb.jpg" else "no-cache")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.parent.name}-{path.name}"')
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(65536, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _upstream_port(self) -> int | None:
        h = self.c.helper
        if h is not None and h.alive and h.kind == "camera":
            return self.c.cam_port
        if self.c.pipeline()["active"]:
            return self.c.pipeline_port
        return None

    def _proxy_snapshot(self) -> None:
        port = self._upstream_port()
        if port is None:
            raise ApiError(503, "no camera running")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request("GET", "/snapshot.jpg")
            r = conn.getresponse()
            body = r.read()
        except (OSError, http.client.HTTPException) as e:
            raise ApiError(503, f"camera not answering: {e}") from None
        self._send(r.status, r.getheader("Content-Type") or "image/jpeg", body)

    def _proxy_stream(self) -> None:
        port = self._upstream_port()
        if port is None:
            raise ApiError(503, "no camera running")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("GET", "/stream.mjpg")
            r = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            raise ApiError(503, f"camera not answering: {e}") from None
        self.send_response(200)
        self.send_header("Content-Type", r.getheader("Content-Type") or "multipart/x-mixed-replace")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            pass
        finally:
            conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tbot console", description="TBOT Console: web app + API for the gun")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=80)
    ap.add_argument("--dev", action="store_true", help="laptop stand-in: synthetic pipeline, no power control")
    ap.add_argument("--sessions", default=None, help="sessions directory (default: record.dir)")
    ap.add_argument("--pipeline-port", type=int, default=8080)
    ap.add_argument("--camera-port", type=int, default=8081)
    ap.add_argument("--dev-camera", default="synthetic",
                    help="--dev: camera source for the camera tool (e.g. video:calib_data/.../twist35.mjpeg)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = Console(software_root(), args.dev, args.pipeline_port, args.camera_port, args.sessions,
                      args.dev_camera)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    try:
        httpd = Server((args.host, args.port), Handler)
    except PermissionError:
        log.error("cannot bind port %d (ports below 1024 need CAP_NET_BIND_SERVICE: the systemd unit grants it)",
                  args.port)
        return 1
    httpd.console = console  # type: ignore[attr-defined]
    log.info("TBOT Console on http://%s:%d/ (%s)", args.host, args.port, "dev" if args.dev else "gun")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        console.stop_helper()
        if args.dev:
            console.system.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
