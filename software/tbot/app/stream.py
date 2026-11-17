"""Debug web server: MJPEG stream + telemetry JSON for a laptop/phone browser (contract: docs/ARCHITECTURE.md).

The pipeline thread only encodes a JPEG (rate limited to cfg.stream.fps) and swaps it under a lock;
HTTP client threads do all the socket writing, so a slow client can never stall the pipeline.
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger("tbot.stream")

_BOUNDARY = "tbotframe"

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>TBOT debug</title>
<style>
 body{background:#111;color:#ddd;font:13px/1.4 monospace;margin:0;padding:8px}
 #wrap{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}
 img{background:#000;max-width:100%;image-rendering:pixelated;border:1px solid #333}
 pre{background:#181818;border:1px solid #333;padding:8px;margin:0;min-width:320px}
 b{color:#7f7}
</style></head><body>
<div id="wrap">
 <img id="v" src="/stream.mjpg" alt="stream">
 <pre id="t">waiting for telemetry...</pre>
</div>
<p><a href="/clips" style="color:#7af">recorded presses (clips)</a></p>
<script>
async function poll(){
  try{
    const r = await fetch('/telemetry', {cache:'no-store'});
    document.getElementById('t').textContent = JSON.stringify(await r.json(), null, 1);
  }catch(e){ document.getElementById('t').textContent = 'telemetry error: ' + e; }
}
poll(); setInterval(poll, 250);
</script></body></html>
"""


# Files of a session the clips page may serve (never frames.bin: hundreds of MB over a debug link).
_CLIP_FILES = {"clip.mp4": "video/mp4", "clip.avi": "video/x-msvideo", "summary.json": "application/json",
               "telemetry.jsonl": "text/plain; charset=utf-8", "events.jsonl": "text/plain; charset=utf-8",
               "meta.json": "application/json", "imu.csv": "text/plain; charset=utf-8",
               "frames.csv": "text/plain; charset=utf-8"}
_SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_CLIPS_LISTED = 40

_CLIPS_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>TBOT clips</title>
<style>
 body{background:#111;color:#ddd;font:13px/1.5 monospace;margin:0;padding:8px}
 a{color:#7af} video{background:#000;max-width:100%%;border:1px solid #333}
 .s{border-top:1px solid #333;padding:8px 0} .n{color:#7f7;font-weight:bold} .w{color:#fa5}
</style></head><body>
<p><a href="/">&larr; live</a> &nbsp; %d recorded press(es) in %s, newest first. Each clip plays every recorded
frame at %g fps (slow motion); the clock in its corner is real time from the press.</p>
%s
</body></html>
"""


def _clip_summary(session: Path) -> str:
    """One line about a session from its summary.json."""
    try:
        d = json.loads((session / "summary.json").read_text())
    except (OSError, ValueError):
        return ""
    bits = []
    if d.get("span_s") is not None:
        bits.append(f"{d['span_s']:.1f} s")
    if d.get("frames") is not None:
        bits.append(f"{d['frames']} frames")
    if d.get("fps"):
        bits.append(f"{d['fps']:.0f} fps")
    if d.get("frames_with_detections") is not None:
        bits.append(f"detections on {d['frames_with_detections']}")
    if d.get("frames_with_confirmed_track") is not None:
        bits.append(f"confirmed track on {d['frames_with_confirmed_track']}")
    if d.get("max_p_hit") is not None:
        bits.append(f"max P(hit) {d['max_p_hit']:.2f}")
    fires = (d.get("events") or {}).get("fire", 0)
    bits.append(f"{fires} shot(s)" + ("" if d.get("live") else " [dry run]"))
    reasons = d.get("decision_reasons") or {}
    if reasons:
        bits.append("mostly: " + next(iter(reasons)))
    if d.get("evicted_frames"):
        bits.append(f"{d['evicted_frames']} early frames lost (press longer than the RAM ring)")
    return ", ".join(bits)


def jsonable(obj: Any) -> Any:
    """numpy/NaN-safe conversion for json.dumps (NaN and Inf become null: strict JSON)."""
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


class _Handler(BaseHTTPRequestHandler):
    server_version = "tbot-debug/1"
    protocol_version = "HTTP/1.0"  # one response per connection: no keep-alive bookkeeping

    @property
    def debug(self) -> "DebugServer":
        return self.server.debug_server  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:  # quiet: the pipeline owns stdout
        log.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if path == "/":
                self._send(200, "text/html; charset=utf-8", _PAGE.encode("utf-8"))
            elif path == "/telemetry":
                body = json.dumps(jsonable(self.debug.telemetry())).encode("utf-8")
                self._send(200, "application/json", body)
            elif path == "/clips":
                self._send(200, "text/html; charset=utf-8", self.debug.clips_page().encode("utf-8"))
            elif path.startswith("/clips/"):
                self._clip_file(path[len("/clips/"):])
            elif path == "/snapshot.jpg":
                self.debug.touch()
                if self.debug.renders_on_demand:  # nothing is drawn while nobody watches: wait for a fresh frame
                    self.debug.wait_frame(self.debug.snapshot()[1], timeout_s=1.5)
                jpg, _ = self.debug.snapshot()
                if jpg is None:
                    self._send(503, "text/plain; charset=utf-8", b"no frame published yet\n")
                else:
                    self._send(200, "image/jpeg", jpg)
            elif path == "/stream.mjpg":
                self._stream()
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _clip_file(self, rest: str) -> None:
        """/clips/<session>/<file>: one whitelisted file of a session, with byte ranges (video seeking)."""
        root = self.debug.clips_dir
        parts = rest.split("/")
        if root is None or len(parts) != 2 or not _SESSION_NAME.match(parts[0]) or parts[1] not in _CLIP_FILES:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")
            return
        path = Path(root) / parts[0] / parts[1]
        try:
            size = path.stat().st_size
            f = open(path, "rb")
        except OSError:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")
            return
        with f:
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", "").strip())
            partial = bool(m and (m.group(1) or m.group(2)))
            if partial:
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:  # suffix range: the last N bytes
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", _CLIP_FILES[parts[1]])
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(65536, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        self.debug.add_viewer(1)
        try:
            while not self.debug.closed:
                jpg, seq = self.debug.wait_frame(last, timeout_s=0.5)
                if jpg is None or seq == last:
                    continue
                last = seq
                head = (
                    f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpg)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(head)
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        finally:
            self.debug.add_viewer(-1)


class DebugServer:
    """ThreadingHTTPServer on cfg.stream.host:port serving /, /stream.mjpg, /snapshot.jpg, /telemetry."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.host = str(cfg.stream.host)
        self.port = int(cfg.stream.port)  # replaced by the bound port after start() (port 0 -> ephemeral)
        self.quality = int(cfg.stream.jpeg_quality)
        fps = float(cfg.stream.fps)
        self._period_ns = int(1e9 / fps) if fps > 0 else 0
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._telemetry: dict = {}
        self._last_pub_ns = -(1 << 62)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.closed = False
        self.published = 0
        self.skipped = 0
        self.clips_dir: Path | None = None  # sessions directory behind /clips (tbot run sets it)
        self.renders_on_demand = False      # True once submit() is in use: frames exist only while watched
        self._viewers = 0
        self._touched_ns = -(1 << 62)
        self._job: tuple | None = None
        self._job_cond = threading.Condition()
        self._render_thread: threading.Thread | None = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> "DebugServer":
        if self._httpd is not None:
            return self
        httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        httpd.daemon_threads = True
        httpd.debug_server = self  # type: ignore[attr-defined]
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        self.closed = False
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.2}, name="tbot-stream", daemon=True
        )
        self._thread.start()
        log.info("debug stream on http://%s:%d/", self.host, self.port)
        return self

    def stop(self) -> None:
        self.closed = True
        with self._cond:
            self._cond.notify_all()
        with self._job_cond:
            self._job_cond.notify_all()
        thread, self._render_thread = self._render_thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def __enter__(self) -> "DebugServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---------------------------------------------------------------- publishing
    def due(self, now_ns: int | None = None) -> bool:
        """True when a new frame may be encoded (cfg.stream.fps rate limit)."""
        if self.closed:
            return False
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        return now - self._last_pub_ns >= self._period_ns

    def publish(self, bgr: np.ndarray, telemetry: dict | None = None) -> bool:
        """Encode and offer one frame; returns False when the rate limit skipped it."""
        now = time.monotonic_ns()
        if not self.due(now):
            self.skipped += 1
            return False
        import cv2

        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)])
        if not ok:
            log.warning("JPEG encoding failed for a %s frame", getattr(bgr, "shape", "?"))
            return False
        self._last_pub_ns = now
        data = buf.tobytes()
        with self._cond:
            self._jpeg = data
            self._seq += 1
            if telemetry is not None:
                self._telemetry = telemetry
            self._cond.notify_all()
        self.published += 1
        return True

    def set_telemetry(self, telemetry: dict) -> None:
        with self._cond:
            self._telemetry = telemetry

    # ---------------------------------------------------------------- on-demand rendering
    WATCH_LINGER_S = 3.0  # a snapshot poll counts as watching for this long

    def add_viewer(self, delta: int) -> None:
        with self._cond:
            self._viewers = max(0, self._viewers + int(delta))

    def touch(self) -> None:
        self._touched_ns = time.monotonic_ns()

    def watched(self) -> bool:
        """Is anybody looking at the picture (an open MJPEG stream, or a snapshot in the last few seconds)?"""
        return self._viewers > 0 or time.monotonic_ns() - self._touched_ns < self.WATCH_LINGER_S * 1e9

    def submit(self, image: np.ndarray, telemetry: dict, render: Callable[[np.ndarray, dict], np.ndarray]) -> None:
        """Hand one frame to the render thread: render(image, telemetry) -> BGR is drawn and JPEG-encoded there
        and published. Only the newest job is kept, so a slow encode never queues work or blocks the caller."""
        self.renders_on_demand = True
        self._last_pub_ns = time.monotonic_ns()  # the rate limit counts submissions
        with self._job_cond:
            self._job = (image, telemetry, render)
            if self._render_thread is None and not self.closed:
                self._render_thread = threading.Thread(target=self._render_loop, name="tbot-stream-render",
                                                       daemon=True)
                self._render_thread.start()
            self._job_cond.notify()

    def _render_loop(self) -> None:
        import cv2

        while not self.closed:
            with self._job_cond:
                while self._job is None and not self.closed:
                    self._job_cond.wait(0.5)
                job, self._job = self._job, None
            if job is None:
                continue
            image, telemetry, render = job
            try:
                bgr = render(image, telemetry)
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.quality)])
            except Exception as exc:  # noqa: BLE001 - a drawing error must not end the stream
                log.warning("overlay render failed: %s", exc)
                continue
            if not ok:
                continue
            with self._cond:
                self._jpeg = buf.tobytes()
                self._seq += 1
                self._cond.notify_all()
            self.published += 1

    # ---------------------------------------------------------------- recorded presses
    def clips_page(self) -> str:
        root = self.clips_dir
        sessions: list[Path] = []
        if root is not None and Path(root).is_dir():
            sessions = sorted((d for d in Path(root).iterdir() if d.is_dir() and _SESSION_NAME.match(d.name)),
                              key=lambda d: d.stat().st_mtime, reverse=True)
        rows = []
        for i, d in enumerate(sessions[:_CLIPS_LISTED]):
            name = html.escape(d.name)
            clip = next((c for c in ("clip.mp4", "clip.avi") if (d / c).exists()), None)
            links = " ".join(f'<a href="/clips/{name}/{f}">{f}</a>' for f in
                             ("summary.json", "telemetry.jsonl", "events.jsonl", "meta.json") if (d / f).exists())
            if clip is None:
                body = '<span class="w">no clip yet (still rendering, or run: tbot clip --missing)</span>'
            elif i < 3 and clip.endswith(".mp4"):
                body = f'<video controls preload="metadata" src="/clips/{name}/{clip}"></video><br>' \
                       f'<a href="/clips/{name}/{clip}">{clip}</a>'
            else:
                body = f'<a href="/clips/{name}/{clip}">{clip}</a>'
            rows.append(f'<div class="s"><span class="n">{name}</span> &nbsp; {html.escape(_clip_summary(d))}<br>'
                        f'{body} &nbsp; {links}</div>')
        fps = float(getattr(getattr(self.cfg, "record", None), "clip_fps", 30.0) or 30.0)
        return _CLIPS_PAGE % (len(sessions), html.escape(str(root)), fps, "\n".join(rows) or "<p>none yet</p>")

    # ---------------------------------------------------------------- readers
    def snapshot(self) -> tuple[bytes | None, int]:
        with self._cond:
            return self._jpeg, self._seq

    def telemetry(self) -> dict:
        with self._cond:
            return dict(self._telemetry)

    def wait_frame(self, last_seq: int, timeout_s: float = 0.5) -> tuple[bytes | None, int]:
        """Block until a frame newer than last_seq is published (or the timeout expires)."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout_s)
            return self._jpeg, self._seq
