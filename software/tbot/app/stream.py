"""Debug web server: MJPEG stream + telemetry JSON for a laptop/phone browser (contract: docs/ARCHITECTURE.md).

The pipeline thread only encodes a JPEG (rate limited to cfg.stream.fps) and swaps it under a lock;
HTTP client threads do all the socket writing, so a slow client can never stall the pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

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
            elif path == "/snapshot.jpg":
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

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
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
