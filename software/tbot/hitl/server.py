"""HTTP server for the HITL display and frame timing protocol."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from tbot.app.stream import jsonable
from tbot.hitl.page import PAGE_HTML

log = logging.getLogger(__name__)


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], store: Any) -> None:
        self.store = store
        super().__init__(address, _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        log.debug("HITL request failed from %s", client_address, exc_info=True)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        log.debug("%s - %s", self.address_string(), format % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        self._send(status, json.dumps(jsonable(value), allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:
        rig_time = self.server.store.rig_ms()
        try:
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/hitl/sync":
                self._json(200, {"t": rig_time})
            elif path == "/hitl/scene":
                self._json(200, self.server.store.program())
            elif path == "/hitl/state":
                self._json(200, self.server.store.rig_state())
            else:
                self._json(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("negative content length")
            body = self.rfile.read(length)
            if urlsplit(self.path).path != "/hitl/log":
                self._json(404, {"error": "not found"})
                return
            payload = json.loads(body)
            self.server.store.post_log(payload, time.monotonic_ns())
            self._json(200, {})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, UnicodeDecodeError):
            self.close_connection = True
            try:
                self._json(400, {"error": "invalid request body"})
            except (BrokenPipeError, ConnectionResetError):
                pass


def _advertised_host(host: str) -> str:
    if host != "0.0.0.0":
        return host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
        if address and not address.startswith("127.") and address != "0.0.0.0":
            return address
    except OSError:
        pass
    return socket.gethostname()


class HitlServer:
    """Serve a store to a browser on the rig's display."""

    def __init__(self, store: Any, host: str = "0.0.0.0", port: int = 8090) -> None:
        self.store = store
        self.host = host
        self.port = port
        self._server: _HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> HitlServer:
        if self._server is not None:
            return self
        server = _HTTPServer((self.host, self.port), self.store)
        self.port = server.server_address[1]
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="hitl-http", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join()
        self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        return f"http://{_advertised_host(self.host)}:{self.port}/"

    def __enter__(self) -> HitlServer:
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()
