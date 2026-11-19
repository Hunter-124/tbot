"""TBOT Console for Windows: finds the gun on the Ethernet cable and opens the console in its own app window.

    pythonw tbot_console.pyw            (what the desktop shortcut runs; install_shortcut.ps1 makes it)
    python  tbot_console.pyw --pi 127.0.0.1:8765   (talk to a `tbot console --dev` on this computer)

How it works: a tiny web server on 127.0.0.1:47811 serves the console's pages from this checkout
(software/tbot/console/web) and forwards /api/... to the gun. So the window opens at once, shows "connect your
TBOT" until the gun answers, and reconnects by itself after a reboot or a pulled cable. No internet anywhere:
the gun is found at its fixed link-local address 169.254.20.20 (or tbot.local over mDNS). The window is a
Microsoft Edge app window (on every Windows 10/11) with its own profile, so it looks and behaves like an app.
Every press with a clip is also copied to Videos/TBOT/<press> on this computer (Archiver), so the videos are
kept even after the gun's SD card is cleaned. Standard library only.
"""

from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 47811  # fixed, so the app's saved preferences (theme, filters) survive restarts
HERE = Path(__file__).resolve().parent
WEB = HERE.parent / "tbot" / "console" / "web"
CANDIDATES = ["169.254.20.20", "tbot.local", "tbot"]
APPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "TBOT Console"
ARCHIVE = Path(os.environ.get("TBOT_ARCHIVE") or Path.home() / "Videos" / "TBOT")
IDLE_EXIT_S = 180.0  # nobody asked for anything this long (a window polls every 2 s, a minimised one ~1/min)


class Link:
    """Which gun we talk to, kept current by a background prober."""

    def __init__(self, candidates: list[str]):
        self.candidates = candidates
        self.host: str | None = None  # "ip:port"
        self.message = "Searching…"
        self.lock = threading.Lock()
        self.last_request = time.monotonic()
        self.wake = threading.Event()

    @staticmethod
    def _split(c: str) -> tuple[str, int]:
        host, _, port = c.partition(":")
        return host, int(port or 80)

    def probe(self, c: str, timeout: float = 1.5) -> bool:
        host, port = self._split(c)
        try:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
            conn.request("GET", "/api/ping")
            r = conn.getresponse()
            ok = r.status == 200 and json.loads(r.read() or b"{}").get("product") == "TBOT"
            conn.close()
            return ok
        except (OSError, ValueError, http.client.HTTPException):
            return False

    def run(self) -> None:
        while True:
            host = self.host
            if host is not None:
                if not self.probe(host, timeout=3.0) and not self.probe(host, timeout=3.0):
                    with self.lock:
                        self.host = None
                    self.message = f"Lost {host}; searching…"
                self.wake.wait(5.0)
                self.wake.clear()
                continue
            for c in self.candidates:
                self.message = f"Trying {c}…"
                if self.probe(c):
                    with self.lock:
                        self.host = c if ":" in c else f"{c}:80"
                    self.message = f"Connected to {self.host}"
                    break
            else:
                self.message = f"No answer from {', '.join(self.candidates)}. Retrying…"
                time.sleep(1.5)

    def lost(self) -> None:
        self.wake.set()


class Archiver:
    """Copies every press that has a clip from the gun into ARCHIVE/<session>/ (clip, preview, notes, score,
    shot images, summary), so the videos outlive the SD card and `tbotctl clean`. A clip never changes once it
    exists; the small files are fetched again whenever the gun's list says they changed."""

    SMALL = ("thumb.jpg", "preview.json", "summary.json", "console.json", "score.json")

    def __init__(self, link: Link):
        self.link = link
        self.state = "waiting for the gun"
        self.saved: set[str] = set()
        self.index_path = ARCHIVE / ".index.json"
        try:
            self.index: dict[str, str] = json.loads(self.index_path.read_text())
        except (OSError, ValueError):
            self.index = {}
        self.saved = {n for n in self.index if (ARCHIVE / n / "clip.mp4").exists()}

    def _get(self, path: str, dest: Path) -> bool:
        target = self.link.host
        if target is None:
            return False
        host, port = Link._split(target)
        conn = http.client.HTTPConnection(host, port, timeout=60)
        try:
            conn.request("GET", path)
            r = conn.getresponse()
            if r.status != 200:
                r.read()
                return False
            tmp = dest.with_name(dest.name + ".part")
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return True
        finally:
            conn.close()

    def run(self) -> None:
        while True:
            time.sleep(20.0)
            if self.link.host is None:
                self.state = "waiting for the gun"
                continue
            try:
                self._pass()
            except (OSError, ValueError, http.client.HTTPException) as e:
                self.state = f"paused: {e}"

    def _pass(self) -> None:
        target = self.link.host
        if target is None:
            return
        host, port = Link._split(target)
        conn = http.client.HTTPConnection(host, port, timeout=20)
        conn.request("GET", "/api/sessions")
        listing = json.loads(conn.getresponse().read())
        conn.close()
        todo = [s for s in listing.get("sessions", []) if s.get("clip") == "clip.mp4"]
        for i, s in enumerate(todo):
            name = s["name"]
            sig = json.dumps([s.get(k) for k in ("thumb_v", "outcome", "note", "starred", "scored", "shot_images")])
            have_clip = (ARCHIVE / name / "clip.mp4").exists()
            if have_clip and self.index.get(name) == sig:
                continue
            d = ARCHIVE / name
            d.mkdir(parents=True, exist_ok=True)
            q = urllib.parse.quote(name)
            self.state = f"saving {name} ({i + 1}/{len(todo)})"
            if not have_clip and not self._get(f"/api/sessions/{q}/clip.mp4", d / "clip.mp4"):
                continue
            for f in (*self.SMALL, *(s.get("shot_images") or [])):
                self._get(f"/api/sessions/{q}/{f}", d / f)
            self.index[name] = sig
            self.saved.add(name)
            self.index_path.write_text(json.dumps(self.index))
        self.state = f"{len(self.saved)} presses saved"


class Handler(BaseHTTPRequestHandler):
    server_version = "tbot-desktop/1"
    protocol_version = "HTTP/1.0"

    @property
    def link(self) -> Link:
        return self.server.link  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # quiet (pythonw has no console)
        pass

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _any(self) -> None:
        self.link.last_request = time.monotonic()
        path = urllib.parse.urlsplit(self.path).path
        try:
            if path.startswith("/api/"):
                return self._proxy()
            if path == "/local/archive":
                a = self.server.archiver  # type: ignore[attr-defined]
                info = {"dir": str(ARCHIVE), "state": a.state, "saved": sorted(a.saved)}
                return self._send(200, "application/json", json.dumps(info).encode())
            if path == "/local/open" and self.command == "POST":
                q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
                name = q.get("name", "")
                d = ARCHIVE / name if name and "/" not in name and "\\" not in name and ".." not in name else ARCHIVE
                d.mkdir(parents=True, exist_ok=True)
                if hasattr(os, "startfile"):
                    os.startfile(str(d))  # noqa: S606 - opens Explorer on our own folder
                return self._send(200, "application/json", b'{"ok": true}')
            if path == "/local/info":
                info = {"connected": self.link.host is not None, "host": self.link.host,
                        "candidates": self.link.candidates, "message": self.link.message}
                return self._send(200, "application/json", json.dumps(info).encode())
            if self.command not in ("GET", "HEAD"):
                return self._send(405, "text/plain", b"method not allowed")
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            f = (WEB / rel).resolve()
            if WEB.resolve() not in f.parents or not f.is_file():
                f = WEB / "index.html"
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            if f.suffix in (".js", ".css", ".html", ".svg"):
                ctype += "; charset=utf-8"
            self._send(200, ctype, f.read_bytes())
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    do_GET = do_POST = do_DELETE = do_HEAD = _any

    def _proxy(self) -> None:
        target = self.link.host
        if target is None:
            return self._send(502, "application/json", b'{"error": "the gun is not connected"}')
        host, port = Link._split(target)
        body = None
        n = int(self.headers.get("Content-Length") or 0)
        if n > 0:
            body = self.rfile.read(n)
        streaming = self.path.startswith("/api/stream.mjpg")
        try:
            conn = http.client.HTTPConnection(host, port, timeout=60 if streaming else 95)
            headers = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "range")}
            conn.request(self.command, self.path, body=body, headers=headers)
            r = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            self.link.lost()
            return self._send(502, "application/json", json.dumps({"error": f"the gun did not answer: {e}"}).encode())
        self.send_response(r.status)
        for k in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "Content-Disposition",
                  "Cache-Control"):
            v = r.getheader(k)
            if v:
                self.send_header(k, v)
        self.end_headers()
        try:
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                if streaming:
                    self.wfile.flush()
        except (OSError, http.client.HTTPException):
            pass
        finally:
            conn.close()


def find_browser() -> list[str] | None:
    """Edge (on every Windows 10/11), else Chrome: both have an app-window mode."""
    pf = [os.environ.get(k, "") for k in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
    names = [r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"]
    for name in names:
        for root in pf:
            p = Path(root) / name
            if root and p.is_file():
                return [str(p)]
    for exe in ("msedge", "chrome", "chromium", "google-chrome"):
        w = shutil.which(exe)
        if w:
            return [w]
    return None


def open_window(url: str) -> subprocess.Popen | None:
    exe = find_browser()
    if exe is None:
        webbrowser.open(url)
        return None
    profile = APPDATA / "window"
    profile.mkdir(parents=True, exist_ok=True)
    args = exe + [f"--app={url}", f"--user-data-dir={profile}", "--window-size=1440,920", "--no-first-run",
                  "--no-default-browser-check", "--disable-features=Translate,msEdgeTranslate"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(args, creationflags=flags)


def already_running() -> bool:
    try:
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1.0)
        c.request("GET", "/local/info")
        ok = c.getresponse().status == 200
        c.close()
        return ok
    except (OSError, http.client.HTTPException):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="TBOT Console desktop app")
    ap.add_argument("--pi", action="append", default=None,
                    help="address of the gun (host or host:port); repeatable. Default: 169.254.20.20, tbot.local")
    ap.add_argument("--no-window", action="store_true", help="only serve on http://127.0.0.1:47811/")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{PORT}/"
    if already_running():  # a second click on the shortcut: just bring up another window
        if not args.no_window:
            open_window(url)
        return 0
    APPDATA.mkdir(parents=True, exist_ok=True)
    sys.stderr = open(APPDATA / "launcher.log", "a", buffering=1) if sys.stderr is None else sys.stderr
    link = Link(args.pi or CANDIDATES)
    threading.Thread(target=link.run, daemon=True).start()

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = False

    httpd = Server(("127.0.0.1", PORT), Handler)
    httpd.link = link  # type: ignore[attr-defined]
    archiver = Archiver(link)
    httpd.archiver = archiver  # type: ignore[attr-defined]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=archiver.run, daemon=True).start()
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True).start()
    if args.no_window:
        print(f"TBOT Console on {url}  (Ctrl+C to quit)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0
    proc = open_window(url)
    if proc is not None:
        # the window's own Edge profile keeps this process alive until its last window closes
        proc.wait()
        link.last_request = min(link.last_request, time.monotonic() - IDLE_EXIT_S + 15.0)
    # Fallback (no Edge, or the window handed off to an Edge already running this profile): the window polls
    # every 2 s, so a long silence means it is closed. Long enough for a minimised window's throttled timers.
    while time.monotonic() - link.last_request < IDLE_EXIT_S:
        time.sleep(2)
    httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
