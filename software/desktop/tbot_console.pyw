"""TBOT Console for Windows: finds the gun on the Ethernet cable and opens the console in its own app window.

    pythonw tbot_console.pyw            (what the desktop shortcut runs; install_shortcut.ps1 makes it)
    python  tbot_console.pyw --pi 127.0.0.1:8765   (talk to a `tbot console --dev` on this computer)

How it works: a tiny web server on 127.0.0.1:47811 serves the console's pages from this checkout
(software/tbot/console/web) and forwards /api/... to the gun. So the window opens at once, shows "connect your
TBOT" until the gun answers, and reconnects by itself after a reboot or a pulled cable. No internet anywhere:
the gun is found at its fixed link-local address 169.254.20.20 (or tbot.local over mDNS). The window is a
Microsoft Edge app window (on every Windows 10/11) with its own profile, so it looks and behaves like an app.
Every press is also copied to Videos/TBOT/<press> on this computer (Archiver), raw data included, and its clip
and preview are RENDERED HERE (a second on this computer, 10-20 s of a gun core on the Pi - which no longer renders
while its pipeline runs). The console shows those local clips as if the gun had made them. The proxy and the
archive use the standard library only; rendering runs this checkout's `tbot clip` / `tbot.console.thumbs`
(numpy + OpenCV).
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


SOFTWARE = HERE.parent           # this checkout's software/ (tbot clip, tbot.console.thumbs)
RAW = ("summary.json", "meta.json", "telemetry.jsonl", "frames.csv", "events.jsonl", "imu.csv", "frames.bin")
LOCAL_FILES = ("clip.mp4", "thumb.jpg", "preview.json")   # served from the archive when the gun has none


def _python() -> str:
    """python.exe next to pythonw.exe (no console window either way: CREATE_NO_WINDOW)."""
    exe = Path(sys.executable)
    alt = exe.with_name("python.exe")
    return str(alt if exe.name.lower() == "pythonw.exe" and alt.exists() else exe)


def _run_tool(args: list[str], timeout: float = 600.0) -> bool:
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    try:
        r = subprocess.run([_python(), *args], cwd=str(SOFTWARE), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=timeout, creationflags=flags)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def local_file(name: str, fname: str) -> Path | None:
    """A clip / preview this computer made (or saved) for press `name`, if it has one."""
    if not name or "/" in name or "\\" in name or ".." in name or fname not in LOCAL_FILES:
        return None
    f = ARCHIVE / name / fname
    return f if f.is_file() else None


def patch_session(row: dict) -> dict:
    """A session as the gun lists it, with what this computer has added (its own clip and preview)."""
    name = str(row.get("name") or "")
    if not row.get("clip") and local_file(name, "clip.mp4") is not None:
        row["clip"] = "clip.mp4"
    thumb = local_file(name, "thumb.jpg")
    if not row.get("thumb") and thumb is not None:
        row["thumb"] = True
        row["thumb_v"] = int(thumb.stat().st_mtime)
        try:
            prev = json.loads((ARCHIVE / name / "preview.json").read_text())
            row["key_t"], row["key_reason"] = prev.get("clip_t"), prev.get("reason")
        except (OSError, ValueError):
            pass
    return row


class Archiver:
    """Copies every finished press from the gun into ARCHIVE/<session>/ - the raw data (frames, IMU, telemetry), the
    notes, score and shot images - and renders its clip and preview here, so the gun never spends a core on
    video and the presses outlive the SD card and `tbotctl clean`. A clip never changes once it exists; the small
    files are fetched again whenever the gun's list says they changed."""

    SMALL = ("summary.json", "console.json", "score.json")

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
        # newest first: the press just shot is the one the shooter wants to see
        todo = [s for s in listing.get("sessions", []) if s.get("clip") == "clip.mp4" or s.get("renderable")]
        for i, s in enumerate(todo):
            name = s["name"]
            sig = json.dumps([s.get(k) for k in ("outcome", "note", "starred", "scored", "shot_images")])
            d = ARCHIVE / name
            have_clip = (d / "clip.mp4").exists()
            if have_clip and (d / "thumb.jpg").exists() and self.index.get(name) == sig:
                continue
            d.mkdir(parents=True, exist_ok=True)
            q = urllib.parse.quote(name)
            self.state = f"saving {name} ({i + 1}/{len(todo)})"
            if not have_clip and s.get("clip") == "clip.mp4":
                have_clip = self._get(f"/api/sessions/{q}/clip.mp4", d / "clip.mp4")
            if not (have_clip and (d / "thumb.jpg").exists()):
                # render here: fetch the raw press (frames.bin last: it is the big one), then clip + preview
                if not all((d / f).exists() or self._get(f"/api/sessions/{q}/{f}", d / f) for f in RAW):
                    continue
                self.state = f"rendering {name} ({i + 1}/{len(todo)})"
                if not have_clip:
                    have_clip = _run_tool(["-m", "tbot.cli", "clip", str(d)]) and (d / "clip.mp4").exists()
                if not (d / "thumb.jpg").exists():
                    _run_tool(["-m", "tbot.console.thumbs", str(d)], timeout=120.0)
                if not have_clip:
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

    def _local(self) -> bool:
        """Serve /api/sessions/<name>/<clip|thumb|preview> from this computer when it made one. True = done."""
        if self.command not in ("GET", "HEAD"):
            return False
        parts = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).split("/")
        if len(parts) != 5 or parts[2] != "sessions":
            return False
        f = local_file(parts[3], parts[4])
        if f is None:
            return False
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        size = f.stat().st_size
        start, end = 0, size - 1
        rng = (self.headers.get("Range") or "").strip()
        partial = rng.startswith("bytes=") and rng[6:].split("-")[0].isdigit()
        if partial:
            a, _, z = rng[6:].partition("-")
            start = int(a)
            end = min(int(z), size - 1) if z.isdigit() else size - 1
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return True
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "GET":
            with open(f, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        return True

    def _proxy(self) -> None:
        if self._local():
            return None
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
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        if self.command == "GET" and r.status == 200 and path.startswith("/api/sessions") and path.count("/") <= 3:
            # the list (/api/sessions) or one press (/api/sessions/<name>): add the clips and previews made here
            try:
                data = json.loads(r.read())
                if isinstance(data.get("sessions"), list):
                    data["sessions"] = [patch_session(x) for x in data["sessions"]]
                elif data.get("name"):
                    data = patch_session(data)
                body = json.dumps(data).encode()
            except (ValueError, OSError, http.client.HTTPException, AttributeError):
                body = None
            conn.close()
            if body is None:
                return self._send(502, "application/json", b'{"error": "bad answer from the gun"}')
            return self._send(200, "application/json", body)
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
