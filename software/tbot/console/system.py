"""The gun's system side for the console: pipeline service, modes, health, clock, power.

PiSystem drives the real thing through scripts/tbotctl and systemd (the pi user has passwordless sudo).
DevSystem stands in on a laptop: the "pipeline" is `tbot run --source synthetic` in a child process, power
actions are refused, so the whole UI can be exercised with no gun attached (`tbot console --dev`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from tbot.hw.health import read_health

MODES = ("dry", "test", "live")


def software_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tbot_command(*args: str) -> list[str]:
    """argv that runs `tbot <args>` with this interpreter (the venv on the Pi)."""
    return [sys.executable, "-u", "-c", "import sys; from tbot.cli import main; sys.exit(main())", *args]


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _meminfo() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                out[k] = int(v.split()[0]) * 1024
    except OSError:
        return {}
    return out


class _Base:
    dev = False

    def __init__(self, root: Path):
        self.root = root
        self._health_cache: tuple[float, dict] = (0.0, {})

    def common_health(self) -> dict:
        now = time.monotonic()
        t, cached = self._health_cache
        if now - t < 2.0 and cached:
            return cached
        h = read_health()
        du = shutil.disk_usage(self.root)
        mem = _meminfo()
        up = _read("/proc/uptime")
        try:
            load = os.getloadavg()
        except (AttributeError, OSError):
            load = None
        out = {
            "cpu_temp_c": h.get("cpu_temp_c"),
            "throttled": h.get("throttled"),
            "arm_freq_mhz": h.get("arm_freq_mhz"),
            "disk_total": du.total, "disk_free": du.free,
            "mem_total": mem.get("MemTotal"), "mem_available": mem.get("MemAvailable"),
            "uptime_s": float(up.split()[0]) if up else None,
            "load": list(load) if load else None,
            "cpus": os.cpu_count(),
        }
        self._health_cache = (now, out)
        return out


class PiSystem(_Base):
    """The Pi: tbot-run.service, /run/tbot/mode, /etc/tbot/boot-mode (scripts/tbotctl is the reference)."""

    unit = "tbot-run.service"

    def __init__(self, root: Path):
        super().__init__(root)
        self.tbotctl = str(root / "scripts" / "tbotctl")

    def _run(self, *argv: str, timeout: float = 30.0, env: dict | None = None) -> tuple[int, str]:
        e = dict(os.environ)
        e.update(env or {})
        try:
            p = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, env=e)
        except subprocess.TimeoutExpired:
            return 124, "timed out"
        return p.returncode, (p.stdout + p.stderr).strip()

    def pipeline(self) -> dict:
        _, active = self._run("systemctl", "is-active", self.unit, timeout=5)
        _, enabled = self._run("systemctl", "is-enabled", self.unit, timeout=5)
        boot_mode = (_read("/etc/tbot/boot-mode") or "dry").strip()
        run_mode = _read("/run/tbot/mode")
        mode = (run_mode or boot_mode).strip()
        _, cam = self._run("systemctl", "is-active", "tbot-camera.service", timeout=5)
        active = active.splitlines()[0] if active else "unknown"
        return {"state": active, "active": active in ("active", "activating", "reloading"),
                "camera_mode": cam.strip().startswith("active"),
                "mode": mode if mode in MODES else "dry", "boot_mode": boot_mode if boot_mode in MODES else "dry",
                "boot_enabled": enabled.startswith("enabled")}

    def start(self, mode: str) -> tuple[int, str]:
        return self._run(self.tbotctl, "start", mode, env={"TBOT_YES": "1"}, timeout=60)

    def stop(self) -> tuple[int, str]:
        return self._run(self.tbotctl, "stop", timeout=30)

    def boot(self, what: str) -> tuple[int, str]:
        return self._run(self.tbotctl, "boot", what, env={"TBOT_YES": "1"}, timeout=30)

    def set_time(self, epoch_s: float) -> tuple[int, str]:
        return self._run("sudo", "-n", "date", "-s", f"@{epoch_s:.3f}", timeout=10)

    def power(self, action: str) -> tuple[int, str]:
        if action not in ("reboot", "poweroff"):
            return 2, "unknown action"
        # detached, so the HTTP reply gets out before the network goes down
        subprocess.Popen(["sh", "-c", f"sleep 1; sudo -n systemctl {action}"], start_new_session=True)
        return 0, f"{action} in 1 s"

    def health(self) -> dict:
        return self.common_health()

    def log_file(self) -> Path:
        return self.root / "logs" / "tbot-run.log"


class DevSystem(_Base):
    """A laptop pretending to be the gun: synthetic camera, simulated GPIO, no power control."""

    dev = True
    SCENES = ("skeet_crossing", "skeet_incoming", "trap_straightaway")

    def __init__(self, root: Path, port: int = 8080):
        super().__init__(root)
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.mode = "dry"
        self.boot_mode = "dry"
        self.boot_enabled = True
        self.want = False
        self.seed = 0
        self.log_path = root / "logs" / "console-dev-pipeline.log"

    def pipeline(self) -> dict:
        return {"state": "active" if self.want else "inactive", "active": self.want, "mode": self.mode,
                "boot_mode": self.boot_mode, "boot_enabled": self.boot_enabled}

    def _spawn(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        self.seed += 1
        scene = self.SCENES[self.seed % len(self.SCENES)]
        args = ["run", "--source", f"synthetic:{scene}", "--stream", "--port", str(self.port), "--record", "off",
                "--set", "gpio.backend=sim", "--set", "actuator.type=sim", "--set", f"sim.seed={self.seed}",
                "--sim-armed", "--quiet"]
        self.proc = subprocess.Popen(tbot_command(*args), cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "TBOT_LOG_FILE": ""})

    def _supervise(self) -> None:
        """systemd's Restart=always: the synthetic scenes end after a few seconds."""
        while self.want:
            p = self.proc
            if p is None or p.poll() is not None:
                self._spawn()
            time.sleep(0.5)

    def start(self, mode: str) -> tuple[int, str]:
        self.stop()
        self.mode = mode
        self.want = True
        threading.Thread(target=self._supervise, daemon=True).start()
        return 0, f"started synthetic pipeline in {mode} (dev)"

    def stop(self) -> tuple[int, str]:
        self.want = False
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(8)
            except subprocess.TimeoutExpired:
                p.kill()
        return 0, "stopped"

    def boot(self, what: str) -> tuple[int, str]:
        if what in MODES:
            self.boot_mode = what
        elif what in ("on", "off"):
            self.boot_enabled = what == "on"
        return 0, f"boot {what} (dev)"

    def set_time(self, epoch_s: float) -> tuple[int, str]:
        return 0, "dev: the laptop clock is left alone"

    def power(self, action: str) -> tuple[int, str]:
        return 1, f"dev mode: {action} is not done on a laptop"

    def health(self) -> dict:
        h = self.common_health()
        if h.get("cpu_temp_c") is None:
            h = dict(h, cpu_temp_c=48.0)
        return h

    def log_file(self) -> Path:
        return self.log_path
