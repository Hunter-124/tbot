"""Config schema from config/default.toml (values + the comments as docs) and comment-preserving edits of
config/local.toml.

`tbot.config.save_local_overrides` rewrites local.toml from scratch and drops every hand-written comment; the
console edits the file line by line instead, so the notes on the Pi's local.toml (why a value is what it is)
survive. The result is still validated by the real loader (tbot.config.load_config) before it replaces the file.
"""

from __future__ import annotations

import re
import tomllib
from typing import Any

_HEADER = re.compile(r"^\s*\[([A-Za-z0-9_.]+)\]\s*(?:#(.*))?$")
_KEY = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$")

#: Section titles and one-line descriptions for the UI (anything not listed gets its key as the title).
SECTION_INFO: dict[str, tuple[str, str]] = {
    "camera": ("Camera", "Sensor mode, frame rate and the auto-exposure rules."),
    "optics": ("Optics", "Lens model prior (calibration/intrinsics.json overrides it)."),
    "mount": ("Mount geometry", "Where the camera and muzzle sit in the gun frame (mm)."),
    "detector": ("Clay detector", "How clays are found against the sky. Thresholds are noise-relative."),
    "tracker": ("Tracker", "Kalman tracking of targets in world coordinates."),
    "imu": ("IMU", "MPU-6050 wiring, rates and axis mapping."),
    "attitude": ("Attitude filter", "Complementary filter and gyro auto-zero."),
    "gun": ("Gun", "Lock and barrel times of the shotgun."),
    "shot": ("Shot & load", "Muzzle velocity, pellets and pattern size."),
    "fire": ("Fire solver", "When to pull: hit-probability bars, range limits, safety windows."),
    "arming": ("Arming button", "The hold-to-arm switch (software interlock)."),
    "actuator": ("Trigger actuator", "Which actuator pulls the trigger."),
    "actuator.servo": ("Trigger servo", "MG996R pulse widths and timing (set these on the Trigger page)."),
    "actuator.pq12": ("PQ12 actuator", "Linear actuator fallback."),
    "actuator.sim": ("Simulated actuator", "Timing model for simulations."),
    "gpio": ("GPIO", "Pin backend and bench pins."),
    "latency": ("Latency budget", "Defaults until calibration/latency.json has measurements."),
    "stream": ("Debug stream", "The pipeline's MJPEG stream on :8080."),
    "record": ("Recording", "What each press of the arming button saves."),
    "paths": ("Paths", "Where calibration files live."),
    "sim": ("Simulator", "Synthetic scenes for tbot sim."),
    "calrig": ("CalRig", "Motorised yaw table (bench only)."),
}

#: Keys shown first under "Common" in the settings page.
COMMON_KEYS = [
    "camera.fps", "camera.idle_fps", "camera.auto_exposure", "camera.exposure_us", "camera.analogue_gain",
    "shot.muzzle_velocity_mps", "shot.pattern_radius_ref_m", "shot.pattern_ref_range_m",
    "fire.min_hit_probability", "fire.min_range_m", "fire.max_range_m", "fire.max_shots_per_arm",
    "record.mode", "record.preroll_s", "record.postroll_s", "record.clip", "record.min_free_mb",
    "actuator.servo.hold_ms", "actuator.servo.preload_frac",
]

_UNITS = [
    ("_mps3", "m/s³"), ("_dps2", "°/s²"), ("_mps", "m/s"), ("_dps", "°/s"), ("_deg_s", "°/s"), ("_us", "µs"),
    ("_ms", "ms"), ("_ns", "ns"), ("_mm", "mm"), ("_um", "µm"), ("_cm", "cm"), ("_m", "m"), ("_s", "s"),
    ("_deg", "°"), ("_px", "px"), ("_hz", "Hz"), ("_mb", "MB"), ("_dn", "DN"), ("_pct", "%"), ("_g", "g"),
    ("_kg_m3", "kg/m³"), ("_c", "°C"),
]


def split_value_comment(rest: str) -> tuple[str, str]:
    """'1.0   # note' -> ('1.0', '# note'); a '#' inside a string is not a comment."""
    quote = None
    i = 0
    while i < len(rest):
        ch = rest[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
        elif ch == "#":
            return rest[:i].rstrip(), rest[i:]
        i += 1
    return rest.rstrip(), ""


def toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        r = repr(v)
        return r if ("." in r or "e" in r or "n" in r) else r + ".0"
    if isinstance(v, str):
        import json

        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def _unit(key: str) -> str:
    k = key.lower()
    for suffix, unit in _UNITS:
        if k.endswith(suffix):
            return unit
    return ""


def _choices(doc: str) -> list[str] | None:
    """'picam | cv:<index> | ...' or 'off | armed (every ...) | continuous' at the start of a comment."""
    first = doc.split("\n", 1)[0]
    if "|" not in first:
        return None
    opts = []
    for part in first.split("|"):
        word = part.strip().split(" ")[0].split("(")[0].strip(" .,:;")
        if not word or not re.match(r"^[A-Za-z0-9_\-]+$", word):
            return None
        opts.append(word)
    return opts if len(opts) >= 2 else None


def _kind(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    return "table"


def _lookup(d: dict, dotted: str) -> tuple[bool, Any]:
    node: Any = d
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return False, None
        node = node[p]
    return True, node


def parse_schema(default_text: str, local_text: str = "") -> dict:
    """Sections and keys of default.toml with docs, defaults, the local override and the effective value."""
    defaults = tomllib.loads(default_text)
    local = tomllib.loads(local_text) if local_text.strip() else {}
    sections: list[dict] = []
    by_name: dict[str, dict] = {}
    section = ""
    last_key: dict | None = None
    for line in default_text.splitlines():
        m = _HEADER.match(line)
        if m:
            section = m.group(1)
            title, desc = SECTION_INFO.get(section, (section, ""))
            note = (m.group(2) or "").strip()
            sec = {"name": section, "title": title, "description": desc, "note": note, "keys": []}
            sections.append(sec)
            by_name[section] = sec
            last_key = None
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            # an indented comment line continues the doc of the key above it
            if last_key is not None and line[:1].isspace():
                last_key["doc"] += ("\n" if last_key["doc"] else "") + stripped.lstrip("#").strip()
            continue
        km = _KEY.match(line)
        if not km or not section:
            last_key = None
            continue
        name = km.group(1)
        _, comment = split_value_comment(km.group(2))
        dotted = f"{section}.{name}"
        ok, default = _lookup(defaults, dotted)
        if not ok or isinstance(default, dict):
            continue
        has_local, local_v = _lookup(local, dotted)
        key = {
            "key": dotted, "name": name, "type": _kind(default), "default": default,
            "local": local_v if has_local else None, "overridden": bool(has_local),
            "value": local_v if has_local else default,
            "doc": comment.lstrip("#").strip(), "unit": _unit(name),
        }
        by_name[section]["keys"].append(key)
        last_key = key
    for sec in sections:
        for key in sec["keys"]:
            key["choices"] = _choices(key["doc"]) if key["type"] == "str" else None
            key["inference"] = "[INFERENCE" in key["doc"]
    return {"sections": [s for s in sections if s["keys"]], "common": COMMON_KEYS}


def coerce(value: Any, default: Any, key: str) -> Any:
    """A value from the UI into the type of the default (int stays int, float accepts int, ...)."""
    kind = _kind(default)
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError(f"{key} must be true or false")
    if kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a whole number")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip(), 0)
            except ValueError:
                pass
        raise ValueError(f"{key} must be a whole number")
    if kind == "float":
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a number")
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number") from None
        if out != out or out in (float("inf"), float("-inf")):
            raise ValueError(f"{key} must be a finite number")
        return out
    if kind == "str":
        return str(value)
    if kind == "list":
        if isinstance(value, str):
            value = tomllib.loads(f"v = {value}")["v"]
        if not isinstance(value, list):
            raise ValueError(f"{key} must be a list")
        return value
    raise ValueError(f"{key}: unsupported type")


def patch_text(text: str, changes: dict[str, Any], removals: set[str] | frozenset = frozenset()) -> str:
    """Set dotted keys to values (and delete `removals`) in TOML text, keeping every comment and blank line.

    A key that exists has only its value replaced (its trailing comment stays). A new key goes after the last
    key of its section; a section that does not exist is appended at the end.
    """
    lines = text.splitlines()
    pending: dict[str, dict[str, Any]] = {}
    for dotted, v in changes.items():
        sec, _, name = dotted.rpartition(".")
        pending.setdefault(sec, {})[name] = v
    drop: dict[str, set[str]] = {}
    for dotted in removals:
        sec, _, name = dotted.rpartition(".")
        drop.setdefault(sec, set()).add(name)

    out: list[str] = []
    section = ""
    seen_sections: set[str] = set()
    last_key_idx: dict[str, int] = {}
    header_idx: dict[str, int] = {}

    for line in lines:
        m = _HEADER.match(line)
        if m:
            section = m.group(1)
            seen_sections.add(section)
            header_idx[section] = len(out)
            out.append(line)
            continue
        km = _KEY.match(line)
        if km and not line.lstrip().startswith("#"):
            name = km.group(1)
            if name in drop.get(section, set()):
                continue
            if name in pending.get(section, {}):
                _, comment = split_value_comment(km.group(2))
                head = line[: line.index("=")].rstrip()  # keeps the indent
                body = f"{head} = {toml_value(pending[section].pop(name))}"
                if comment:  # keep the comment's column where possible
                    col = line.rfind(comment)
                    body = body + (" " * max(2, col - len(body))) + comment
                out.append(body)
                last_key_idx[section] = len(out) - 1
                continue
            last_key_idx[section] = len(out)
        out.append(line)

    # new keys in existing sections: insert after the last key (back to front so indices stay valid)
    inserts: list[tuple[int, list[str]]] = []
    for sec, keys in pending.items():
        if not keys or sec not in seen_sections:
            continue
        at = last_key_idx.get(sec, header_idx[sec]) + 1
        inserts.append((at, [f"{k} = {toml_value(v)}" for k, v in keys.items()]))
    for at, new in sorted(inserts, key=lambda t: -t[0]):
        out[at:at] = new
    for sec, keys in pending.items():
        if keys and sec not in seen_sections:
            if out and out[-1].strip():
                out.append("")
            out.append(f"[{sec}]")
            out += [f"{k} = {toml_value(v)}" for k, v in keys.items()]
    if removals:  # a section a removal emptied (no comment on its header, only blank lines after it) goes too
        cleaned: list[str] = []
        i = 0
        while i < len(out):
            m = _HEADER.match(out[i])
            if m and not m.group(2) and m.group(1) in drop:
                j = i + 1
                while j < len(out) and not out[j].strip():
                    j += 1
                if j == len(out) or _HEADER.match(out[j]):
                    while cleaned and not cleaned[-1].strip():
                        cleaned.pop()
                    if j < len(out) and cleaned:
                        cleaned.append("")
                    i = j
                    continue
            cleaned.append(out[i])
            i += 1
        out = cleaned
    result = "\n".join(out).rstrip("\n") + "\n"
    tomllib.loads(result)  # never write something the loader cannot read
    return result
