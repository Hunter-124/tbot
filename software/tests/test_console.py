"""TBOT Console back end: config editing that keeps comments, schema, bore-laser ring fit, session indexing."""

from __future__ import annotations

import json
import math
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import pytest

from tbot.console import sessions, tomledit, twist

ROOT = Path(__file__).resolve().parents[1]

LOCAL = """# Per-rig overrides

[actuator.servo]
rest_us = 1340.0              # touch 1320, 20 us clear
fire_us = 1075.0              # break ~1095

[optics]                      # 12 mm lens
model = "pinhole"
"""


def test_patch_keeps_comments_and_replaces_values():
    out = tomledit.patch_text(LOCAL, {"actuator.servo.rest_us": 1345.0, "optics.model": "fisheye"})
    assert "rest_us = 1345.0              # touch 1320, 20 us clear" in out
    assert "# break ~1095" in out and "[optics]                      # 12 mm lens" in out
    data = tomllib.loads(out)
    assert data["actuator"]["servo"]["rest_us"] == 1345.0 and data["optics"]["model"] == "fisheye"


def test_patch_adds_keys_and_sections():
    out = tomledit.patch_text(LOCAL, {"actuator.servo.hold_ms": 400.0, "fire.min_range_m": 7.0})
    data = tomllib.loads(out)
    assert data["actuator"]["servo"]["hold_ms"] == 400.0
    assert data["fire"]["min_range_m"] == 7.0
    # the new servo key lands inside its own section, before [optics]
    assert out.index("hold_ms") < out.index("[optics]")


def test_patch_removal_drops_emptied_section_but_keeps_commented_ones():
    out = tomledit.patch_text(LOCAL + "\n[fire]\nmin_range_m = 7.0\n", {}, {"fire.min_range_m"})
    assert "[fire]" not in out
    out = tomledit.patch_text(LOCAL, {}, {"optics.model"})
    assert "[optics]                      # 12 mm lens" in out  # a commented header is the user's: kept


def test_patch_hash_inside_string_is_not_a_comment():
    out = tomledit.patch_text('[a]\ns = "x # y"  # note\n', {"a.s": "z"})
    assert out.splitlines()[1].startswith('s = "z"') and "# note" in out


def test_coerce_types():
    assert tomledit.coerce("12", 1, "k") == 12 and isinstance(tomledit.coerce(3, 1.0, "k"), float)
    assert tomledit.coerce("true", False, "k") is True
    with pytest.raises(ValueError):
        tomledit.coerce("abc", 1, "k")
    with pytest.raises(ValueError):
        tomledit.coerce(float("nan"), 1.0, "k")


def test_schema_from_default_toml():
    schema = tomledit.parse_schema((ROOT / "config" / "default.toml").read_text(), LOCAL)
    keys = {k["key"]: k for s in schema["sections"] for k in s["keys"]}
    assert keys["actuator.servo.rest_us"]["overridden"] and keys["actuator.servo.rest_us"]["value"] == 1340.0
    assert keys["record.mode"]["choices"] == ["off", "armed", "continuous"]
    assert keys["fire.min_range_m"]["unit"] == "m" and keys["camera.exposure_us"]["unit"] == "µs"
    assert all(k in keys for k in schema["common"])


def _ring_frames(cx, cy, r, n=60, size=(400, 640), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    for k in range(n):
        a = 2 * math.pi * k / n
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        img = 60 + rng.normal(0, 2, size) + 150 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.8 ** 2))
        yield np.clip(img, 0, 255).astype(np.uint8)


def test_twist_ring_centre():
    rows = [twist.find_dot(f, (300, 200, 90)) for f in _ring_frames(301.3, 198.6, 30)]
    ring = twist.fit_ring([r for r in rows if r])
    assert abs(ring["cx"] - 301.3) < 0.3 and abs(ring["cy"] - 198.6) < 0.3 and abs(ring["r"] - 30) < 0.5
    assert ring["sectors"] == 12 and ring["grade"] in ("excellent", "good")


def test_twist_rejects_too_few_dots():
    with pytest.raises(ValueError):
        twist.fit_ring([(1.0, 2.0, 30.0, 1.0)] * 5)


def test_sessions_shots_and_notes():
    with tempfile.TemporaryDirectory() as d:
        s = Path(d) / "20261117-104603-press"
        s.mkdir()
        (s / "summary.json").write_text(json.dumps({"span_s": 4.1, "frames": 3, "fps": 30.0, "live": True,
                                                     "events": {"fire": 1},
                                                     "fire_events": [{"type": "fire", "p_hit": 0.35}]}))
        (s / "frames.csv").write_text("seq,t_exposure_ns,t_capture_ns,exposure_us,gain\n5,0,0,9,1\n6,0,0,9,1\n7,0,0,9,1\n")
        (s / "telemetry.jsonl").write_text('{"seq": 5, "fired": false}\n{"seq": 6, "fired": false}\n'
                                           '{"seq": 7, "fired": true}\n')
        det = sessions.detail(s, 30.0)
        assert det["shot_list"][0]["frame"] == 2 and det["shot_list"][0]["clip_t"] == pytest.approx(2 / 30)
        assert det["shot_list"][0]["p_hit"] == 0.35
        sessions.save_notes(s, {"outcome": "hit", "note": "station 4"})
        listed = sessions.list_sessions(Path(d), 30.0)
        assert listed[0]["outcome"] == "hit" and listed[0]["note"] == "station 4"
        with pytest.raises(ValueError):
            sessions.save_notes(s, {"outcome": "maybe"})
        assert sessions.servable("shot_1.png") == "image/png" and sessions.servable("frames.bin") == "application/octet-stream"  # the laptop app renders from it


def test_shot_placed_by_seq_not_telemetry_line():
    """frames.csv has a frame the pipeline never processed: the clip frame of the shot is its frames.csv row."""
    with tempfile.TemporaryDirectory() as d:
        s = Path(d) / "p"
        s.mkdir()
        (s / "frames.csv").write_text("seq,t_exposure_ns,t_capture_ns,exposure_us,gain\n"
                                      "10,0,0,9,1\n11,0,0,9,1\n12,0,0,9,1\n13,0,0,9,1\n")
        (s / "telemetry.jsonl").write_text('{"seq": 10, "fired": false}\n{"seq": 12, "fired": false}\n'
                                           '{"seq": 13, "fired": true}\n')
        assert sessions.shot_frames(s) == [3]


def test_key_frame_prefers_the_shot_then_best_p_hit():
    pytest.importorskip("cv2")
    from tbot.console.thumbs import key_frame

    rows = [(s, 0, 1000 * s) for s in range(10, 20)]
    tele = {12: {"decision": {"p_hit": 0.3}}, 15: {"decision": {"p_hit": 0.6}}, 17: {"fired": True}}
    assert key_frame(rows, tele, None) == (7, "shot")
    del tele[17]
    assert key_frame(rows, tele, None) == (5, "best P(hit)")
    assert key_frame(rows, {}, [12000, 16000])[1] == "press"


def test_press_recording_switch_wins_over_the_service_script(tmp_path, monkeypatch):
    """The Console's Press recording switch is record.mode in local.toml; the service script must not override it."""
    from tbot.config import load_config

    default_text = (ROOT / "config" / "default.toml").read_text()
    schema = tomledit.parse_schema(default_text, "")
    keys = {k["key"]: k for s in schema["sections"] for k in s["keys"]}
    assert keys["record.mode"]["default"] == "armed" and "record.mode" in schema["common"]
    off = tomledit.patch_text("", {"record.mode": tomledit.coerce("off", "armed", "record.mode")}, set())
    local = tmp_path / "local.toml"
    local.write_text(off)
    monkeypatch.setenv("TBOT_LOCAL_CONFIG", str(local))
    assert load_config().record.mode == "off"
    back = tomledit.patch_text(off, {}, {"record.mode"})
    local.write_text(back)
    assert load_config().record.mode == "armed"
    script = (ROOT / "scripts" / "tbot-run.sh").read_text()
    assert "--record" not in [ln.split("#")[0] for ln in script.splitlines() if "args=(" in ln][0]
    assert "args+=(--record" not in script
