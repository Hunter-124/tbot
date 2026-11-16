"""CSI lane monitor (tbot.tools.csi_lanes): pure decoding, on histograms measured on the rig 2026-11-16."""

from __future__ import annotations

from tbot.tools import csi_lanes as cl

# Sensor streaming, link down: data lanes burst (both wires low ~8 %), clock wire B never goes low.
CLK_FAULT = {0xE0000005: 71102, 0x65000005: 7257, 0x60000005: 68, 0xE5000005: 40, 0xED000005: 10}
DAT_OK = {0xC0000005: 71314, 0x06000005: 6580, 0x81000005: 238, 0x02000005: 108, 0xC6000005: 66,
          0x80000005: 56, 0x01000005: 50, 0x03000005: 47}
IDLE = {0xC0000005: 39938}


def test_clock_wire_stuck_is_named_with_its_pins():
    clk, d0, d1 = cl.lane_state("CLK", CLK_FAULT), cl.lane_state("D0", DAT_OK), cl.lane_state("D1", DAT_OK)
    assert clk.verdict == "ONE WIRE STUCK" and clk.moving == (True, False) and clk.error
    assert d0.verdict == "ok" and not d0.error
    msg = cl.summarise([clk, d0, d1], 0)
    assert "CLK pair" in msg and "8/9" in msg and "D0" not in msg


def test_idle_all_ok_and_link_up():
    idle = [cl.lane_state(n, IDLE) for n in ("CLK", "D0", "D1")]
    assert all(s.verdict == "idle" for s in idle)
    assert "no lane activity" in cl.summarise(idle, 0)
    ok = [cl.lane_state(n, DAT_OK) for n in ("CLK", "D0", "D1")]
    assert "no packet completes" in cl.summarise(ok, 0)
    assert cl.summarise(ok, 120).startswith("LINK UP")


def test_unicam_base_from_iomem(tmp_path):
    p = tmp_path / "iomem"
    p.write_text("3f801000-3f8017ff : 3f801000.csi unicam\n3f802004-3f802007 : 3f801000.csi cmi\n")
    assert cl.unicam_base(str(p)) == 0x3F801000
    p.write_text("00000000-00000000 : 3f801000.csi unicam\n")  # non-root read: addresses are zeroed
    assert cl.unicam_base(str(p)) is None
    assert cl.unicam_base(str(tmp_path / "missing")) is None
