import math

import pytest

from tbot.tools.tune import candidate_metrics, candidates, objective, pareto_front, parse_grid


def test_grid_parsing_and_sampling():
    grid = parse_grid(["tracker.confirm_hits=2,3", "fire.min_hit_probability=0.3,0.5"])
    assert grid == {"tracker.confirm_hits": [2, 3], "fire.min_hit_probability": [0.3, 0.5]}
    assert len(candidates(grid)) == 4
    assert candidates(grid, 2) == candidates(grid, 2)
    assert len(candidates(grid, 9)) == 4
    strings = parse_grid(['optics.model="fish,eye",pinhole'])
    assert strings["optics.model"] == ["fish,eye", "pinhole"]
    for param in ["tracker.confirm_hits=2.5", "tracker.nonexistent=2", "tracker.confirm_hits=2,", "missing"]:
        with pytest.raises(ValueError):
            parse_grid([param])


def test_objective_and_pareto():
    def result(name, false, hits, latency, rate=0.5):
        return {"id": name, "metrics": {"false_shots": false, "hits": hits,
                                         "first_fire_ms": latency, "hit_rate": rate}}
    fast = result("fast", 0, 1, 100)
    hits = result("hits", 0, 2, 150)
    unsafe = result("unsafe", 1, 10, 1)
    dominated = result("dominated", 0, 1, 200)
    missing = result("missing", 0, 1, None)
    ranked = sorted([unsafe, fast, missing, hits, dominated], key=objective)
    assert [r["id"] for r in ranked] == ["hits", "fast", "dominated", "missing", "unsafe"]
    assert {r["id"] for r in pareto_front(ranked)} == {"fast", "hits", "unsafe"}
    assert math.isinf(objective(missing)[2])


def test_metrics_count_safety_violations_once():
    def run(scenario, shots, visible=10, detected=8, armed=True):
        return {"scenario": scenario, "armed": armed, "shots": shots, "fires": len(shots),
                "hits": sum(s["hit"] for s in shots), "visible_frames": visible,
                "detected_frames": detected, "detection_rate": detected / visible if visible else math.nan,
                "track_latency_ms": 25.0, "lead_error_deg": {"p50": 0.5, "p95": 1.0},
                "t_first_visible_ns": 1_000_000 if visible else None}
    hit = {"hit": True, "miss_m": 0.2, "reason": "", "armed_at_fire": True, "t_fire_ns": 11_000_000}
    unsafe = {"hit": False, "miss_m": math.nan, "reason": "no clay in flight at shot exit",
              "armed_at_fire": False, "t_fire_ns": 11_000_000}
    metrics = candidate_metrics([run("static_gun", [hit])], [run("no_clay", [unsafe], 0, 0, False)])
    assert metrics["false_shots"] == 1
    assert metrics["hits"] == metrics["shots"] == 1
    assert metrics["detection_rate"] == 0.8
    assert metrics["track_latency_ms"] == 25.0
    assert metrics["first_fire_ms"] == 10.0
    assert metrics["miss_p95_m"] == 0.2
