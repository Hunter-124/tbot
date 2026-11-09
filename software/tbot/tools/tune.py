"""Offline synthetic search; results are not evidence of safe real-world operation.

Candidates are ordered lexicographically by false shots (ascending), hits (descending),
then mean visible-to-first-fire time (ascending; missing is worst). Hit rate breaks ties.
The Pareto front uses those first three objectives. Track and fire latencies are means
among runs with a visible clay and a corresponding event. Lead p95 is the p95 of the
per-run lead p95 values, since the simulator exposes summaries rather than samples.
Every candidate also runs the requested scenarios unarmed and a no-clay safety scenario.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

from tbot.config import (
    DEFAULT_CONFIG, LOCAL_CONFIG, LatencyBudget, _toml_dumps, add_common_args,
    load_calibration, load_config, local_config_path, now_iso, resolve_path, save_json,
)
from tbot.sim.scenario import SCENARIOS
from tbot.stats import summarize
from tbot.tools.sim_eval import evaluate_run, summarize_runs


def _split_values(text: str) -> list[str]:
    """Split a comma-separated list without splitting TOML strings or arrays."""
    values, start, depth, quote, escape = [], 0, 0, None, False
    for i, char in enumerate(text):
        if quote:
            if quote == '"' and escape:
                escape = False
            elif quote == '"' and char == "\\":
                escape = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced parameter value")
        elif char == "," and depth == 0:
            values.append(text[start:i].strip())
            start = i + 1
    if quote or depth:
        raise ValueError("unbalanced parameter value")
    values.append(text[start:].strip())
    if any(not value for value in values):
        raise ValueError("parameter values must not be empty")
    return values


def _get_path(obj, path: str):
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj


def parse_grid(params: Sequence[str], overrides: Sequence[str] = (), sets: Sequence[str] = ()) -> dict[str, list]:
    """Validate each grid value through the normal configuration schema."""
    grid = {}
    for spec in params:
        if "=" not in spec:
            raise ValueError(f"--param expects key.path=value[,value...], got {spec!r}")
        key, text = spec.split("=", 1)
        key = key.strip()
        if not key or key in grid:
            raise ValueError(f"empty or repeated parameter key: {key!r}")
        values = []
        for value in _split_values(text):
            cfg = load_config(overrides=list(overrides), sets=[*sets, f"{key}={value}"])
            parsed = _get_path(cfg, key)
            if dataclasses.is_dataclass(parsed):
                raise ValueError(f"parameter must name a config value, not a table: {key}")
            values.append(parsed)
        grid[key] = values
    if not grid:
        raise ValueError("at least one --param is required")
    return grid


def candidates(grid: dict[str, list], random_count: int | None = None) -> list[dict]:
    """Generate the grid, or a deterministic sample without replacement (seed 0)."""
    keys = list(grid)
    sizes = [len(grid[key]) for key in keys]
    count = math.prod(sizes)
    if random_count is None:
        combos = itertools.product(*(grid[key] for key in keys))
    else:
        if random_count < 1:
            raise ValueError("--random must be >= 1")
        indexes = sorted(random.Random(0).sample(range(count), min(random_count, count)))
        def decode(index):
            vals = [None] * len(keys)
            for i in range(len(keys) - 1, -1, -1):
                index, digit = divmod(index, sizes[i])
                vals[i] = grid[keys[i]][digit]
            return vals
        combos = map(decode, indexes)
    return [dict(zip(keys, vals)) for vals in combos]


def _override_tree(params: dict) -> dict:
    tree = {}
    for key, value in params.items():
        node = tree
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


def _assign(cfg, key: str, value) -> None:
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)


def candidate_metrics(runs: Sequence[dict], safety_runs: Sequence[dict]) -> dict:
    """Aggregate armed runs plus unarmed/no-clay safety checks."""
    summary = summarize_runs(runs)
    shots = [s for run in runs for s in run["shots"]]
    safety_shots = [s for run in safety_runs for s in run["shots"]]
    # A shot can violate both conditions; count the event only once.
    false_shots = sum(
        1 for run in [*runs, *safety_runs] for s in run["shots"]
        if not run["armed"] or not s["armed_at_fire"] or run["scenario"] == "no_clay"
        or s["reason"] == "no clay in flight at shot exit"
    )
    visible = sum(r["visible_frames"] for r in runs)
    detected = sum(r["detected_frames"] for r in runs)
    fire_times = [(r["shots"][0]["t_fire_ns"] - r["t_first_visible_ns"]) / 1e6
                  for r in runs if r["t_first_visible_ns"] is not None and r["shots"]]
    track_times = [r["track_latency_ms"] for r in runs]
    lead = [r["lead_error_deg"]["p95"] for r in runs]
    miss = [s["miss_m"] for s in shots]
    def finite_or_none(value):
        return float(value) if math.isfinite(value) else None
    return {
        "hits": summary["hits"], "shots": summary["fires"],
        "hit_rate": finite_or_none(summary["hit_rate"]), "false_shots": false_shots,
        "safety_shots": len(safety_shots),
        "detection_rate": detected / visible if visible else None,
        "track_latency_ms": finite_or_none(summarize(track_times)["mean"]),
        "first_fire_ms": finite_or_none(summarize(fire_times)["mean"]),
        "lead_p95_deg": finite_or_none(summarize(lead)["p95"]),
        "miss_p95_m": finite_or_none(summarize(miss)["p95"]),
    }


def objective(result: dict) -> tuple:
    m = result["metrics"]
    return (m["false_shots"], -m["hits"],
            m["first_fire_ms"] if m["first_fire_ms"] is not None else math.inf,
            -(m["hit_rate"] if m["hit_rate"] is not None else 0.0))


def pareto_front(results: Sequence[dict]) -> list[dict]:
    """Return nondominated candidates on false shots, hits and first-fire latency."""
    points = [objective(r)[:3] for r in results]
    return [r for i, r in enumerate(results) if not any(
        j != i and all(a <= b for a, b in zip(other, points[i]))
        and any(a < b for a, b in zip(other, points[i]))
        for j, other in enumerate(points)
    )]


def _evaluate_candidate(job: tuple) -> dict:
    index, params, overrides, sets, calib_dir, names, seeds = job
    # Load per candidate: calibration paths and budgets may themselves be tuned.
    from tbot.config import _toml_value
    assignments = [f"{k}={_toml_value(v)}" for k, v in params.items()]
    cfg = load_config(overrides=overrides, sets=[*sets, *assignments])
    if calib_dir is not None:
        cfg.paths.calib_dir = calib_dir
    for key, value in params.items():
        _assign(cfg, key, value)
    calib = load_calibration(cfg)
    # Detector calibration is normally applied last; explicitly tuned values must win.
    for key, value in params.items():
        _assign(cfg, key, value)
    calib.budget = LatencyBudget.from_config(cfg, calib.latency)
    runs = [evaluate_run(cfg, calib, name, seed, armed=True)
            for name in names for seed in range(1, seeds + 1)]
    safety_names = list(dict.fromkeys([*names, "no_clay"]))
    safety = [evaluate_run(cfg, calib, name, seed, armed=False)
              for name in safety_names for seed in range(1, seeds + 1)]
    if "no_clay" not in names:
        safety += [evaluate_run(cfg, calib, "no_clay", seed, armed=True)
                   for seed in range(1, seeds + 1)]
    return {"id": index, "params": params, "metrics": candidate_metrics(runs, safety),
            "summary": summarize_runs(runs), "runs": runs, "safety_runs": safety}


def _fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def _json_finite(value):
    """Encode unavailable simulation statistics as JSON null, not NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return value


def _protected_output(path: str) -> Path:
    result = resolve_path(path).resolve()
    protected = [DEFAULT_CONFIG, LOCAL_CONFIG, local_config_path()]
    if any(p is not None and result == p.resolve() for p in protected):
        raise ValueError("output must not overwrite default.toml or local.toml")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbot tune", description=__doc__)
    add_common_args(parser)
    parser.add_argument("--param", action="append", default=[], metavar="KEY=V1,V2", help="config grid (repeatable)")
    parser.add_argument("--scenarios", nargs="+", default=None, metavar="NAME", help="scenarios (default: all)")
    parser.add_argument("--seeds", type=int, default=1, help="seeds 1..N per scenario")
    parser.add_argument("--random", type=int, default=None, metavar="N", help="sample up to N grid points, seed 0")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1), metavar="K")
    parser.add_argument("--out", metavar="JSON", help="write full results as JSON")
    parser.add_argument("--write-best", metavar="TOML", help="write winning parameter overrides")
    args = parser.parse_args(argv)
    names = args.scenarios or list(SCENARIOS)
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenarios: {', '.join(unknown)}; available: {', '.join(SCENARIOS)}")
    if args.seeds < 1 or args.jobs < 1:
        parser.error("--seeds and --jobs must be >= 1")
    try:
        grid = parse_grid(args.param, args.config, args.set)
        points = candidates(grid, args.random)
        out = _protected_output(args.out) if args.out else None
        best_path = _protected_output(args.write_best) if args.write_best else None
        if out is not None and out == best_path:
            raise ValueError("--out and --write-best must be different files")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print("Caveat: synthetic-scene results must be confirmed on recorded sessions / the HITL bench.", flush=True)
    print("Objective: fewest false shots, most hits, earliest mean first fire; hit rate breaks ties.", flush=True)
    jobs = [(i + 1, params, args.config, args.set, args.calib_dir, names, args.seeds)
            for i, params in enumerate(points)]
    if args.jobs == 1:
        results = list(map(_evaluate_candidate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(_evaluate_candidate, jobs))
    ranked = sorted(results, key=objective)
    front_ids = {r["id"] for r in pareto_front(ranked)}
    print("rank  id  false  hits  shots  hit%  det%  track ms  fire ms  lead p95  miss p95  parameters")
    for rank, r in enumerate(ranked, 1):
        m = r["metrics"]
        pct = lambda x: None if x is None else x * 100
        print(f"{rank:>4}  {r['id']:>2}  {m['false_shots']:>5}  {m['hits']:>4}  {m['shots']:>5}  "
              f"{_fmt(pct(m['hit_rate'])):>5}  {_fmt(pct(m['detection_rate'])):>5}  "
              f"{_fmt(m['track_latency_ms']):>8}  {_fmt(m['first_fire_ms']):>7}  "
              f"{_fmt(m['lead_p95_deg'], 2):>8}  {_fmt(m['miss_p95_m'], 2):>8}  "
              + ", ".join(f"{k}={v!r}" for k, v in r["params"].items()))
    print("Pareto front (candidate IDs): " + ", ".join(str(r["id"]) for r in ranked if r["id"] in front_ids))
    if out:
        save_json(out, _json_finite({"created": now_iso(), "scenarios": names, "seeds": args.seeds,
                        "grid": grid, "objective": "false shots, -hits, first fire ms, -hit rate",
                        "pareto_ids": [r["id"] for r in ranked if r["id"] in front_ids], "results": ranked}))
        print(f"wrote {out}")
    if best_path:
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(_toml_dumps(_override_tree(ranked[0]["params"])) + "\n")
        print(f"wrote {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
