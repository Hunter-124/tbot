"""Explain one real pipeline run, without ever accessing physical GPIO or actuators."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from tbot.app.run import Pipeline
from tbot.camera import open_source
from tbot.camera.synthetic import SyntheticSource, T0_NS
from tbot.config import add_common_args, load_from_args
from tbot.fire.solver import _DTOF_STEP_M, _hit_probability, _tof
from tbot.hw.gpio import SimGpio
from tbot.tools.sim_eval import DET_TOL_PX

STAGES = ("visible", "detected", "associated", "confirmed", "range_in_window",
          "elevation_ok", "p_hit_ok", "best_moment", "fire")


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _stats(rows, key):
    values = [(r['frame'], r[key]) for r in rows if r[key] is not None]
    if not values:
        return {'min': None, 'p50': None, 'max': None}
    a = np.array([v for _, v in values], dtype=np.float64)
    return {'min': float(np.min(a)), 'p50': float(np.median(a)), 'max': float(np.max(a))}


def _truth_diagnostics(source, pipe, t_ns, exposure_us):
    """Direct measurements, not a claim that any one observation caused the miss."""
    truth = source.truth(t_ns)
    t = (t_ns - T0_NS) * 1e-9
    visible = bool(truth['visible'])
    result = {'visible': visible, 'in_frame': False, 'in_mask': False, 'u': None, 'v': None,
              'truth_range_m': _finite(truth['clay_range_m']), 'diameter_px_upper': None,
              'motion_px': None, 'peak_contrast_dn': None, 'threshold_dn': None,
              'threshold_pixels': None, 'miss_factors': ''}
    if not visible:
        return result
    px = source._clay_pixel(t)
    if px is None:
        result['miss_factors'] = 'outside image'
        return result
    u, v = map(float, px)
    result.update(u=u, v=v)
    ui, vi = int(round(u)), int(round(v))
    inside = 0 <= ui < source.width and 0 <= vi < source.height
    result['in_frame'] = inside
    if not inside:
        result['miss_factors'] = 'outside image'
        return result
    result['in_mask'] = bool(pipe.mask[vi, ui])
    rng = result['truth_range_m']
    ifov = float(np.asarray(pipe.cam.ifov(np.array([[u, v]], dtype=np.float64))).ravel()[0])
    if rng and ifov > 0:
        # Bounding diameter: tilted disc is often substantially smaller in one dimension.
        result['diameter_px_upper'] = 2 * math.atan2(0.5 * math.hypot(source.scenario.clay_diameter_m,
                                                       source.scenario.clay_thickness_m), rng) / ifov
    dt = exposure_us * 0.5e-6
    p0, p1 = source._clay_pixel(t - dt), source._clay_pixel(t + dt)
    if p0 is not None and p1 is not None:
        result['motion_px'] = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    det = pipe.cfg.detector
    threshold = max(float(det.k_sigma) * pipe.detector.noise_sigma, float(det.min_contrast))
    result['threshold_dn'] = _finite(threshold)
    debug = pipe.detector.debug
    radius = max(2, min(20, int(math.ceil((result['diameter_px_upper'] or 2) / 2 + (result['motion_px'] or 0) / 2 + 1))))
    x0, x1 = max(0, ui-radius), min(source.width, ui+radius+1)
    y0, y1 = max(0, vi-radius), min(source.height, vi+radius+1)
    contrast = debug['contrast'][y0:y1, x0:x1]
    signed = contrast if det.polarity == 'bright' else -contrast if det.polarity == 'dark' else np.abs(contrast)
    result['peak_contrast_dn'] = float(np.max(signed))
    result['threshold_pixels'] = int(np.count_nonzero(debug['binary'][y0:y1, x0:x1]))
    factors = []
    if not result['in_mask']:
        factors.append('masked centre')
    if result['peak_contrast_dn'] <= threshold:
        factors.append('local contrast below threshold')
    if result['threshold_pixels'] < int(det.min_area_px):
        factors.append('local threshold area below minimum')
    if result['motion_px'] is not None and result['motion_px'] >= 0.5:
        factors.append('motion blur >= 0.5 px')
    if not factors:
        factors.append('component filtering, centroid tolerance, or detection limit')
    result['miss_factors'] = '; '.join(factors)
    return result


def explain_run(cfg, calib, source_spec='synthetic:skeet_crossing', seed=1, frames=None,
                *, diagnostics=False):
    """Run a source and return JSON-compatible summary and per-frame observations.

    Synthetic stages follow a truth-matched detection and its actual track, not unrelated
    clutter tracks. Replay has no truth; its stages use any detection and associated track.
    Stage counts are cumulative within a frame; a coasting track is not a detection.
    """
    cfg = copy.deepcopy(cfg)
    cfg.gpio.backend = 'sim'
    cfg.actuator.type = 'sim'
    cfg.fire.live = True
    cfg.record.mode = 'off'
    cfg.stream.enabled = False
    cfg.detector.debug = True
    kind, _, arg = source_spec.partition(':')
    if kind == 'synthetic':
        source = SyntheticSource(cfg, calib, scenario=arg or None, seed=seed)
    elif kind == 'replay' and arg:
        source = open_source(source_spec, cfg, calib)
    else:
        raise ValueError('source must be synthetic:<scenario> or replay:<session>')
    gpio = SimGpio()
    pipe = Pipeline(cfg, calib, source, gpio=gpio)
    if kind == 'synthetic':
        gpio.inject(int(cfg.arming.pin), 0 if cfg.arming.active_low else 1)
    rows = []
    sigma_rows = []
    try:
        while frames is None or len(rows) < frames:
            tele = pipe.step()
            if tele is None:
                break
            t = int(tele['t_exposure_ns'])
            truth = _truth_diagnostics(source, pipe, t, tele['exposure_us']) if kind == 'synthetic' else None
            if truth is not None:
                matched = [d for d in pipe.detections if truth['visible'] and truth['in_frame'] and
                           math.hypot(d.u - truth['u'], d.v - truth['v']) <= DET_TOL_PX]
            else:
                matched = list(pipe.detections)
            associated = [tr for tr in pipe.tracks if tr.misses == 0 and
                          any(tr.last_detection is d for d in matched)]
            selected = next((tr for tr in associated if tr.track_id == pipe.decision.track_id), None)
            dec = pipe.decision
            # The solver only evaluates its best track, and also requires min_track_hits.
            confirmed = selected is not None and selected.confirmed and selected.hits >= cfg.fire.min_track_hits
            range_ok = confirmed and cfg.fire.min_range_m <= dec.range_m <= cfg.fire.max_range_m
            elevation = dec.extra.get('clay_el_rad')
            elevation_ok = range_ok and elevation is not None and elevation >= math.radians(cfg.fire.min_elevation_deg)
            p_ok = elevation_ok and dec.p_hit >= cfg.fire.min_hit_probability
            best = p_ok and not any(p > dec.p_hit for p in dec.extra.get('future_p_hit', []))
            visible = truth['visible'] if truth is not None else None
            detected = bool(matched) and (visible is not False)
            assoc = detected and bool(associated)
            row = {'frame': int(tele['seq']), 't_exposure_ns': t, 'armed': bool(tele['armed']),
                   'visible': visible, 'detected': detected, 'associated': assoc,
                   'confirmed': bool(assoc and confirmed), 'range_in_window': bool(assoc and range_ok),
                   'elevation_ok': bool(assoc and elevation_ok), 'p_hit_ok': bool(assoc and p_ok),
                   'best_moment': bool(assoc and best), 'fire': bool(assoc and best and tele['fired']),
                   'decision_fire': bool(dec.fire), 'fired': bool(tele['fired']),
                   'reason': dec.reason, 'track_id': dec.track_id,
                   'p_hit': _finite(dec.p_hit), 'miss_deg': _finite(math.degrees(dec.miss_rad)),
                   'sigma_deg': _finite(math.degrees(dec.sigma_rad)), 'pattern_deg': _finite(math.degrees(dec.pattern_rad)),
                   'range_m': _finite(dec.range_m)}
            if diagnostics and dec.track_id is not None:
                track = next((tr for tr in pipe.tracks if tr.track_id == dec.track_id), None)
                if (track is not None and track.confirmed and track.hits >= cfg.fire.min_track_hits
                        and 'track_sigma_rad' in dec.extra):
                    dist = dec.extra['dist_m']
                    lo = max(dist - _DTOF_STEP_M, 1e-3)
                    dtof_dr = (_tof(pipe.shot, dist + _DTOF_STEP_M) - _tof(pipe.shot, lo)) / (
                        dist + _DTOF_STEP_M - lo)
                    rate = dec.extra['clay_rate_rad_s']
                    sigma_rows.append({
                        'frame': row['frame'], 'p_hit': dec.p_hit,
                        'track': dec.extra['track_sigma_rad'],
                        'jitter': rate * pipe.budget.fire_jitter_s(pipe.solver._atype),
                        'range': rate * dtof_dr * float(track.range_sigma_m),
                        'miss': dec.miss_rad, 'pattern': dec.pattern_rad, 'sigma': dec.sigma_rad,
                        'range_ok': bool(cfg.fire.min_range_m <= dec.range_m <= cfg.fire.max_range_m),
                        'elevation_ok': bool(elevation is not None and elevation >= math.radians(cfg.fire.min_elevation_deg)),
                    })
            if truth is not None:
                row.update({k: v for k, v in truth.items() if k != 'visible'})
                if detected:
                    row['miss_factors'] = ''
            rows.append(row)
    finally:
        pipe.close()
        gpio.close()
    funnel = {}
    for stage in STAGES:
        if stage == 'visible' and kind == 'replay':
            funnel[stage] = {'count': None, 'first_frame': None}
        else:
            hits = [r['frame'] for r in rows if r[stage]]
            funnel[stage] = {'count': len(hits), 'first_frame': hits[0] if hits else None}
    reasons = dict(Counter(r['reason'] for r in rows))
    miss_factors = Counter()
    for row in rows:
        if row.get('visible') and not row['detected']:
            miss_factors.update(row['miss_factors'].split('; '))
    timelines = {key: _stats(rows, key) for key in ('p_hit', 'miss_deg', 'sigma_deg', 'range_m')}
    best_rows = [r for r in rows if r['p_hit'] is not None]
    timelines['best_p_hit_frame'] = max(best_rows, key=lambda r: r['p_hit'])['frame'] if best_rows else None
    counts = [funnel[s]['count'] for s in STAGES]
    # Compare the same-frame losses, excluding truth visibility for replay.
    if kind == 'synthetic' and not counts[0]:
        blocker = 'No airborne clay was visible in the synthetic truth.'
    elif not any(r['detected'] for r in rows):
        blocker = 'No clay-matched detections were obtained.' if kind == 'synthetic' else 'No detections were obtained.'
    else:
        pairs = [(STAGES[i], STAGES[i+1], counts[i] - counts[i+1]) for i in range(len(STAGES)-1)
                 if counts[i] is not None and counts[i+1] is not None]
        a, b, drop = max(pairs, key=lambda p: p[2])
        blocker = f'{a} -> {b} is the largest observed same-frame funnel loss ({drop} frames).'
    result = {'source': source_spec, 'seed': seed if kind == 'synthetic' else None, 'frames': len(rows),
            'funnel': funnel, 'actual_fires': sum(r['fired'] for r in rows),
            'reasons': reasons, 'miss_factors': dict(miss_factors),
            'timelines': timelines, 'biggest_blocker': blocker, 'per_frame': rows,
            'notes': ['Stages are same-frame cumulative; confirmed also requires the solver minimum hits and selection of the matched track. Actual fires include decisions on coasting tracks.',
                      'Truth visibility means airborne above horizon, not necessarily inside the image or mask.',
                      'Miss factors are local measurements and can overlap; they are not proven causes.',
                      'Replay has no truth and is disarmed; detections and tracks are not truth-matched.']}
    if diagnostics:
        result['_diagnostics'] = {'sigma_rows': sigma_rows, 'min_hit_probability': cfg.fire.min_hit_probability,
                                  'min_range_m': cfg.fire.min_range_m, 'max_range_m': cfg.fire.max_range_m}
    return result


def _format_number(value, precision):
    return f'{value:.{precision}f}' if value is not None else 'n/a'


def _reason_buckets(rows):
    buckets = {}
    for row in rows:
        reason = re.sub(r'p_hit [\d.]+ < ([\d.]+)', r'p_hit below \1', row['reason'])
        reason = re.sub(r'(?<![A-Za-z_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\d.])', 'N', reason) \
            if not reason.startswith('p_hit below') else reason
        buckets.setdefault(reason, []).append(row['p_hit'])
    return buckets


def _physical_blocker(result):
    diag = result['_diagnostics']
    rows = result['per_frame']
    stages = result['funnel']
    pairs = [(STAGES[i + 1], stages[STAGES[i]]['count'] - stages[STAGES[i + 1]]['count'])
             for i in range(len(STAGES) - 1)
             if stages[STAGES[i]]['count'] is not None and stages[STAGES[i + 1]]['count'] is not None]
    target = max(pairs, key=lambda p: p[1])[0] if pairs else None
    if target == 'detected' and result['source'].startswith('synthetic:'):
        window = [r for r in rows if r['visible'] and r['truth_range_m'] is not None
                  and diag['min_range_m'] <= r['truth_range_m'] <= diag['max_range_m']]
        missed = [r for r in window if not r['detected']]
        factors = Counter(f for r in missed for f in r['miss_factors'].split('; ') if f)
        diameters = [r['diameter_px_upper'] for r in window if r['diameter_px_upper'] is not None]
        factor = factors.most_common(1)[0][0] if factors else 'no local miss factor available'
        diameter = f'; median diameter upper bound {np.median(diameters):.1f} px' if diameters else ''
        return (f"clay detected on {sum(r['detected'] for r in window)}/{len(window)} in-window frames"
                f'; most common observed miss factor: {factor}{diameter}')
    eligible = [r for r in diag['sigma_rows'] if r['range_ok'] and r['elevation_ok']]
    if target == 'p_hit_ok' and eligible:
        best = max(eligible, key=lambda r: r['p_hit'])
        if best['p_hit'] < diag['min_hit_probability']:
            terms = {k: best[k] ** 2 for k in ('track', 'jitter', 'range')}
            dominant = max(terms, key=terms.get)
            share = 100 * terms[dominant] / sum(terms.values()) if sum(terms.values()) else 0
            return (f"P(hit) never reached {diag['min_hit_probability']:.2f}: best {best['p_hit']:.2f} "
                    f"at frame {best['frame']} (miss {math.degrees(best['miss']):.2f} deg, "
                    f"pattern {math.degrees(best['pattern']):.2f} deg, sigma {math.degrees(best['sigma']):.2f} deg; "
                    f"sigma dominated by {dominant} term {share:.0f} %)")
    descriptions = {
        'associated': 'detections did not consistently associate with a track',
        'confirmed': 'matched tracks were not selected and confirmed with enough hits',
        'range_in_window': 'confirmed tracks were outside the configured range window',
        'elevation_ok': 'the clay was below the minimum elevation',
        'p_hit_ok': 'hit probability was below the configured threshold',
        'best_moment': 'a later frame had a higher predicted hit probability',
        'fire': 'the actuator did not fire at eligible moments',
    }
    if target in descriptions:
        return f"{descriptions[target]} ({dict(pairs)[target]} same-frame losses)"
    return result['biggest_blocker']


def _print_explanation(result):
    print(f"{result['source']}: {result['frames']} frames")
    print('Funnel (same-frame cumulative; first frame zero-based):')
    for name, item in result['funnel'].items():
        print(f"  {name:18} {str(item['count']):>5}  first={item['first_frame']}")
    print(f"Actual actuator fires (including coasting tracks): {result['actual_fires']}")
    print('Decision reasons (p_hit min / p50 / max):')
    for reason, values in sorted(_reason_buckets(result['per_frame']).items(), key=lambda x: -len(x[1])):
        count = len(values)
        values = [v for v in values if v is not None]
        stats = (min(values), np.median(values), max(values)) if values else (None, None, None)
        formatted = ' / '.join(_format_number(v, 2) for v in stats)
        print(f"  {count:5}  {reason:32} {formatted}")
    print('Detection miss factors (overlapping):')
    for factor, count in sorted(result['miss_factors'].items(), key=lambda x: -x[1]):
        print(f'  {count:5}  {factor}')
    print('Timelines (min / p50 / max):')
    for key, label, precision in (('p_hit', 'P(hit)', 2), ('miss_deg', 'miss [deg]', 2),
                                  ('sigma_deg', 'sigma [deg]', 2), ('range_m', 'range [m]', 1)):
        stats = result['timelines'][key]
        print(f"  {label:12} " + ' / '.join(f"{_format_number(stats[k], precision):>6}"
                                                  for k in ('min', 'p50', 'max')))
    print(f"  best P(hit) frame: {result['timelines']['best_p_hit_frame']}")
    sigma_rows = result['_diagnostics']['sigma_rows']
    if sigma_rows:
        best = max(sigma_rows, key=lambda r: r['p_hit'])
        print(f"Sigma budget [deg] (confirmed tracks; min / p50 / best frame {best['frame']}; best share of sigma^2):")
        total = sum(best[k] ** 2 for k in ('track', 'jitter', 'range'))
        for key in ('track', 'jitter', 'range'):
            values = [math.degrees(r[key]) for r in sigma_rows]
            share = 100 * best[key] ** 2 / total if total else 0.0
            print(f"  {key:12} {min(values):6.2f} / {np.median(values):6.2f} / "
                  f"{math.degrees(best[key]):6.2f}  {share:5.1f} %")
        implied = _hit_probability(best['miss'], best['pattern'], best['sigma'])
        print(f"  best geometry [deg]: miss {math.degrees(best['miss']):.2f}, "
              f"pattern {math.degrees(best['pattern']):.2f}, sigma {math.degrees(best['sigma']):.2f}; "
              f"P(hit) {implied:.2f}")
    print(f"Biggest blocker: {_physical_blocker(result)}")
    print('Notes: funnel is truth-matched for synthetic; miss factors overlap and are not proven causes.')
    if result['source'].startswith('replay:'):
        print('Note: replay has no truth, is disarmed, and is not truth-matched.')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='tbot explain', description='Diagnose the pipeline funnel for one synthetic or replay run')
    add_common_args(parser)
    parser.set_defaults(source='synthetic:skeet_crossing')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--frames', type=int, default=None)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--csv', metavar='FILE')
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames < 1:
        parser.error('--frames must be >= 1')
    cfg, calib = load_from_args(args)
    result = explain_run(cfg, calib, args.source, args.seed, args.frames, diagnostics=not args.json)
    if args.csv:
        path = Path(args.csv).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as f:
            if result['per_frame']:
                writer = csv.DictWriter(f, fieldnames=list(result['per_frame'][0]))
                writer.writeheader()
                writer.writerows(result['per_frame'])
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        _print_explanation(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
