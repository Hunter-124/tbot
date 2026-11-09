"""Self-contained, printable HITL report generated from analysis results."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

from tbot.hitl.analysis import FLASH_ROWS


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _fmt(value, places=2):
    value = _number(value)
    return '-' if value is None else f'{value:.{places}f}'


def _mean(stats):
    return (stats or {}).get('mean') if (stats or {}).get('n') else None


def _table(headers, rows):
    if not rows:
        return ''
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        f'<th>{_esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{_esc(v)}</td>' for v in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def _section(title, body, intro=''):
    if not body:
        return ''
    return f'<section><h2>{_esc(title)}</h2>{intro}{body}</section>'


def _svg_start(title, height):
    return [f'<svg viewBox="0 0 900 {height}" role="img" aria-label="{_esc(title)}" xmlns="http://www.w3.org/2000/svg">',
            f'<title>{_esc(title)}</title>']


def _text(x, y, text, anchor='start', cls='tick'):
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" class="{cls}">{_esc(text)}</text>'


def _line(x1, y1, x2, y2, cls='grid'):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="{cls}"/>'


def _ticks(lo, hi, target=5):
    """Return rounded axis bounds, tick values and their display precision."""
    raw = (hi - lo) / target
    exponent = math.floor(math.log10(raw))
    magnitude = 10 ** exponent
    fraction = raw / magnitude
    step = next(n for n in (1, 2, 5, 10) if n >= fraction) * magnitude
    first = math.floor(lo / step)
    last = math.ceil(hi / step)
    values = [i * step for i in range(first, last + 1)]
    return first * step, last * step, values, max(0, -exponent)


def _tick_label(value, precision):
    return f'{value:.{precision}f}'


def _waterfall(rep):
    stats = (rep.get('flash') or {}).get('stats') or {}
    stages = [('photon_to_exposure_end_ms', 'Photon → exposure end'),
              ('exposure_end_to_user_ms', 'Exposure end → user'),
              ('user_to_pipeline_ms', 'User → pipeline'), ('detect_ms', 'Detection'),
              ('track_wait_ms', 'Detection → decision frame'),
              ('decision_to_cmd_ms', 'Decision → command')]
    act = 'cmd_to_break_ms' if _number(_mean(stats.get('cmd_to_break_ms'))) is not None else 'cmd_to_click_ms'
    stages.append((act, 'Command → break' if act == 'cmd_to_break_ms' else 'Command → click'))
    display = _number(_mean(stats.get('display_ms')))
    bars = []
    if display is not None:
        bars.append(('Draw → photon (display)', -display, 0, 'display'))
    current = 0.0
    for key, label in stages:
        v = _number(_mean(stats.get(key)))
        if v is None:
            break  # A missing interval prevents positioning all subsequent intervals.
        bars.append((label, current, current + v, 'stage'))
        current += v
    if not bars:
        return ''
    budget = _number((rep.get('budget') or {}).get('fire_delay_ms'))
    # The budget begins at the decision, not at the actuator command.
    decision = next((start for label, start, _, _ in bars if label == 'Decision → command'), None)
    if budget is not None and decision is not None:
        bars.append(('Budgeted fire delay (decision → exit)', decision, decision + budget, 'budget'))
    lo = min(0, *(min(a, b) for _, a, b, _ in bars))
    hi = max(1, *(max(a, b) for _, a, b, _ in bars))
    lo, hi, ticks, precision = _ticks(lo, hi)
    # Reserve room outside the plot for the longest value, including units.
    label_width = max(len(_fmt(b-a) + ' ms') for _, a, b, _ in bars) * 8
    left, right = 255, 890 - label_width
    scale = lambda x: left + (x - lo) / (hi - lo) * (right - left)
    height = 105 + 34 * len(bars)
    axis_y = height - 48
    svg = _svg_start('Latency waterfall, milliseconds relative to photon', height)
    svg.extend(['<rect x="255" y="7" width="12" height="12" class="bar-display"/>',
                _text(273, 17, 'Display'), '<rect x="365" y="7" width="12" height="12" class="bar-stage"/>',
                _text(383, 17, 'Measured stages'), '<rect x="520" y="7" width="12" height="12" class="bar-budget"/>',
                _text(538, 17, 'Budgeted fire delay')])
    for v in ticks:
        x = scale(v)
        svg.extend([_line(x, 30, x, axis_y), _text(x, axis_y + 18, _tick_label(v, precision), 'middle')])
    svg.append(_line(scale(0), 25, scale(0), axis_y, 'zero'))
    for i, (label, a, b, kind) in enumerate(bars):
        y = 38 + i * 34
        svg.append(_text(left - 12, y + 13, label, 'end', 'label'))
        x1, x2 = scale(a), scale(b)
        svg.append(f'<rect x="{min(x1,x2):.2f}" y="{y}" width="{max(abs(x2-x1),1):.2f}" height="18" rx="3" class="bar-{kind}"/>')
        svg.append(_text(max(x1, x2) + 5, y + 13, _fmt(b-a) + ' ms'))
    svg.append(_text((left + right) / 2, height - 7, 'Time relative to photon (ms)', 'middle', 'axis-title'))
    svg.append('</svg>')
    return ''.join(svg) + '<p class="caption">Bars accumulate separate stage means; they are not a single trial. The budget runs from decision to muzzle exit; break and click are earlier endpoints. Intervals after a missing stage are omitted.</p>'


def _display_chart(rep):
    disp = rep.get('display') or {}
    samples = (disp.get('samples') or {}).get('rise') or []
    points = [(_number(s.get('y')), _number(s.get('d_ms'))) for s in samples]
    points = [(x, y) for x, y in points if x is not None and y is not None]
    if not points:
        return ''
    H = _number((rep.get('geometry') or {}).get('screen_h'))
    rise = disp.get('rise') or {}
    d0, scan = _number(rise.get('d0_ms')), _number(rise.get('scanout_ms'))
    has_fit = H is not None and H > 0 and d0 is not None and scan is not None and rise.get('n')
    xmin = min(0, *(p[0] for p in points))
    xmax = max([p[0] for p in points] + ([H] if H and H > 0 else [1]))
    values = [p[1] for p in points] + ([d0, d0 + scan] if has_fit else [])
    ymin, ymax = min(values), max(values)
    pad = max((ymax - ymin) * .15, 1.0)
    xmin, xmax, xticks, xprecision = _ticks(xmin, xmax)
    ymin, ymax, yticks, yprecision = _ticks(ymin - pad, ymax + pad)
    sx = lambda v: 75 + (v - xmin) / (xmax - xmin) * 790
    sy = lambda v: 290 - (v - ymin) / (ymax - ymin) * 240
    s = _svg_start('Display rise latency by screen row', 370)
    for v in xticks:
        x = sx(v)
        s += [_line(x, 50, x, 290), _text(x, 310, _tick_label(v, xprecision), 'middle')]
    for v in yticks:
        y = sy(v)
        s += [_line(75, y, 865, y), _text(65, y+4, _tick_label(v, yprecision), 'end')]
    if has_fit:
        s.append(_line(sx(0), sy(d0), sx(H), sy(d0+scan), 'fit'))
    for x, y in points:
        s.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" class="point"/>')
    s += [_text(470, 345, 'Screen row y (px)', 'middle', 'axis-title'),
          '<text transform="translate(17 170) rotate(-90)" text-anchor="middle" class="axis-title">Draw → photon (ms)</text>',
          '<circle cx="80" cy="19" r="4" class="point"/>', _text(91, 23, 'Rise samples'),
          *([_line(210, 19, 240, 19, 'fit'), _text(248, 23, 'Fitted scan-out')] if has_fit else []), '</svg>']
    return ''.join(s)


def _sweep_chart(sweep):
    points = []
    for trial in sweep.get('trials') or []:
        if not trial.get('decision'):
            continue
        result = trial.get('actual') or trial.get('pred') or {}
        x, y = _number(result.get('along_deg')), _number(result.get('cross_deg'))
        if x is not None and y is not None:
            points.append((x, y, result.get('hit'), _number(result.get('pattern_deg')), bool(trial.get('actual'))))
    if not points:
        return ''
    patterns = [p[3] for p in points if p[3] is not None and p[3] >= 0]
    radius = sum(patterns)/len(patterns) if patterns else None
    lim = max([abs(v) for p in points for v in p[:2]] + ([radius] if radius is not None else []) + [0.1]) * 1.15
    _, lim, ticks, precision = _ticks(-lim, lim, 4)
    # Equal scales preserve the geometry of the pattern circle.
    cx, cy, half = 450, 215, 160
    scale = half / lim
    s = _svg_start('Sweep true miss relative to pattern', 440)
    for v in ticks:
        x, y = cx + v*scale, cy - v*scale
        s += [_line(x, cy-half, x, cy+half), _line(cx-half, y, cx+half, y),
              _text(x, cy+half+19, _tick_label(v, precision), 'middle'), _text(cx-half-9, y+4, _tick_label(v, precision), 'end')]
    s += [_line(cx-half, cy, cx+half, cy, 'zero'), _line(cx, cy-half, cx, cy+half, 'zero')]
    if radius is not None:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{radius*scale:.2f}" class="pattern"/>')
    for x, y, hit, _, _ in points:
        cls = 'hit' if hit is True else 'miss' if hit is False else 'unknown'
        s.append(f'<circle cx="{cx+x*scale:.2f}" cy="{cy-y*scale:.2f}" r="5" class="{cls}"/>')
    s += [_text(cx, 421, 'Along-track miss (deg)', 'middle', 'axis-title'),
          '<text transform="translate(250 215) rotate(-90)" text-anchor="middle" class="axis-title">Cross-track miss (deg)</text>',
          '<circle cx="295" cy="20" r="5" class="hit"/>', _text(306, 24, 'Hit'),
          '<circle cx="365" cy="20" r="5" class="miss"/>', _text(376, 24, 'Miss'),
          '<circle cx="440" cy="20" r="5" class="unknown"/>', _text(451, 24, 'Unknown'),
          *(['<circle cx="540" cy="20" r="6" class="pattern"/>', _text(553, 24, 'Mean pattern radius')] if radius is not None else []), '</svg>']
    note = 'Actual exit timing where measured; otherwise solver-budget exit timing. Positions compare the shot with the true target trajectory.'
    return ''.join(s) + f'<p class="caption">{_esc(note)}</p>'


def write_html(rep: dict, path: Path) -> Path:
    """Write a standalone HTML visualization of an analyzed HITL run."""
    path = Path(path)
    meta, page = rep.get('meta') or {}, rep.get('page') or {}
    status = page.get('status') or {}
    def meta_item(label, value):
        return f'<div><span>{_esc(label)}</span><strong>{_esc(value)}</strong></div>'
    mode = 'LIVE' if meta.get('live') is True else 'Dry run' if meta.get('live') is False else '-'
    header = '<header><div class="eyebrow">TBOT / HARDWARE-IN-THE-LOOP</div><h1>Bench report</h1><p>Vision, timing and virtual shot evidence</p><div class="meta">' + ''.join([
        meta_item('Run date', meta.get('created') or '-'), meta_item('Source', meta.get('source') or '-'),
        meta_item('Actuator', meta.get('actuator') or '-'), meta_item('Mode', mode),
        meta_item('Page refresh', _fmt(page.get('refresh_hz'), 1) + ' Hz'),
        meta_item('Min sync RTT', _fmt(status.get('rtt_min_ms')) + ' ms')]) + '</div></header>'
    sections = []
    checks = [('Bench checks', rep.get('checks') or []), ('Simulation self-check', rep.get('sim_check') or [])]
    verdict = []
    for heading, entries in checks:
        if not entries:
            continue
        verdict.append(f'<h3>{_esc(heading)}</h3><div class="checks">')
        for c in entries:
            level = str(c.get('level') or 'INFO')
            cls = level.lower() if level in ('PASS', 'WARN', 'FAIL', 'INFO') else 'info'
            verdict.append(f'<div class="check"><span class="badge {cls}">{_esc(level)}</span><div><strong>{_esc(c.get("what") or "-")}</strong><p>{_esc(c.get("detail") or "-")}</p></div></div>')
        verdict.append('</div>')
    sections.append(_section('Verdict', ''.join(verdict)))
    sections.append(_section('Latency waterfall', _waterfall(rep)))
    flash = rep.get('flash') or {}
    stats = flash.get('stats') or {}
    rows = []
    for key, label in FLASH_ROWS:
        st = stats.get(key) or {}
        if st:
            rows.append([label] + [_fmt(st.get(k)) if st.get('n') else '-' for k in ('mean', 'std', 'p50', 'p95')] + [str(st.get('n', 0))])
    sections.append(_section('Flash stage statistics', _table(['Stage (ms unless noted)', 'Mean', 'Std', 'P50', 'P95', 'n'], rows)))
    sections.append(_section('Display rise latency', _display_chart(rep)))
    sweep = rep.get('sweep') or {}
    swrows = []
    for speed, group in (sweep.get('by_speed') or {}).items():
        swrows.append([speed, f'{group.get("hits", "-")}/{group.get("decisions", "-")}',
                       *[_fmt(_mean(group.get(k))) for k in ('miss_deg', 'pattern_deg', 'late_ms', 'cross_deg', 'exit_late_ms', 'pred_miss_deg')]])
    swbody = _sweep_chart(sweep) + _table(['Speed (deg/s)', 'Hits / decisions', 'Miss (deg)', 'Pattern (deg)', 'Late (ms)', 'Cross (deg)', 'Exit late (ms)', 'Pred miss (deg)'], swrows)
    sections.append(_section('Sweep accuracy', swbody))
    detect = rep.get('detect') or {}
    dtrows = []
    for r, group in (detect.get('by_range') or {}).items():
        dtrows.append([r, _fmt(group.get('detect_rate'), 3), _fmt(group.get('centroid_rms_px')),
                       _fmt(_mean(group.get('diameter_px'))), _fmt(_mean(group.get('size_range_m'))),
                       _fmt((group.get('size_range_m') or {}).get('std') if (group.get('size_range_m') or {}).get('n') else None),
                       _fmt(group.get('false_per_frame'), 3), _fmt(group.get('psf_fit_px'))])
    dtintro = ''
    if detect:
        dtintro = f'<p class="caption">Blank false detections/frame: {_esc(_fmt(detect.get("false_per_frame_blank"), 3))} · Configured PSF: {_esc(_fmt(detect.get("psf_px")))} px</p>'
        if 'psf_fit_px' in detect:
            dtintro += f'<p class="caption">Pooled PSF fit: {_esc(_fmt(detect.get("psf_fit_px")))} px</p>'
    sections.append(_section('Detection', _table(['Range (m)', 'Rate', 'Centroid RMS (px)', 'Diameter (px)', 'Size range mean (m)', 'Size range std (m)', 'False / frame', 'PSF fit (px)'], dtrows) or dtintro, dtintro if dtrows else ''))
    geo = rep.get('geometry') or {}
    grows = [[label, value] for label, value in [('Screen width (px)', _fmt(geo.get('screen_w'), 0)), ('Screen height (px)', _fmt(geo.get('screen_h'), 0)), ('Markers', _fmt(geo.get('n_points'), 0)), ('RMS error (px)', _fmt(geo.get('rms_px'))), ('Max error (px)', _fmt(geo.get('max_px')))] if value != '-']
    sections.append(_section('Geometry', _table(['Measurement', 'Value'], grows)))
    details = '<details><summary>Full JSON report</summary><pre>' + _esc(json.dumps(rep, indent=2, ensure_ascii=False, default=str)) + '</pre></details>'
    css = '''
:root{color-scheme:light;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#193147;background:#edf2f6}
*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:38px 32px 70px}header{background:#15374c;color:white;border-radius:12px;padding:32px 36px;margin-bottom:24px}h1{font-size:36px;letter-spacing:-.03em;margin:8px 0}header p{color:#c7d9e3;margin:0 0 26px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.13em;color:#9dd8df}.meta{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.meta span{display:block;color:#b3cbd8;font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}.meta strong{font-size:15px;overflow-wrap:anywhere}section,details{background:white;border:1px solid #dce5eb;border-radius:10px;padding:24px 28px;margin:18px 0;box-shadow:0 2px 8px #1a39420a}h2{font-size:21px;letter-spacing:-.02em;margin:0 0 18px}h3{font-size:14px;color:#526b7c;margin:20px 0 10px}.checks{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 22px}.check{display:flex;gap:10px;padding:10px 0;border-bottom:1px solid #edf1f3;font-size:14px;align-items:flex-start}.check strong{font-size:14px}.check p{color:#536b7a;margin:3px 0;line-height:1.45}.badge{font-size:10px;font-weight:800;letter-spacing:.05em;border-radius:4px;padding:4px 6px;min-width:43px;text-align:center}.pass{background:#d9f2e7;color:#116347}.warn{background:#fff0cf;color:#865500}.fail{background:#fde1df;color:#a32a25}.info{background:#e2edf7;color:#305f84}.caption{color:#617786;font-size:13px;line-height:1.5}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:9px 10px;border-bottom:1px solid #e7edf0;text-align:right;white-space:nowrap}th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#617786}td:first-child,th:first-child{text-align:left;white-space:normal}tbody tr:last-child td{border-bottom:0}svg{display:block;width:100%;height:auto;overflow:visible;font-family:inherit}.grid{stroke:#e5edf1;stroke-width:1}.zero{stroke:#8299a6;stroke-width:1.5}.fit{stroke:#dc8b39;stroke-width:2.5}.point{fill:#168493;opacity:.8}.bar-display{fill:#b7d6df}.bar-stage{fill:#168493}.bar-budget{fill:#e0a34a}.tick{fill:#607987;font-size:12px}.label{fill:#314e60;font-size:12px}.axis-title{fill:#314e60;font-size:13px;font-weight:600}.pattern{fill:#e2f1f4;fill-opacity:.3;stroke:#168493;stroke-width:2;stroke-dasharray:5 3}.hit{fill:#159464}.miss{fill:#d65d4c}.unknown{fill:#899aa4}details summary{cursor:pointer;font-weight:700}pre{overflow:auto;max-height:600px;background:#f4f7f9;padding:16px;font-size:12px} @media(max-width:680px){main{padding:16px}header{padding:24px}.meta{grid-template-columns:repeat(2,1fr)}.checks{grid-template-columns:1fr}section,details{padding:20px 16px}svg{min-width:600px}section{overflow-x:auto}}@media print{:root{background:white}main{padding:0;max-width:none}header{print-color-adjust:exact;-webkit-print-color-adjust:exact}section,details{box-shadow:none;break-inside:avoid}svg{max-height:430px}details:not([open]){display:none}}
'''
    doc = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TBOT HITL report</title><style>' + css + '</style></head><body><main>' + header + ''.join(sections) + details + '</main></body></html>'
    path.write_text(doc, encoding='utf-8')
    return path
