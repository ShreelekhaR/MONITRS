"""
Provenance visualization from per-frame aligned records.

Renders, per event:
    1. SOURCES   verified articles (+ collapsible rejects with reasons)
    2. FACTS     extent/containment timeseries, in-chip features, filtered items
    3. FRAMES    each Sentinel-2 acquisition with its phase, numeric bounds,
                 and the defensible claims for that specific date

Usage:
    python pipeline/visualize_aligned.py
    python pipeline/visualize_aligned.py --event 453 5357
"""

import argparse
import base64
import json
import os

ALIGNED = 'Data/aligned_frames.json'
HARVEST_DIR = 'Data/harvest'
OUT_HTML = 'Data/provenance.html'

PHASE_COLOR = {
    'pre-event': '#6b7f99',
    'onset':     '#c9761d',
    'during':    '#b03030',
    'post':      '#7a5aa8',
    'recovery':  '#2f7d51',
    'unknown':   '#8b96aa',
}


def img_b64(path, box=380, quality=72):
    try:
        from PIL import Image
        import io
        im = Image.open(path).convert('RGB')
        im.thumbnail((box, box))
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=quality)
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        try:
            with open(path, 'rb') as f:
                return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
        except Exception:
            return ''


def fmt_bounds(b):
    lo, hi = b.get('lower'), b.get('upper')
    if lo and hi:
        return (f'<span class="num">&ge; {lo["value"]:,.0f}</span> {lo["unit"]} '
                f'<span class="dim">&rarr;</span> '
                f'<span class="num">&le; {hi["value"]:,.0f}</span> {hi["unit"]}')
    if lo:
        return f'<span class="num">&ge; {lo["value"]:,.0f}</span> {lo["unit"]}'
    if hi:
        return f'<span class="dim">before first report;</span> <span class="num">&le; {hi["value"]:,.0f}</span> {hi["unit"]}'
    return '<span class="dim">no numeric bound</span>'


def render_event(ev, harvest):
    eid = ev['event_id']
    cov = ev.get('coverage', {})
    h = [f'<section class="event"><h2>Event {eid} — {ev.get("event","")}</h2>',
         f'<div class="meta">{ev.get("type")} &middot; {ev.get("county")} County, '
         f'{ev.get("state")} &middot; FEMA {ev.get("fema_start")} → {ev.get("fema_end")} '
         f'&middot; {ev.get("n_relevant_articles")} verified articles '
         f'&middot; coverage {cov.get("score",0):.2f}</div>']

    # 1. sources
    rel = [f for f in harvest['facts'] if f.get('is_about_target_event')]
    rej = [f for f in harvest['facts'] if f.get('is_about_target_event') is False]
    h.append('<h3>1. Sources <span class="sub">verified as reporting this incident</span></h3>')
    h.append('<table><tr><th>date</th><th>domain</th><th>title</th><th>why accepted</th></tr>')
    for f in sorted(rel, key=lambda x: str(x.get('pub_date'))):
        h.append(f'<tr><td class="mono">{f.get("pub_date") or "?"}</td>'
                 f'<td class="mono">{f.get("domain","")}</td>'
                 f'<td><a href="{f.get("url","")}" target="_blank">{(f.get("title") or "")[:70]}</a></td>'
                 f'<td class="dim">{(f.get("relevance_reason") or "")[:88]}</td></tr>')
    h.append('</table>')
    if rej:
        h.append(f'<details><summary>{len(rej)} rejected — not this event</summary><table>')
        for f in rej[:18]:
            h.append(f'<tr><td class="mono">{f.get("domain","")}</td>'
                     f'<td class="dim">{(f.get("relevance_reason") or "")[:120]}</td></tr>')
        h.append('</table></details>')

    # 2. facts
    h.append('<h3>2. Fact timeseries <span class="sub">local scope only</span></h3>')
    ts = ev.get('extent_timeseries') or []
    if ts:
        h.append('<table><tr><th>as-of</th><th>extent</th><th>source</th></tr>')
        for r in ts:
            h.append(f'<tr><td class="mono">{r["date"]}</td>'
                     f'<td class="num">{r["value"]:,.0f} {r["unit"]}</td>'
                     f'<td class="mono dim">{r.get("source","")}</td></tr>')
        h.append('</table>')
    else:
        h.append('<p class="dim">No local-scope extent figures reported.</p>')

    nd = ev.get('notable_dates') or {}
    if any(nd.values()):
        h.append('<p>' + ' &middot; '.join(
            f'<b>{k}</b> <span class="mono">{v}</span>'
            for k, v in nd.items() if v) + '</p>')

    feats = ev.get('features_in_chip') or []
    if feats:
        h.append('<p><b>In-chip features:</b> ' + ' '.join(
            f'<span class="chip">{f["name"]}</span>' for f in feats) + '</p>')

    scope_rej = [f for f in harvest['facts'] if f.get('rejected_extent')]
    dropped = {}
    for f in harvest['facts']:
        for x in f.get('dropped_features') or []:
            if x.get('name') != '__geocoder__':
                dropped.setdefault(x['name'].lower(), x)
    if scope_rej or dropped:
        h.append('<details><summary>Filtered out</summary>')
        if scope_rej:
            h.append('<p class="dim"><b>Wrong scope:</b> ' + ', '.join(
                f'{f["rejected_extent"]["value"]:,.0f} {f["rejected_extent"]["unit"] or ""} '
                f'({f["rejected_extent"]["scope"]})' for f in scope_rej) + '</p>')
        if dropped:
            h.append('<p class="dim"><b>Outside chip:</b> ' + ', '.join(
                f'{v["name"]} [{v.get("reason","")}]' for v in list(dropped.values())[:25]) + '</p>')
        h.append('</details>')

    # 3. frames
    h.append('<h3>3. Frames <span class="sub">each acquisition with its bounded claim</span></h3>')
    h.append('<div class="strip">')
    for fr in ev['frames']:
        c = PHASE_COLOR.get(fr['phase'], '#8b96aa')
        stmts = ''.join(f'<li>{s}</li>' for s in fr['statements']) or '<li class="dim">&mdash;</li>'
        cont = fr['bounds'].get('containment_at_least')
        cont_html = (f'<div class="cont">containment &ge; {cont:.0f}%</div>'
                     if cont is not None else '')
        h.append(
            f'<figure>'
            f'<img src="{img_b64(fr["path"])}">'
            f'<figcaption>'
            f'<div class="date">{fr["date"]}</div>'
            f'<div class="phase" style="background:{c}">{fr["phase"]}</div>'
            f'<div class="bounds">{fmt_bounds(fr["bounds"])}</div>'
            f'{cont_html}'
            f'<ul class="stmts">{stmts}</ul>'
            f'</figcaption></figure>')
    h.append('</div></section>')
    return '\n'.join(h)


CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;
background:#eef1f5;color:#1b2431;line-height:1.5}
.wrap{max-width:1240px;margin:0 auto;padding:26px}
h1{font-size:25px;margin:0 0 4px}
.lead{color:#5b6780;margin:0 0 22px;font-size:14px}
.event{background:#fff;border-radius:12px;padding:24px 28px;margin:18px 0;
box-shadow:0 1px 5px rgba(20,30,50,.09)}
h2{font-size:18px;margin:0 0 4px}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.7px;color:#3f6699;
margin:22px 0 8px;border-bottom:1px solid #e4e8ef;padding-bottom:5px}
.sub{text-transform:none;letter-spacing:0;color:#93a0b4;font-weight:400}
.meta{color:#5b6780;font-size:12.5px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:6px 0}
th{text-align:left;color:#5b6780;font-weight:600;border-bottom:1px solid #dfe4ec;padding:5px 7px}
td{padding:5px 7px;border-bottom:1px solid #f1f3f7;vertical-align:top}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}
.num{font-variant-numeric:tabular-nums;font-weight:700;color:#1b2431}
.dim{color:#93a0b4}
.chip{display:inline-block;background:#e7effb;color:#24508d;padding:2px 9px;
border-radius:11px;font-size:11.5px;margin:2px 2px 2px 0}
details{margin:8px 0}
summary{cursor:pointer;color:#3f6699;font-size:12px}
.strip{display:flex;gap:14px;overflow-x:auto;padding:8px 2px 4px}
figure{margin:0;min-width:230px;max-width:230px}
figure img{width:230px;height:230px;object-fit:cover;border-radius:8px;border:1px solid #dbe1e9}
figcaption{margin-top:6px}
.date{font-weight:700;font-size:12.5px;font-variant-numeric:tabular-nums}
.phase{display:inline-block;color:#fff;font-size:10px;text-transform:uppercase;
letter-spacing:.5px;padding:1px 8px;border-radius:9px;margin:3px 0}
.bounds{font-size:11.5px;margin-top:3px}
.cont{font-size:11px;color:#5b6780;margin-top:2px}
.stmts{margin:5px 0 0;padding-left:15px;font-size:10.5px;color:#42506a;line-height:1.45}
.stmts li{margin-bottom:3px}
a{color:#24508d}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aligned', default=ALIGNED)
    ap.add_argument('--harvest-dir', default=HARVEST_DIR)
    ap.add_argument('--out', default=OUT_HTML)
    ap.add_argument('--event', nargs='+', type=int, default=None)
    args = ap.parse_args()

    aligned = json.load(open(args.aligned))
    keys = sorted(aligned, key=lambda k: int(k))
    if args.event:
        keys = [k for k in keys if int(k) in args.event]

    parts = []
    for k in keys:
        hp = os.path.join(args.harvest_dir, f'{k}.json')
        if not os.path.exists(hp):
            continue
        parts.append(render_event(aligned[k], json.load(open(hp))))

    n_frames = sum(len(aligned[k]['frames']) for k in keys)
    mean_cov = (sum(aligned[k].get('coverage', {}).get('score', 0) for k in keys)
                / max(1, len(keys)))
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>MONITRS — article-to-imagery alignment</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>MONITRS — article-to-imagery alignment</h1>
<p class="lead">{len(keys)} events &middot; {n_frames} frames &middot; mean coverage
{mean_cov:.2f}. Event-level fact timeseries sliced per acquisition date; every
claim bounded and traceable to a dated, verified source.</p>
{''.join(parts)}
</div></body></html>"""

    with open(args.out, 'w') as f:
        f.write(html)
    print(f'Wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB, '
          f'{len(keys)} events, {n_frames} frames)')


if __name__ == '__main__':
    main()
