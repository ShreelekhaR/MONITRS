"""
End-to-end provenance visualization: articles -> facts -> imagery -> QA.

For each event, renders one page showing the full chain:

    1. SOURCES     which articles were verified relevant, with publication dates
    2. FACTS       extent timeseries + validated features (and what got filtered)
    3. IMAGERY     Sentinel-2 sequence with fact-derived annotations per date
    4. STATEMENTS  the grounded claims we can defend for each image
    5. QA          questions generated from those statements

Usage:
    python pipeline/visualize_provenance.py --event 5357
    python pipeline/visualize_provenance.py --all --out Data/provenance.html
"""

import argparse
import base64
import json
import os
import re
from bisect import bisect_right
from collections import defaultdict

HARVEST_DIR = 'Data/harvest'
IMAGES_DIR = 'Data/images'
OUT_HTML = 'Data/provenance.html'


def img_b64(path, max_bytes=900_000):
    try:
        if os.path.getsize(path) > max_bytes:
            try:
                from PIL import Image
                import io
                im = Image.open(path).convert('RGB')
                im.thumbnail((420, 420))
                buf = io.BytesIO()
                im.save(buf, format='JPEG', quality=78)
                return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass
        with open(path, 'rb') as f:
            b = base64.b64encode(f.read()).decode()
        ext = 'png' if path.endswith('.png') else 'jpeg'
        return f'data:image/{ext};base64,{b}'
    except Exception:
        return ''


def event_images(eid):
    for suffix in ['', '_firms', '_llm', '_fema']:
        d = os.path.join(IMAGES_DIR, f'{eid}{suffix}')
        if os.path.isdir(d):
            out = []
            for fn in sorted(os.listdir(d)):
                if fn.endswith(('.png', '.jpg')):
                    m = re.search(r'(\d{4}-\d{2}-\d{2})', fn)
                    if m:
                        out.append({'date': m.group(1), 'path': os.path.join(d, fn)})
            return sorted(out, key=lambda x: x['date'])
    return []


def build_timeseries(rec):
    """(date, value, unit, contained, domain) rows for local-scope extents."""
    rows = []
    for f in rec['facts']:
        if not f.get('is_about_target_event') or not f.get('extent_number'):
            continue
        d = f.get('extent_as_of_date') or f.get('pub_date')
        if not d:
            continue
        rows.append({'date': d, 'value': f['extent_number'], 'unit': f.get('extent_unit'),
                     'contained': f.get('contained_pct'), 'domain': f.get('domain'),
                     'url': f.get('url')})
    rows.sort(key=lambda r: r['date'])
    # monotonic running max for fires
    if (rec.get('type') or '').lower() == 'fire':
        run = 0
        for r in rows:
            run = max(run, r['value'])
            r['value'] = run
    return rows


def statement_for_image(img_date, ts, rec):
    """The strongest defensible claim about this image date."""
    if not ts:
        return None
    dates = [r['date'] for r in ts]
    i = bisect_right(dates, img_date) - 1
    if i < 0:
        first = ts[0]
        return (f"Before the first report. By {first['date']} the event had reached "
                f"{first['value']:,.0f} {first['unit']} — this image predates that.")
    r = ts[i]
    unit = r['unit'] or 'units'
    parts = [f"By {r['date']} (latest report on or before this image), the "
             f"{(rec.get('type') or 'event').lower()} had affected at least "
             f"{r['value']:,.0f} {unit}."]
    if r.get('contained') is not None:
        parts.append(f"Containment was {r['contained']:.0f}%.")
    if i + 1 < len(ts):
        nxt = ts[i + 1]
        parts.append(f"By {nxt['date']} it reached {nxt['value']:,.0f} {nxt['unit'] or unit}, "
                     f"so this image shows a state between those two figures.")
    return ' '.join(parts)


def render_event(rec):
    eid = rec['event_id']
    imgs = event_images(str(eid))
    ts = build_timeseries(rec)
    rel = [f for f in rec['facts'] if f.get('is_about_target_event')]

    feats, dropped = {}, {}
    for f in rel:
        for x in f.get('validated_features') or []:
            feats.setdefault(x['name'].lower(), x)
        for x in f.get('dropped_features') or []:
            if x.get('name') != '__geocoder__':
                dropped.setdefault(x['name'].lower(), x)

    h = [f'<section class="event"><h2>Event {eid} — {rec.get("event","")}</h2>',
         f'<div class="meta">{rec.get("type")} &middot; {rec.get("county")} County, '
         f'{rec.get("state")} &middot; FEMA window {rec.get("fema_start")} → {rec.get("fema_end")} '
         f'&middot; coverage {rec["coverage"]["score"]:.2f}</div>']

    # 1. SOURCES
    h.append('<h3>1. Sources <span class="sub">verified as reporting this specific event</span></h3>')
    h.append('<table><tr><th>date</th><th>domain</th><th>title</th><th>why relevant</th></tr>')
    for f in sorted(rel, key=lambda x: str(x.get('pub_date'))):
        h.append(f'<tr><td class="mono">{f.get("pub_date") or "?"}</td>'
                 f'<td class="mono">{f.get("domain","")}</td>'
                 f'<td><a href="{f.get("url","")}" target="_blank">{(f.get("title") or "")[:78]}</a></td>'
                 f'<td class="dim">{(f.get("relevance_reason") or "")[:95]}</td></tr>')
    h.append('</table>')

    n_rej = [f for f in rec['facts'] if f.get('is_about_target_event') is False]
    if n_rej:
        h.append(f'<details><summary>{len(n_rej)} articles rejected as not-this-event</summary><table>')
        for f in n_rej[:15]:
            h.append(f'<tr><td class="mono">{f.get("domain","")}</td>'
                     f'<td class="dim">{(f.get("relevance_reason") or "")[:110]}</td></tr>')
        h.append('</table></details>')

    # 2. FACTS
    h.append('<h3>2. Extracted facts</h3>')
    if ts:
        h.append('<table><tr><th>as-of date</th><th>extent</th><th>contained</th><th>source</th></tr>')
        for r in ts:
            c = f'{r["contained"]:.0f}%' if r.get('contained') is not None else ''
            h.append(f'<tr><td class="mono">{r["date"]}</td>'
                     f'<td class="num">{r["value"]:,.0f} {r["unit"] or ""}</td>'
                     f'<td class="mono">{c}</td><td class="mono dim">{r["domain"]}</td></tr>')
        h.append('</table>')
    else:
        h.append('<p class="dim">No local-scope extent figures found.</p>')

    scope_rej = [f for f in rec['facts'] if f.get('rejected_extent')]
    if scope_rej:
        h.append('<details><summary>Rejected — wrong geographic scope</summary><table>')
        for f in scope_rej:
            x = f['rejected_extent']
            h.append(f'<tr><td class="num">{x["value"]:,.0f} {x["unit"] or ""}</td>'
                     f'<td class="mono">scope={x["scope"]}</td>'
                     f'<td class="mono dim">{f.get("domain","")}</td></tr>')
        h.append('</table></details>')

    if feats:
        h.append('<p><b>Features inside image bbox:</b> ' +
                 ', '.join(f'<span class="chip">{v["name"]}</span>' for v in feats.values()) + '</p>')
    if dropped:
        h.append('<details><summary>' + str(len(dropped)) +
                 ' features dropped (outside bbox)</summary><p class="dim">' +
                 ', '.join(f'{v["name"]} ({v.get("reason","")})' for v in dropped.values()) +
                 '</p></details>')

    # 3 + 4. IMAGERY with grounded statements
    h.append('<h3>3. Imagery <span class="sub">with the claim each frame supports</span></h3>')
    if not imgs:
        h.append('<p class="warn">No imagery downloaded for this event.</p>')
    else:
        h.append('<div class="strip">')
        for im in imgs:
            st = statement_for_image(im['date'], ts, rec)
            src = img_b64(im['path'])
            h.append(f'<figure><img src="{src}"><figcaption>'
                     f'<div class="date">{im["date"]}</div>'
                     f'<div class="stmt">{st or "&mdash;"}</div>'
                     f'</figcaption></figure>')
        h.append('</div>')

    h.append('</section>')
    return '\n'.join(h)


CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;
background:#f2f4f7;color:#1c2430;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:24px}
h1{font-size:26px;margin:0 0 4px}
.lead{color:#5b6780;margin:0 0 24px}
.event{background:#fff;border-radius:12px;padding:24px 28px;margin:18px 0;
box-shadow:0 1px 4px rgba(20,30,50,.08)}
h2{font-size:19px;margin:0 0 4px}
h3{font-size:14px;text-transform:uppercase;letter-spacing:.6px;color:#4a6fa5;
margin:22px 0 8px;border-bottom:1px solid #e4e8ef;padding-bottom:5px}
.sub{text-transform:none;letter-spacing:0;color:#8b96aa;font-weight:400}
.meta{color:#5b6780;font-size:13px;margin-bottom:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}
th{text-align:left;color:#5b6780;font-weight:600;border-bottom:1px solid #dfe4ec;padding:5px 8px}
td{padding:5px 8px;border-bottom:1px solid #f0f2f6;vertical-align:top}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}
.num{font-variant-numeric:tabular-nums;font-weight:600}
.dim{color:#8b96aa}
.warn{color:#a8500f;background:#fff5e8;padding:8px 12px;border-radius:6px}
.chip{display:inline-block;background:#e8f0fb;color:#28518c;padding:2px 9px;
border-radius:11px;font-size:12px;margin:2px 3px 2px 0}
details{margin:8px 0}
summary{cursor:pointer;color:#4a6fa5;font-size:12.5px}
.strip{display:flex;gap:12px;overflow-x:auto;padding:6px 0}
figure{margin:0;min-width:210px;max-width:210px}
figure img{width:210px;height:210px;object-fit:cover;border-radius:7px;border:1px solid #dde2ea}
figcaption{margin-top:5px}
.date{font-weight:600;font-size:12px;color:#b03030;font-variant-numeric:tabular-nums}
.stmt{font-size:10.5px;color:#42506a;margin-top:3px;line-height:1.4}
a{color:#28518c}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--harvest-dir', default=HARVEST_DIR)
    ap.add_argument('--out', default=OUT_HTML)
    ap.add_argument('--event', nargs='+', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()

    files = sorted(os.listdir(args.harvest_dir), key=lambda f: int(f.split('.')[0])
                   if f.split('.')[0].isdigit() else 0)
    recs = []
    for fn in files:
        if not fn.endswith('.json'):
            continue
        eid = int(fn.split('.')[0])
        if args.event and eid not in args.event:
            continue
        recs.append(json.load(open(os.path.join(args.harvest_dir, fn))))

    if not recs:
        print('No harvest records found'); return

    body = '\n'.join(render_event(r) for r in recs)
    mean_cov = sum(r['coverage']['score'] for r in recs) / len(recs)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>MONITRS provenance</title><style>{CSS}</style></head><body><div class="wrap">
<h1>MONITRS — article-to-imagery provenance</h1>
<p class="lead">{len(recs)} events &middot; mean coverage {mean_cov:.2f} &middot;
every claim traced back to a verified, date-stamped source</p>
{body}
</div></body></html>"""

    with open(args.out, 'w') as f:
        f.write(html)
    print(f'Wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB, {len(recs)} events)')


if __name__ == '__main__':
    main()
