"""
Align event-level fact timeseries to individual image acquisition dates.

The event timeseries is the source of truth. Sentinel-2 revisits every ~5 days,
so a frame rarely lands on an article date. Rather than asserting exact values,
each frame gets a BOUNDED slice of the timeseries:

    lower bound  = latest reported value at or before the frame date
    upper bound  = earliest reported value after the frame date
    phase        = pre-event / onset / during / post / recovery

For monotonic quantities (fire acreage) the bounds are strict: if 10,000 acres
were reported on the 9th and 14,473 on the 10th, a frame from the 9th shows at
least 10,000 and at most 14,473.

Writes Data/aligned_frames.json:
    { "<event_id>": { "frames": [ {date, path, phase, bounds, statements,
                                   features_in_chip}, ... ] } }

Usage:
    python pipeline/align_frames.py
    python pipeline/align_frames.py --event 453
"""

import argparse
import json
import os
import re
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta

HARVEST_DIR = 'Data/harvest'
IMAGES_DIR = 'Data/images'
OUT_PATH = 'Data/aligned_frames.json'

MONOTONIC_TYPES = {'Fire'}          # extent only grows during the event


def _d(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except Exception:
        return None


def event_images(eid):
    d = os.path.join(IMAGES_DIR, str(eid))
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(('.png', '.jpg')):
            m = re.search(r'(\d{4}-\d{2}-\d{2})', fn)
            if m:
                out.append({'date': m.group(1), 'path': os.path.join(d, fn)})
    return sorted(out, key=lambda x: x['date'])


def extent_timeseries(rec):
    """Local-scope extent figures, deduped by date, monotonic for fires."""
    by_date = {}
    for f in rec['facts']:
        if not f.get('is_about_target_event') or not f.get('extent_number'):
            continue
        d = f.get('extent_as_of_date') or f.get('pub_date')
        if not d or not _d(d):
            continue
        unit = f.get('extent_unit') or 'units'
        prev = by_date.get(d)
        # if multiple articles report the same date, take the larger figure
        if prev is None or f['extent_number'] > prev['value']:
            by_date[d] = {'date': d, 'value': float(f['extent_number']),
                          'unit': unit, 'contained': f.get('contained_pct'),
                          'source': f.get('domain'), 'url': f.get('url')}
    ts = [by_date[k] for k in sorted(by_date)]
    if (rec.get('type') or '') in MONOTONIC_TYPES:
        run = 0.0
        for r in ts:
            run = max(run, r['value'])
            r['value'] = run
    return ts


def containment_timeseries(rec):
    by_date = {}
    for f in rec['facts']:
        if not f.get('is_about_target_event'):
            continue
        c = f.get('contained_pct')
        if c is None:
            continue
        d = f.get('extent_as_of_date') or f.get('pub_date')
        if not d or not _d(d):
            continue
        by_date[d] = max(by_date.get(d, 0.0), float(c))
    return [{'date': k, 'value': by_date[k]} for k in sorted(by_date)]


def notable_dates(rec):
    """Most-agreed start / peak / contained dates across relevant articles."""
    from collections import Counter
    buckets = {}
    for f in rec['facts']:
        if not f.get('is_about_target_event'):
            continue
        for k, v in (f.get('notable_dates') or {}).items():
            if v and _d(v):
                buckets.setdefault(k, Counter())[v] += 1
    return {k: c.most_common(1)[0][0] for k, c in buckets.items()}


def chip_features(rec):
    seen = {}
    for f in rec['facts']:
        if not f.get('is_about_target_event'):
            continue
        for x in f.get('validated_features') or []:
            seen.setdefault(x['name'].lower(), x)
    return list(seen.values())


def classify_phase(frame_date, nd, fema_start, fema_end):
    """pre-event / onset / during / post / recovery relative to the event."""
    fd = _d(frame_date)
    start = _d(nd.get('start') or fema_start or '')
    end = _d(nd.get('contained') or fema_end or '')
    if not fd or not start:
        return 'unknown'
    if fd < start - timedelta(days=1):
        return 'pre-event'
    if fd <= start + timedelta(days=2):
        return 'onset'
    if end and fd <= end:
        return 'during'
    if end and fd <= end + timedelta(days=21):
        return 'post'
    return 'recovery'


def bound_at(ts, frame_date):
    """(lower, upper) timeseries entries bracketing this frame date."""
    if not ts:
        return None, None
    dates = [r['date'] for r in ts]
    i = bisect_right(dates, frame_date) - 1
    lower = ts[i] if i >= 0 else None
    j = bisect_left(dates, frame_date)
    # first entry strictly after the frame date
    while j < len(ts) and ts[j]['date'] <= frame_date:
        j += 1
    upper = ts[j] if j < len(ts) else None
    return lower, upper


def build_statements(rec, frame, phase, lower, upper, contain_lower, feats):
    """Human-readable, defensible claims about this specific frame."""
    etype = (rec.get('type') or 'event').lower()
    out = []

    if phase == 'pre-event':
        out.append(f'This frame predates the reported onset of the {etype}; '
                   f'it serves as a baseline.')
    elif phase == 'onset':
        out.append(f'This frame is within two days of the reported onset of the {etype}.')

    if lower:
        s = (f'By {lower["date"]}, the {etype} had affected at least '
             f'{lower["value"]:,.0f} {lower["unit"]}.')
        if upper:
            s += (f' By {upper["date"]} the reported figure was '
                  f'{upper["value"]:,.0f} {upper["unit"]}, so this frame falls '
                  f'between those two states.')
        out.append(s)
    elif upper and phase in ('pre-event', 'onset'):
        out.append(f'The first reported figure is {upper["value"]:,.0f} '
                   f'{upper["unit"]} on {upper["date"]}; this frame precedes it.')

    if contain_lower is not None:
        out.append(f'Containment had reached {contain_lower:.0f}% by this point.')

    if feats:
        names = ', '.join(f['name'] for f in feats[:5])
        out.append(f'Named features reported affected and located inside this '
                   f'image chip: {names}.')

    return out


def align_event(rec):
    eid = rec['event_id']
    imgs = event_images(eid)
    if not imgs:
        return None

    ts = extent_timeseries(rec)
    cts = containment_timeseries(rec)
    nd = notable_dates(rec)
    feats = chip_features(rec)

    frames = []
    for im in imgs:
        lower, upper = bound_at(ts, im['date'])
        cl, _ = bound_at(cts, im['date'])
        phase = classify_phase(im['date'], nd, rec.get('fema_start'), rec.get('fema_end'))
        frames.append({
            'date': im['date'],
            'path': im['path'],
            'phase': phase,
            'bounds': {
                'lower': lower,
                'upper': upper,
                'containment_at_least': cl['value'] if cl else None,
            },
            'statements': build_statements(rec, im, phase, lower, upper,
                                           cl['value'] if cl else None, feats),
        })

    return {
        'event_id': eid,
        'event': rec.get('event'),
        'type': rec.get('type'),
        'state': rec.get('state'),
        'county': rec.get('county'),
        'center': rec.get('center'),
        'halfwidth': rec.get('halfwidth', 0.05),
        'fema_start': rec.get('fema_start'),
        'fema_end': rec.get('fema_end'),
        'notable_dates': nd,
        'extent_timeseries': ts,
        'containment_timeseries': cts,
        'features_in_chip': feats,
        'n_relevant_articles': sum(1 for f in rec['facts']
                                   if f.get('is_about_target_event')),
        'coverage': rec.get('coverage', {}),
        'frames': frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--harvest-dir', default=HARVEST_DIR)
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--event', nargs='+', type=int, default=None)
    args = ap.parse_args()

    out = {}
    for fn in sorted(os.listdir(args.harvest_dir)):
        if not fn.endswith('.json'):
            continue
        eid = int(fn.split('.')[0])
        if args.event and eid not in args.event:
            continue
        rec = json.load(open(os.path.join(args.harvest_dir, fn)))
        aligned = align_event(rec)
        if aligned:
            out[str(eid)] = aligned

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    n_frames = sum(len(v['frames']) for v in out.values())
    with_bounds = sum(1 for v in out.values() for fr in v['frames']
                      if fr['bounds']['lower'] or fr['bounds']['upper'])
    from collections import Counter
    phases = Counter(fr['phase'] for v in out.values() for fr in v['frames'])

    print(f'Aligned {len(out)} events, {n_frames} frames -> {args.out}')
    print(f'  frames with numeric bounds: {with_bounds}/{n_frames}')
    print(f'  phases: {dict(phases)}')


if __name__ == '__main__':
    main()
