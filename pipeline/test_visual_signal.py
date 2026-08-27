"""
Test 1 — visual signal detection.

Does the disaster actually show up in the pixels? For each event, compare
pre-event frames against during/post frames and measure whether the expected
spectral change appears.

We only have RGB thumbnails (no NIR), so we use RGB proxies:

  greenness   G / (R+G+B)      vegetation proxy; drops after fire
  wetness     B / (R+G+B)      water proxy; rises with flooding
  brightness  mean luminance    rises with snow/ash, drops with burn scar
  texture     local std         drops when terrain is smoothed by snow/water

Per event we emit:
  signal_strength   magnitude of change in the expected channel (0..1)
  direction_match   does the change go the way the disaster type predicts
  verdict           strong / weak / none / CONTRADICTED

CONTRADICTED usually means a mis-centered chip or wrong geolocation, not a
subtle signature — worth surfacing as a data error.

This produces a LABEL, not a filter. Weak-signal events stay in the dataset:
they are exactly the cases no existing model handles, and stratifying results
by signal strength is how we show representation learning.

Usage:
    python pipeline/test_visual_signal.py
    python pipeline/test_visual_signal.py --event 5357 453
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

ALIGNED = 'Data/aligned_frames.json'
OUT_PATH = 'Data/visual_signal.json'

# Which channel should move, and in which direction, per disaster type.
EXPECTED = {
    'Fire':             [('greenness', 'down'), ('brightness', 'down')],
    'Flood':            [('wetness', 'up'), ('texture', 'down')],
    'Hurricane':        [('wetness', 'up'), ('greenness', 'down')],
    'Tropical Storm':   [('wetness', 'up')],
    'Coastal Storm':    [('wetness', 'up')],
    'Severe Storm':     [('greenness', 'down'), ('texture', 'down')],
    'Tornado':          [('greenness', 'down'), ('texture', 'down')],
    'Snowstorm':        [('brightness', 'up'), ('texture', 'down')],
    'Severe Ice Storm': [('brightness', 'up'), ('greenness', 'down')],
    'Winter Storm':     [('brightness', 'up'), ('texture', 'down')],
    'Landslide':        [('greenness', 'down'), ('texture', 'up')],
    'Earthquake':       [('texture', 'up')],
    'Volcanic Eruption':[('brightness', 'down'), ('greenness', 'down')],
}

# Types whose signature is TRANSIENT — visible for a day or two, then gone.
# Sentinel-2 revisits every ~5 days, so averaging post-event frames washes the
# spike out entirely. For these we look for a single extreme frame instead.
EPHEMERAL_TYPES = {
    'Snowstorm', 'Severe Ice Storm', 'Winter Storm',
    'Flood', 'Coastal Storm', 'Tropical Storm', 'Hurricane',
}

# Types whose signature PERSISTS (burn scar, debris path, landslide scarp)
PERSISTENT_TYPES = {
    'Fire', 'Landslide', 'Tornado', 'Severe Storm',
    'Earthquake', 'Volcanic Eruption',
}


def frame_metrics(path):
    """RGB-proxy indices for one frame."""
    try:
        a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    except Exception:
        return None
    if a.size == 0:
        return None

    lum = a.mean(axis=2)
    # Ignore no-data and saturated pixels so they don't dominate the means
    valid = (lum > 12) & (lum < 245)
    if valid.mean() < 0.35:
        return None

    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    total = np.clip(r + g + b, 1e-6, None)

    # local texture: std of a coarse block grid
    h, w = lum.shape
    bs = 16
    blocks = lum[:h // bs * bs, :w // bs * bs].reshape(h // bs, bs, w // bs, bs)
    block_means = blocks.mean(axis=(1, 3))

    return {
        'greenness':  float((g / total)[valid].mean()),
        'wetness':    float((b / total)[valid].mean()),
        'brightness': float(lum[valid].mean() / 255.0),
        'texture':    float(block_means.std() / 255.0),
        'valid_frac': float(valid.mean()),
    }


def analyze_event(ev):
    etype = ev.get('type', '')
    frames = ev.get('frames', [])

    pre, post = [], []
    for fr in frames:
        m = frame_metrics(fr['path'])
        if not m:
            continue
        entry = {'date': fr['date'], 'phase': fr['phase'], **m}
        if fr['phase'] == 'pre-event':
            pre.append(entry)
        elif fr['phase'] in ('onset', 'during', 'post'):
            post.append(entry)

    if not pre or not post:
        return {
            'event_id': ev['event_id'], 'type': etype,
            'verdict': 'insufficient_frames',
            'n_pre': len(pre), 'n_post': len(post),
        }

    def mean_of(rows, k):
        return float(np.mean([r[k] for r in rows]))

    ephemeral = etype in EPHEMERAL_TYPES
    expected = EXPECTED.get(etype, [])

    deltas, rel, peak_frame = {}, {}, {}
    for k in ['greenness', 'wetness', 'brightness', 'texture']:
        base = mean_of(pre, k)
        if ephemeral:
            # Transient signature: the strongest single post frame, not the
            # mean. Snow and standing water are gone within a day or two, and
            # Sentinel-2's ~5-day revisit means most post frames miss it
            # entirely — averaging them washes the spike out.
            want = dict(expected).get(k)
            if want == 'up':
                best = max(post, key=lambda r: r[k])
            elif want == 'down':
                best = min(post, key=lambda r: r[k])
            else:
                best = max(post, key=lambda r: abs(r[k] - base))
            val = best[k]
            peak_frame[k] = best['date']
        else:
            val = mean_of(post, k)
        deltas[k] = val - base
        rel[k] = (val - base) / max(abs(base), 1e-6)

    checks = []
    for chan, want in expected:
        d = deltas[chan]
        moved = 'up' if d > 0 else 'down'
        mag = min(abs(rel[chan]), 1.0)
        c = {'channel': chan, 'expected': want, 'observed': moved,
             'delta': round(d, 5), 'rel_change': round(rel[chan], 4),
             'match': moved == want, 'magnitude': round(mag, 4)}
        if chan in peak_frame:
            c['peak_frame'] = peak_frame[chan]
        checks.append(c)

    if not checks:
        verdict, strength = 'no_expectation', 0.0
    else:
        matched = [c for c in checks if c['match']]
        strength = max((c['magnitude'] for c in matched), default=0.0)
        strong_wrong = [c for c in checks
                        if not c['match'] and c['magnitude'] > 0.15]
        if not matched and strong_wrong:
            verdict = 'CONTRADICTED'
        elif strength >= 0.15:
            verdict = 'strong'
        elif strength >= 0.05:
            verdict = 'weak'
        else:
            verdict = 'none'

    # How close did we actually get to the event? An ephemeral signature is
    # only observable if a frame landed within a couple of days of onset.
    nearest_gap = None
    nd = ev.get('notable_dates') or {}
    onset = nd.get('start') or ev.get('fema_start')
    if onset:
        from datetime import datetime
        try:
            o = datetime.strptime(onset, '%Y-%m-%d')
            gaps = []
            for r in post:
                try:
                    gaps.append(abs((datetime.strptime(r['date'], '%Y-%m-%d') - o).days))
                except Exception:
                    pass
            nearest_gap = min(gaps) if gaps else None
        except Exception:
            pass

    if (ephemeral and verdict in ('none', 'weak')
            and nearest_gap is not None and nearest_gap > 3):
        verdict = 'unobserved_transient'

    return {
        'event_id': ev['event_id'], 'type': etype,
        'ephemeral': ephemeral,
        'n_pre': len(pre), 'n_post': len(post),
        'nearest_post_frame_gap_days': nearest_gap,
        'deltas': {k: round(v, 5) for k, v in deltas.items()},
        'rel_change': {k: round(v, 4) for k, v in rel.items()},
        'checks': checks,
        'signal_strength': round(strength, 4),
        'verdict': verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aligned', default=ALIGNED)
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--event', nargs='+', type=int, default=None)
    args = ap.parse_args()

    aligned = json.load(open(args.aligned))
    keys = sorted(aligned, key=lambda k: int(k))
    if args.event:
        keys = [k for k in keys if int(k) in args.event]

    results = {}
    for k in keys:
        results[k] = analyze_event(aligned[k])

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'{"event":>7} {"type":<18} {"pre/post":>9} {"strength":>9}  verdict')
    print('-' * 68)
    for k in keys:
        r = results[k]
        print(f'{r["event_id"]:>7} {str(r["type"])[:18]:<18} '
              f'{r.get("n_pre",0):>4}/{r.get("n_post",0):<4} '
              f'{r.get("signal_strength",0):>9.3f}  {r["verdict"]}')

    counts = defaultdict(int)
    for r in results.values():
        counts[r['verdict']] += 1
    print(f'\nverdicts: {dict(counts)}')

    by_type = defaultdict(list)
    for r in results.values():
        if 'signal_strength' in r:
            by_type[r['type']].append(r['signal_strength'])
    if by_type:
        print('\nmean signal strength by type:')
        for t, xs in sorted(by_type.items(), key=lambda x: -np.mean(x[1])):
            print(f'   {t:<20} {np.mean(xs):.3f}  (n={len(xs)})')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
