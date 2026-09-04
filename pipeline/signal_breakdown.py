"""
Break one event type's signal results apart by verdict and by covariate.

A category mean is close to useless here: CONTRADICTED events contribute
exactly 0.000 by construction, so a bimodal category -- half its events found
cleanly, half pointing the wrong way -- reports the same mean as a category
where nothing was found at all. Those need opposite fixes.

Usage:
    python pipeline/signal_breakdown.py --type Fire
    python pipeline/signal_breakdown.py --type Fire --placed
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

SIGNAL = 'Data/visual_signal.json'
EVENTS_PATH = 'Data/events_processed.json'


def _q(vals):
    v = sorted(vals)
    if not v:
        return ''
    def at(f):
        return v[min(len(v) - 1, int(f * len(v)))]
    return f'p10 {at(.10):.3f}  med {at(.50):.3f}  p90 {at(.90):.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signal', default=SIGNAL)
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--type', default='Fire')
    ap.add_argument('--placed', action='store_true',
                    help='Only events relocated onto a FIRMS cluster')
    ap.add_argument('--n', type=int, default=12)
    args = ap.parse_args()

    sig = json.load(open(args.signal))
    ev = json.load(open(args.events)) if os.path.exists(args.events) else {}

    rows = []
    for k, v in sig.items():
        if v.get('type') != args.type:
            continue
        meta = ev.get(k, {})
        if args.placed and meta.get('strategy') != 'firms_cluster':
            continue
        rows.append((k, v, meta))
    if not rows:
        print('nothing matched')
        return 1

    print(f'{args.type}: {len(rows)} events'
          f'{" (placed only)" if args.placed else ""}\n')

    verdicts = Counter(v.get('verdict') for _, v, _ in rows)
    for name, n in verdicts.most_common():
        s = [v.get('signal_strength', 0) for _, v, _ in rows
             if v.get('verdict') == name]
        print(f'  {name:<22} {n:>4}   {_q(s)}')

    scored = [(k, v, m) for k, v, m in rows
              if v.get('signal_strength') is not None
              and v.get('verdict') not in ('insufficient_frames',
                                           'no_expectation')]
    if not scored:
        print('\nnothing scored')
        return 0

    # Which channel and direction actually won, when anything did.
    chans = Counter()
    for _, v, _ in scored:
        for c in v.get('checks') or []:
            if c.get('match') and c.get('magnitude') == v.get('signal_strength'):
                chans[f'{c["channel"]} {c["expected"]} ({c.get("mode","?")})'] += 1
    if chans:
        print('\nwinning channel:')
        for name, n in chans.most_common():
            print(f'  {name:<26} {n:>4}')

    # Only for CONTRADICTED events: wrong_way_channel is recorded whenever
    # any check fails, and on a matched event that is just the runner-up.
    wrong = Counter(v['wrong_way_channel'] for _, v, _ in scored
                    if v.get('verdict') == 'CONTRADICTED'
                    and v.get('wrong_way_channel'))
    if wrong:
        print('\nchannel that moved the wrong way (CONTRADICTED events):')
        for name, n in wrong.most_common():
            print(f'  {name:<22} {n:>4}')

    # Does the signal track fire size? If big fires score and small ones do
    # not, the category mean is describing footprint, not measurement failure.
    buckets = defaultdict(list)
    for _, v, m in scored:
        fc = m.get('firms_cluster') or {}
        n = fc.get('n_detections')
        if n is None:
            continue
        b = ('   <50 dets' if n < 50 else '  50-499' if n < 500
             else ' 500-4999' if n < 5000 else '   5000+')
        buckets[b].append(v.get('signal_strength', 0))
    if buckets:
        print('\nstrength by fire size:')
        for b in sorted(buckets):
            a = buckets[b]
            print(f'  {b:<12} n={len(a):<4} mean={sum(a)/len(a):.3f}   {_q(a)}')

    # Pre-frame count gates the noise floor, and one pre frame caps a verdict
    # at 'weak' no matter how strong the change.
    pb = defaultdict(list)
    for _, v, _ in scored:
        npre = v.get('n_pre', 0)
        pb['1 pre frame' if npre <= 1 else '2-3 pre' if npre <= 3
           else '4+ pre'].append(v.get('signal_strength', 0))
    print('\nstrength by pre-event frame count:')
    for b in sorted(pb):
        a = pb[b]
        print(f'  {b:<12} n={len(a):<4} mean={sum(a)/len(a):.3f}   {_q(a)}')

    scored.sort(key=lambda r: -(r[1].get('signal_strength') or 0))
    def show(title, rs):
        print(f'\n{title}')
        for k, v, m in rs:
            fc = m.get('firms_cluster') or {}
            print(f'  ev{k:<6} {v.get("signal_strength", 0):.3f} '
                  f'{v.get("verdict",""):<13} '
                  f'@{v.get("scale_cells") or "-"}  '
                  f'{(v.get("area_frac") or 0):.0%} area  '
                  f'pre={v.get("n_pre")} post={v.get("n_post")}  '
                  f'{fc.get("n_detections","-")} dets  '
                  f'hw={m.get("halfwidth","-")}')
    show(f'strongest {args.n}:', scored[:args.n])
    show(f'weakest {args.n}:', scored[-args.n:])
    return 0


if __name__ == '__main__':
    sys.exit(main())
