"""
Check whether downloaded chips still match the centers they are labelled with.

Relocation moved 3,139 event centers. Chips downloaded before that point were
fetched at the OLD coordinate, and nothing invalidated them — so an event can
have a correct center in events_processed.json and imagery of somewhere else
entirely. Such a chip is worse than a missing one: it trains the model to
associate a disaster with unrelated ground.

The signature is two events sharing byte-identical frames while their current
centers are far apart (3514/7070 are 695 km apart and share 5 frames). If
their pre-relocation centers coincide, the chips are stale.

Usage:
    python pipeline/check_imagery_staleness.py
    python pipeline/check_imagery_staleness.py --pair 3514 7070
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_centers import haversine_km

EVENTS_PATH = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--images', default=IMAGES_DIR)
    ap.add_argument('--pair', nargs=2, metavar=('A', 'B'))
    args = ap.parse_args()

    ev = json.load(open(args.events))
    dirs = sorted(d for d in os.listdir(args.images)
                  if os.path.isdir(os.path.join(args.images, d)))

    def eid_of(d):
        return d.replace('event_', '').replace('ev', '')

    # ---- 1. How much imagery belongs to a relocated event? ----
    with_img = [d for d in dirs
                if any(f.endswith(('.png', '.jpg'))
                       for f in os.listdir(os.path.join(args.images, d)))]
    reloc, moved_far, not_reloc, missing = [], [], [], []
    for d in with_img:
        v = ev.get(eid_of(d))
        if not v:
            missing.append(d)
            continue
        o = v.get('center_original')
        if not o:
            not_reloc.append(d)
            continue
        reloc.append(d)
        km = haversine_km(tuple(o[:2]), tuple(v['center'][:2]))
        if km > 1.0:
            moved_far.append((d, km))

    n_any_reloc = sum(1 for v in ev.values()
                      if isinstance(v, dict) and v.get('center_original'))
    print(f'{len(ev)} events, {n_any_reloc} relocated')
    print(f'{len(with_img)} event dirs hold imagery')
    print(f'  center was relocated       {len(reloc):>5}')
    print(f'    and moved >1 km          {len(moved_far):>5}  '
          f'<-- chips are of the OLD place if downloaded before relocation')
    print(f'  center never moved         {len(not_reloc):>5}')
    print(f'  no event record            {len(missing):>5}')

    if moved_far:
        moved_far.sort(key=lambda x: -x[1])
        print(f'\n  furthest moved, with imagery on disk:')
        for d, km in moved_far[:10]:
            v = ev[eid_of(d)]
            n = len([f for f in os.listdir(os.path.join(args.images, d))
                     if f.endswith(('.png', '.jpg'))])
            print(f'    {d:<8} moved {km:>8.1f} km  {n:>3} frames  '
                  f'{v.get("state")} {v.get("county")}')

    # ---- 2. Do shared frames trace back to a shared ORIGINAL center? ----
    hashes = defaultdict(list)
    for d in with_img:
        p = os.path.join(args.images, d)
        for f in os.listdir(p):
            if f.endswith(('.png', '.jpg')):
                try:
                    with open(os.path.join(p, f), 'rb') as fh:
                        hashes[hashlib.md5(fh.read()).hexdigest()].append(d)
                except Exception:
                    pass
    pairs = set()
    for h, ds in hashes.items():
        u = sorted(set(ds))
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                pairs.add((u[i], u[j]))
    if args.pair:
        want = tuple(sorted(args.pair))
        pairs = {p for p in pairs if tuple(sorted(p)) == want}

    print(f'\nshared-frame pairs: {len(pairs)}')
    print('  comparing CURRENT centers vs the centers the chips were '
          'fetched at:\n')
    verdicts = defaultdict(int)
    for d1, d2 in sorted(pairs):
        v1, v2 = ev.get(eid_of(d1), {}), ev.get(eid_of(d2), {})
        c1, c2 = v1.get('center'), v2.get('center')
        o1 = v1.get('center_original') or c1
        o2 = v2.get('center_original') or c2
        now = haversine_km(tuple(c1[:2]), tuple(c2[:2])) if c1 and c2 else None
        then = haversine_km(tuple(o1[:2]), tuple(o2[:2])) if o1 and o2 else None
        if now is None or then is None:
            verdict = 'unknown'
        elif now < 1.0:
            verdict = 'same place then and now'
        elif then < 1.0:
            verdict = 'STALE — chips predate relocation'
        else:
            verdict = 'UNEXPLAINED — far apart both before and after'
        verdicts[verdict] += 1
        print(f'  {d1} / {d2}')
        print(f'      current centers  {now:>8.1f} km apart'
              if now is not None else '      current centers  ?')
        print(f'      original centers {then:>8.1f} km apart'
              if then is not None else '      original centers ?')
        print(f'      -> {verdict}')

    print('\nsummary:')
    for k, n in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f'  {n:>4}  {k}')
    if verdicts.get('STALE — chips predate relocation'):
        print('\nIf any pair is STALE, every relocated event with imagery '
              'needs its chips re-downloaded — not just the colliding ones.')


if __name__ == '__main__':
    main()
