"""
Explain the byte-identical frames that quality_checks reports across events.

A shared frame means two events downloaded the same chip: same center, same
date. That is not automatically a bug. Adjacent counties declared for one
disaster (Caldor across El Dorado/Amador, CZU across Santa Clara/Santa Cruz)
legitimately image the same ground on the same day.

It IS a bug when the two events sit in counties that do not touch, or when a
center falls outside its own declared county — then the chip is showing a
place the event did not happen.

This reports the pairs, whether each center is inside its own county, and how
far apart the two counties are, so the two cases can be told apart.

Usage:
    python pipeline/find_identical_frames.py
    python pipeline/find_identical_frames.py --show 30
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_centers import county_geo_cached, haversine_km, in_county

EVENTS_PATH = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'
CACHE_PATH = 'Data/county_geo.json'



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--images', default=IMAGES_DIR)
    ap.add_argument('--cache', default=CACHE_PATH)
    ap.add_argument('--show', type=int, default=20)
    args = ap.parse_args()

    ev = json.load(open(args.events))
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}

    dirs = sorted(d for d in os.listdir(args.images)
                  if os.path.isdir(os.path.join(args.images, d)))
    hashes = defaultdict(list)
    for d in dirs:
        p = os.path.join(args.images, d)
        for f in os.listdir(p):
            if not f.endswith(('.png', '.jpg')):
                continue
            try:
                with open(os.path.join(p, f), 'rb') as fh:
                    hashes[hashlib.md5(fh.read()).hexdigest()].append((d, f))
            except Exception:
                pass

    shared = {h: v for h, v in hashes.items()
              if len({d for d, _ in v}) > 1}
    print(f'{len(dirs)} event dirs, {sum(len(v) for v in hashes.values())} frames')
    print(f'{len(shared)} frames shared between events\n')
    if not shared:
        return

    # Collapse to event pairs
    pairs = defaultdict(list)
    for h, v in shared.items():
        eids = sorted({d for d, _ in v})
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                pairs[(eids[i], eids[j])].append(v[0][1])

    def eid_of(d):
        return d.replace('event_', '').replace('ev', '')

    legit, broken, unknown = [], [], []
    for (d1, d2), frames in pairs.items():
        v1 = ev.get(eid_of(d1), {})
        v2 = ev.get(eid_of(d2), {})
        c1, bb1 = county_geo_cached(v1.get('county'), v1.get('state'), cache)
        c2, bb2 = county_geo_cached(v2.get('county'), v2.get('state'), cache)
        ok1 = in_county(v1.get('center'), c1, bb1)
        ok2 = in_county(v2.get('center'), c2, bb2)
        # Distance between the two county centroids: adjacent counties are
        # tens of km apart, unrelated ones hundreds.
        county_km = haversine_km(c1, c2) if c1 and c2 else None
        row = (d1, d2, len(frames), v1, v2, ok1, ok2, county_km)
        if ok1 is None or ok2 is None:
            unknown.append(row)
        elif ok1 and ok2:
            legit.append(row)
        else:
            broken.append(row)

    print(f'event pairs sharing frames: {len(pairs)}')
    print(f'  both centers inside own county   {len(legit):>4}  '
          f'(adjacent counties, one disaster — expected)')
    print(f'  a center outside its own county  {len(broken):>4}  '
          f'(bad placement — chip shows the wrong place)')
    print(f'  county geometry unavailable      {len(unknown):>4}')

    def dump(title, rows):
        if not rows:
            return
        print(f'\n{title}:')
        for d1, d2, n, v1, v2, ok1, ok2, km in rows[:args.show]:
            gap = f'{km:.0f} km' if km is not None else '?'
            print(f'  {d1} / {d2}   {n} shared frame(s), counties {gap} apart')
            for d, v, ok in ((d1, v1, ok1), (d2, v2, ok2)):
                mark = 'in' if ok else ('OUT' if ok is False else '?')
                print(f'      [{mark:>3}] {v.get("state")} {v.get("county")} '
                      f'[{v.get("type")}] {v.get("start_date")} '
                      f'{str(v.get("event"))[:38]}')

    dump('bad placements', broken)
    dump('shared but legitimate (sample)', legit)

    far = [r for r in legit if r[7] is not None and r[7] > 150]
    if far:
        print(f'\n{len(far)} "legitimate" pairs have county centroids >150 km '
              f'apart — check these; adjacency is implausible at that range.')


if __name__ == '__main__':
    main()
