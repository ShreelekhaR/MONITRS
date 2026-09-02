"""
Find events whose chips image the same place.

Two events with the same center produce the same imagery, which means
duplicated training examples and — if they are different disasters —
mislabelled ones.

Being in different counties is NOT the test. One disaster is routinely
declared for several adjacent counties (Caldor for El Dorado and Amador, CZU
for Santa Clara and Santa Cruz), and a fire that straddles the county line
puts the nearest in-county hotspot for each within a few hundred metres of the
other. Both centers are correct.

The real test is whether each center sits inside ITS OWN declared county. A
center outside its county is imaging a place the event did not happen.

Usage:
    python pipeline/find_center_collisions.py
    python pipeline/find_center_collisions.py --min-km 1.0
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_centers import county_geo_cached, haversine_km

EVENTS_PATH = 'Data/events_processed.json'
CACHE_PATH = 'Data/county_geo.json'


def in_bbox(center, bbox):
    """True/False if the county polygon is known, None if it is not."""
    if not bbox or not center:
        return None
    s, n, w, e = bbox
    return s <= center[0] <= n and w <= center[1] <= e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--cache', default=CACHE_PATH)
    ap.add_argument('--min-km', type=float, default=1.0,
                    help='Centers closer than this count as colliding')
    ap.add_argument('--show', type=int, default=15)
    args = ap.parse_args()

    ev = json.load(open(args.events))
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    if not cache:
        print(f'WARNING: no county geometry cache at {args.cache}. Every pair '
              f'will land in "county geom unknown" and nothing can be judged.\n'
              f'Run this where fix_centers.py has already built the cache '
              f'(the workbench), or run fix_centers.py --audit first.\n')
    pts = [(k, v) for k, v in ev.items()
           if 'error' not in v and v.get('center')]

    # Bucket by rounded coordinate so we only compare nearby pairs
    buckets = defaultdict(list)
    for k, v in pts:
        la, lo = v['center'][0], v['center'][1]
        buckets[(round(la, 1), round(lo, 1))].append((k, v))

    same_county, adjacent, misplaced, unknown = [], [], [], []
    for _, group in buckets.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                (k1, v1), (k2, v2) = group[i], group[j]
                d = haversine_km(tuple(v1['center'][:2]),
                                 tuple(v2['center'][:2]))
                if d > args.min_km:
                    continue
                row = (k1, k2, round(d, 2), v1, v2)
                if (v1.get('state'), v1.get('county')) == \
                        (v2.get('state'), v2.get('county')):
                    same_county.append(row)
                    continue
                _, bb1 = county_geo_cached(v1.get('county'), v1.get('state'), cache)
                _, bb2 = county_geo_cached(v2.get('county'), v2.get('state'), cache)
                ok1 = in_bbox(v1.get('center'), bb1)
                ok2 = in_bbox(v2.get('center'), bb2)
                if ok1 is None or ok2 is None:
                    unknown.append(row)
                elif ok1 and ok2:
                    adjacent.append(row)
                else:
                    misplaced.append(row + (ok1, ok2))

    print(f'{len(pts)} events with centers\n')
    print(f'collisions within {args.min_km} km:')
    print(f'  same county            {len(same_county):>5}  '
          f'(one county, several disasters or years — expected)')
    print(f'  adjacent counties      {len(adjacent):>5}  '
          f'(both centers in their own county — expected)')
    print(f'  MISPLACED              {len(misplaced):>5}  '
          f'(a center falls outside its own county)')
    print(f'  county geom unknown    {len(unknown):>5}  '
          f'(not in {args.cache}; cannot judge)')

    if misplaced:
        print(f'\nmisplaced centers:')
        for k1, k2, d, v1, v2, ok1, ok2 in misplaced[:args.show]:
            print(f'  ev{k1} / ev{k2}   {d} km apart')
            for k, v, ok in ((k1, v1, ok1), (k2, v2, ok2)):
                mark = 'in ' if ok else 'OUT'
                print(f'      [{mark}] ev{k} {v.get("state")} '
                      f'{v.get("county")} [{v.get("type")}] '
                      f'{v.get("start_date")} strategy={v.get("strategy")}')

    if same_county:
        print(f'\nsame-county collisions (sample):')
        for k1, k2, d, v1, v2 in same_county[:6]:
            same_event = v1.get('event') == v2.get('event')
            tag = 'same declaration' if same_event else 'different disasters'
            print(f'  ev{k1} / ev{k2}  {v1.get("state")} {v1.get("county")} '
                  f'({tag})')
            print(f'      {v1.get("start_date")} {str(v1.get("event"))[:40]}')
            print(f'      {v2.get("start_date")} {str(v2.get("event"))[:40]}')

    print('\nNote: a shared footprint is only a duplicate FRAME when the dates '
          'also coincide. Run find_identical_frames.py against the downloaded '
          'imagery to see which collisions actually produced the same chip.')


if __name__ == '__main__':
    main()
