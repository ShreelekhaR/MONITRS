"""
Find events whose chips image the same place.

quality_checks flagged 76 byte-identical frames shared between events. Two
events with the same center produce the same imagery, which means duplicated
training examples and — if they are different disasters — mislabelled ones.

Legitimate cases exist: the same county can be declared for several disasters
across years, and neighbouring counties in one storm may genuinely overlap.
What is not legitimate is two events in DIFFERENT counties resolving to the
same point, which is a relocation failure.

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
from fix_centers import haversine_km

EVENTS_PATH = 'Data/events_processed.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--min-km', type=float, default=1.0,
                    help='Centers closer than this count as colliding')
    ap.add_argument('--show', type=int, default=15)
    args = ap.parse_args()

    ev = json.load(open(args.events))
    pts = [(k, v) for k, v in ev.items()
           if 'error' not in v and v.get('center')]

    # Bucket by rounded coordinate so we only compare nearby pairs
    buckets = defaultdict(list)
    for k, v in pts:
        la, lo = v['center'][0], v['center'][1]
        buckets[(round(la, 1), round(lo, 1))].append((k, v))

    same_county, diff_county = [], []
    for _, group in buckets.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                (k1, v1), (k2, v2) = group[i], group[j]
                d = haversine_km(tuple(v1['center'][:2]),
                                 tuple(v2['center'][:2]))
                if d > args.min_km:
                    continue
                c1 = (v1.get('state'), v1.get('county'))
                c2 = (v2.get('state'), v2.get('county'))
                (same_county if c1 == c2 else diff_county).append(
                    (k1, k2, round(d, 2), v1, v2))

    print(f'{len(pts)} events with centers\n')
    print(f'collisions within {args.min_km} km:')
    print(f'  same county      {len(same_county):>5}  '
          f'(expected — one county, several disasters or years)')
    print(f'  DIFFERENT county {len(diff_county):>5}  '
          f'(relocation failure — distinct places, same coordinate)')

    if diff_county:
        print(f'\ncross-county collisions:')
        for k1, k2, d, v1, v2 in diff_county[:args.show]:
            print(f'  ev{k1} / ev{k2}   {d} km apart')
            print(f'      {v1.get("state")} {v1.get("county")} '
                  f'[{v1.get("type")}] {v1.get("start_date")} '
                  f'strategy={v1.get("strategy")}')
            print(f'      {v2.get("state")} {v2.get("county")} '
                  f'[{v2.get("type")}] {v2.get("start_date")} '
                  f'strategy={v2.get("strategy")}')

    if same_county:
        print(f'\nsame-county collisions (sample):')
        for k1, k2, d, v1, v2 in same_county[:6]:
            same_event = v1.get('event') == v2.get('event')
            tag = 'same declaration' if same_event else 'different disasters'
            print(f'  ev{k1} / ev{k2}  {v1.get("state")} {v1.get("county")} '
                  f'({tag})')
            print(f'      {v1.get("start_date")} {str(v1.get("event"))[:40]}')
            print(f'      {v2.get("start_date")} {str(v2.get("event"))[:40]}')

    print('\nNote: same-county collisions across different dates are fine — '
          'the imagery differs even though the chip footprint matches. '
          'Identical FRAMES only occur when the dates also coincide.')


if __name__ == '__main__':
    main()
