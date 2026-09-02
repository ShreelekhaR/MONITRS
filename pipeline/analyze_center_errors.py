"""
Check whether broken centers follow a systematic transformation.

Magnitudes of 12,000-15,000 km are near-antipodal, which suggests something
mechanical (swapped lat/lon, flipped sign) rather than the LLM naming the wrong
town. A swap or sign flip is recoverable arithmetically and does not need
re-geocoding.

Usage:
    python pipeline/analyze_center_errors.py
"""

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_centers import county_geo, haversine_km

EVENTS_PATH = 'Data/events_processed.json'
CACHE_PATH = 'Data/county_geo.json'
THRESHOLD_KM = 150.0


def main():
    if not os.path.exists(CACHE_PATH):
        print(f'missing {CACHE_PATH} — run: python pipeline/fix_centers.py --audit')
        return
    ev = json.load(open(EVENTS_PATH))
    cache = json.load(open(CACHE_PATH))

    patterns = Counter()
    examples = defaultdict(list)
    dist_buckets = Counter()
    by_state_none = Counter()
    checked = 0

    for k, v in ev.items():
        if 'error' in v or not v.get('center'):
            continue
        c, _ = county_geo(v.get('county'), v.get('state'), cache)
        if c is None:
            continue
        la, lo = v['center'][0], v['center'][1]
        d0 = haversine_km((la, lo), c)
        if d0 <= THRESHOLD_KM:
            continue
        checked += 1
        dist_buckets['<500' if d0 < 500 else
                     '500-2000' if d0 < 2000 else
                     '2000-8000' if d0 < 8000 else '>8000'] += 1

        cands = {
            'lonlat_swap': (lo, la),
            'lon_sign_flip': (la, -lo),
            'lat_sign_flip': (-la, lo),
            'both_sign_flip': (-la, -lo),
        }
        best, bestd = 'unrecoverable', d0
        for name, cand in cands.items():
            if abs(cand[0]) > 90 or abs(cand[1]) > 180:
                continue
            dd = haversine_km(cand, c)
            if dd < bestd and dd <= THRESHOLD_KM:
                best, bestd = name, dd
        patterns[best] += 1
        if best == 'unrecoverable':
            by_state_none[v.get('state')] += 1
        if len(examples[best]) < 4:
            examples[best].append(
                (k, v.get('state'), str(v.get('county'))[:20],
                 round(la, 2), round(lo, 2), round(d0), round(bestd)))

    print(f'{checked} centers further than {THRESHOLD_KM:.0f} km from their county\n')
    print('distance distribution:')
    for b in ['<500', '500-2000', '2000-8000', '>8000']:
        n = dist_buckets[b]
        print(f'  {b:>10} km: {n:>5} ({100*n/max(1,checked):.1f}%)')

    print('\nrecoverable by arithmetic transform:')
    for name, n in patterns.most_common():
        print(f'  {name:<16} {n:>5} ({100*n/max(1,checked):.1f}%)')
        for e in examples[name]:
            tail = f'-> {e[6]}km' if name != 'unrecoverable' else ''
            print(f'      ev{e[0]:<6} {e[1]} {e[2]:<20} '
                  f'({e[3]}, {e[4]}) {e[5]}km {tail}')

    if by_state_none:
        print('\nunrecoverable, by state:')
        for s, n in by_state_none.most_common(12):
            print(f'  {s}: {n}')

    rec = checked - patterns['unrecoverable']
    print(f'\n{rec} of {checked} ({100*rec/max(1,checked):.0f}%) fixable arithmetically; '
          f'{patterns["unrecoverable"]} need re-estimation from articles.')


if __name__ == '__main__':
    main()
