"""
Summarise how relocated centers were placed.

"Inside the county" is necessary but not sufficient — a model returning the
geographic middle every time passes bbox validation trivially while giving no
more information than a centroid. This separates event-meaningful placements
from centroid-equivalents.

Usage:
    python pipeline/check_relocation.py
    python pipeline/check_relocation.py --show 25
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_centers import county_geo, haversine_km

EVENTS_PATH = 'Data/events_processed.json'
CACHE_PATH = 'Data/county_geo.json'

# Notes that indicate no real placement was made
GENERIC = re.compile(
    r'geographic (center|centre)|county (center|centre)|'
    r'middle of the county|no basis|unable to determine|'
    r'^\s*$', re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--cache', default=CACHE_PATH)
    ap.add_argument('--show', type=int, default=12)
    args = ap.parse_args()

    ev = json.load(open(args.events))
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}

    relocated = {k: v for k, v in ev.items() if v.get('center_original')}
    if not relocated:
        print('no relocated events found (looking for center_original)')
        return

    print(f'{len(relocated)} events relocated\n')
    print('strategy:')
    for s, n in Counter(v.get('strategy') for v in relocated.values()).most_common():
        print(f'  {str(s):<22} {n:>5} ({100*n/len(relocated):.1f}%)')

    generic, specific = [], []
    for k, v in relocated.items():
        note = v.get('center_note') or ''
        (generic if GENERIC.search(note) else specific).append((k, v, note))

    print(f'\nplacement quality:')
    print(f'  named a specific place   {len(specific):>5} '
          f'({100*len(specific)/len(relocated):.1f}%)')
    print(f'  centroid-equivalent      {len(generic):>5} '
          f'({100*len(generic)/len(relocated):.1f}%)')

    # How far from the county centroid did placements land? A cluster at ~0
    # means the model is returning the middle regardless of what it says.
    if cache:
        dists = []
        for k, v, _ in specific:
            c, _bb = county_geo(v.get('county'), v.get('state'), cache)
            if c:
                dists.append(haversine_km(tuple(v['center'][:2]), c))
        if dists:
            dists.sort()
            near0 = sum(1 for d in dists if d < 2.0)
            print(f'\n  distance from county centroid (named placements, n={len(dists)}):')
            print(f'    median {dists[len(dists)//2]:.1f} km   '
                  f'max {dists[-1]:.1f} km')
            print(f'    within 2 km of centroid: {near0} '
                  f'({100*near0/len(dists):.0f}%) — these are centroids in '
                  f'all but name')

    print(f'\nby state, centroid-equivalent:')
    for s, n in Counter(v.get('state') for _, v, _ in generic).most_common(8):
        print(f'  {s}: {n}')

    print(f'\nsample named placements:')
    for k, v, note in specific[:args.show]:
        o, c = v['center_original'], v['center']
        print(f'  ev{k:<6} {v.get("state")} {str(v.get("county"))[:20]:<20} '
              f'({o[0]:.1f},{o[1]:.1f}) -> ({c[0]:.2f},{c[1]:.2f})')
        print(f'         {note[:78]}')

    if generic:
        print(f'\nsample centroid-equivalent:')
        for k, v, note in generic[:5]:
            print(f'  ev{k:<6} {v.get("state")} {str(v.get("county"))[:20]:<20} '
                  f'{note[:56]}')
        print(f'\n  These have the right county but no evidence of where inside '
              f'it.\n  Re-estimate from article evidence once harvested:\n'
              f'      python pipeline/fix_centers.py --repair')


if __name__ == '__main__':
    main()
