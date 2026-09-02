"""
Diagnose why two events ended up with byte-identical frames.

find_identical_frames.py says WHICH pairs collide. It cannot say why, and the
obvious explanation (same center) is not the only one:

  1. Same center. Two events genuinely image the same ground.
  2. Degenerate frame. A chip that is entirely nodata or entirely cloud
     encodes to the same bytes no matter where on earth it was taken, so two
     unrelated events collide. That is a quality-filter leak, not a
     coordinate bug — and it means the md5 check is partly measuring the
     wrong thing.
  3. Bad county geometry. If Nominatim returned the wrong polygon, the
     in-county test fails for both events in a county at once, which looks
     like two bad placements but is one bad lookup.

This prints the evidence for each: the two centers and how far apart they
are, the county bbox actually used, and the pixel statistics of the shared
frames.

Usage:
    python pipeline/diagnose_shared_frames.py
    python pipeline/diagnose_shared_frames.py --pair 3514 7070
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_centers import clean_county, county_geo_cached, haversine_km

EVENTS_PATH = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'
CACHE_PATH = 'Data/county_geo.json'


def frame_stats(path):
    """Return (mean, std, %pure-black, %pure-white) for an RGB chip."""
    try:
        import numpy as np
        from PIL import Image
        a = np.asarray(Image.open(path).convert('RGB')).astype('float32')
    except Exception as e:
        return None, str(e)
    black = float((a.max(axis=2) < 1.0).mean() * 100)
    white = float((a.min(axis=2) > 245.0).mean() * 100)
    return (float(a.mean()), float(a.std()), black, white), None


def describe_county(v, cache):
    raw, state = v.get('county'), v.get('state')
    bare, desig = clean_county(raw)
    centroid, bbox = county_geo_cached(raw, state, cache)
    if not centroid:
        return f'{raw}, {state}: NOT CACHED'
    s, n, w, e = bbox if bbox else (None,) * 4
    span = (f'{(n-s):.2f} x {(e-w):.2f} deg' if bbox else 'no bbox')
    c = v.get('center')
    d = haversine_km(tuple(c[:2]), centroid) if c else None
    inside = (bbox and s <= c[0] <= n and w <= c[1] <= e) if c else None
    return (f'{bare} {desig}, {state}\n'
            f'        query centroid ({centroid[0]:.4f}, {centroid[1]:.4f})  '
            f'bbox {span}\n'
            f'        center ({c[0]:.4f}, {c[1]:.4f})  '
            f'{d:.1f} km from centroid  inside={inside}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--images', default=IMAGES_DIR)
    ap.add_argument('--cache', default=CACHE_PATH)
    ap.add_argument('--pair', nargs=2, metavar=('A', 'B'),
                    help='Only diagnose this one pair')
    args = ap.parse_args()

    ev = json.load(open(args.events))
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}

    dirs = sorted(d for d in os.listdir(args.images)
                  if os.path.isdir(os.path.join(args.images, d)))
    hashes = defaultdict(list)
    for d in dirs:
        p = os.path.join(args.images, d)
        for f in os.listdir(p):
            if f.endswith(('.png', '.jpg')):
                try:
                    with open(os.path.join(p, f), 'rb') as fh:
                        hashes[hashlib.md5(fh.read()).hexdigest()].append((d, f))
                except Exception:
                    pass

    shared = {h: v for h, v in hashes.items() if len({d for d, _ in v}) > 1}
    pairs = defaultdict(list)
    for h, v in shared.items():
        eids = sorted({d for d, _ in v})
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                pairs[(eids[i], eids[j])].append((h, v))

    if args.pair:
        want = tuple(sorted(args.pair))
        pairs = {k: v for k, v in pairs.items() if tuple(sorted(k)) == want}
        if not pairs:
            print(f'no shared frames between {args.pair[0]} and {args.pair[1]}')
            return

    def eid_of(d):
        return d.replace('event_', '').replace('ev', '')

    for (d1, d2), frames in sorted(pairs.items()):
        v1, v2 = ev.get(eid_of(d1), {}), ev.get(eid_of(d2), {})
        c1, c2 = v1.get('center'), v2.get('center')
        km = haversine_km(tuple(c1[:2]), tuple(c2[:2])) if c1 and c2 else None
        print('=' * 70)
        print(f'{d1} / {d2}   {len(frames)} shared frame(s)')
        print(f'  centers {km:.2f} km apart' if km is not None
              else '  centers unavailable')
        for d, v in ((d1, v1), (d2, v2)):
            print(f'  {d}: [{v.get("type")}] {v.get("start_date")} '
                  f'{str(v.get("event"))[:44]}')
            print(f'        {describe_county(v, cache)}')

        print('  shared frames:')
        for h, occurrences in frames[:6]:
            names = ', '.join(f'{d}/{f}' for d, f in occurrences)
            st, err = frame_stats(os.path.join(args.images, *occurrences[0]))
            if err:
                print(f'    {h[:8]}  {names}  (unreadable: {err})')
                continue
            mean, std, black, white = st
            flag = ''
            if std < 4.0:
                flag = '  <-- FLAT: identical bytes are a hash collision, ' \
                       'not a shared location'
            elif black > 95 or white > 95:
                flag = '  <-- SATURATED: quality filter should have dropped this'
            print(f'    {h[:8]}  mean={mean:6.1f} std={std:6.2f} '
                  f'black={black:5.1f}% white={white:5.1f}%{flag}')
            print(f'              {names}')
    print('=' * 70)


if __name__ == '__main__':
    main()
