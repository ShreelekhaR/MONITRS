"""
Validate and repair event centers using article evidence, constrained to the
declared county.

FEMA tells us the county authoritatively. What it does not tell us is WHERE in
that county the event happened — and a county centroid is often the wrong
answer, since a fire or flood may sit at one edge. That placement is what the
LLM is for, and what article text can actually support.

So the rule is not "is this in the right state" but:

  1. Is the center inside, or plausibly adjacent to, the declared county?
     Near-border placements are kept: FIRMS thermal detections and good LLM
     estimates legitimately fall just outside a county polygon when the event
     straddles a line. A center 2,000 km away is broken.

  2. If broken, re-estimate from the harvested article text — the specific
     places the articles name — with the county boundary as a hard constraint.

  3. Only if that fails, fall back to the county centroid.

Usage:
    python pipeline/fix_centers.py --audit
    python pipeline/fix_centers.py --repair
    python pipeline/fix_centers.py --repair --max-km 40
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EVENTS_PATH = 'Data/events_processed.json'
HARVEST_DIR = 'Data/harvest'
CACHE_PATH = 'Data/county_geo.json'

# A center more than this far from its county's centroid is treated as broken.
# Generous: the largest US counties span a few hundred km, so this only catches
# gross errors (wrong region, wrong hemisphere), never near-border placements.
DEFAULT_MAX_KM = 150.0

STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'DC': 'District of Columbia', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana',
    'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
    'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan',
    'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana',
    'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina',
    'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
    'PR': 'Puerto Rico', 'VI': 'US Virgin Islands', 'GU': 'Guam',
    'MP': 'Northern Mariana Islands', 'AS': 'American Samoa',
}


def haversine_km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp = math.radians(la2 - la1)
    dl = math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def clean_county(raw):
    """'Fajardo (Municipio)' -> ('Fajardo', 'Municipio')"""
    if not raw:
        return '', 'County'
    m = re.search(r'\s*\(([^)]+)\)\s*$', raw.strip())
    if m:
        return re.sub(r'\s*\([^)]+\)\s*$', '', raw).strip(), m.group(1).strip()
    return raw.strip(), 'County'


def _nominatim(params, cache_key, cache):
    if cache_key in cache:
        return cache[cache_key]
    import requests
    endpoints = []
    nom = os.environ.get('NOMINATIM_URL')
    if nom:
        endpoints.append(f'{nom}/search')
    endpoints.append('https://nominatim.openstreetmap.org/search')
    for url in endpoints:
        try:
            if 'openstreetmap.org' in url:
                time.sleep(1.1)
            r = requests.get(url, params=params,
                             headers={'User-Agent': 'MONITRS/2.0 (research)'},
                             timeout=15)
            if r.status_code == 200 and r.json():
                cache[cache_key] = r.json()[0]
                return cache[cache_key]
        except Exception:
            continue
    cache[cache_key] = None
    return None


def county_geo(county_raw, state, cache):
    """Return (centroid, bbox) for the declared county.

    bbox = (south, north, west, east); used to constrain re-estimation.
    """
    bare, desig = clean_county(county_raw)
    state_full = STATE_NAMES.get((state or '').upper(), state or '')
    key = f'county|{bare}|{desig}|{state}'
    r = _nominatim({'q': f'{bare} {desig}, {state_full}, USA',
                    'format': 'json', 'limit': 1}, key, cache)
    if not r:
        return None, None
    centroid = (float(r['lat']), float(r['lon']))
    bb = r.get('boundingbox')
    bbox = tuple(float(x) for x in bb) if bb and len(bb) == 4 else None
    return centroid, bbox


def county_geo_cached(county_raw, state, cache):
    """county_geo without the network — a cache miss returns (None, None).

    Analysis scripts sweep every event, so a cold cache would mean thousands of
    throttled Nominatim calls. They want "unknown" for what isn't cached, not
    an hours-long geocoding run they never asked for.
    """
    bare, desig = clean_county(county_raw)
    r = cache.get(f'county|{bare}|{desig}|{state}')
    if not r:
        return None, None
    centroid = (float(r['lat']), float(r['lon']))
    bb = r.get('boundingbox')
    bbox = tuple(float(x) for x in bb) if bb and len(bb) == 4 else None
    return centroid, bbox


# Nominatim sometimes returns a node rather than a polygon, giving a bbox of
# zero area. Arecibo Municipio, PR comes back 0.00 x 0.00 deg. Treating that as
# the county boundary rejects every point on earth, so it has to be detected
# rather than trusted.
MIN_BBOX_DEG = 0.01

# A bbox is a rectangle, so it is already larger than the county inside it.
# Being outside it is therefore strong evidence, but FIRMS detections for an
# event on the county line can sit just beyond — allow a margin (~5 km).
BBOX_MARGIN_DEG = 0.05


def bbox_usable(bbox):
    """False for a missing or degenerate bbox, which cannot bound anything."""
    if not bbox:
        return False
    s, n, w, e = bbox
    return (n - s) >= MIN_BBOX_DEG and (e - w) >= MIN_BBOX_DEG


def in_county(center, centroid, bbox, max_km=DEFAULT_MAX_KM,
              margin=BBOX_MARGIN_DEG):
    """Is this center plausibly inside its declared county?

    Prefers the bbox: a center can sit 70 km from the centroid and still be in
    a large county, or 70 km away and squarely in the next county over, and
    only the boundary distinguishes those. Falls back to distance when the
    bbox is degenerate.
    """
    if not center or not centroid:
        return None
    if bbox_usable(bbox):
        s, n, w, e = bbox
        return (s - margin <= center[0] <= n + margin and
                w - margin <= center[1] <= e + margin)
    return haversine_km(tuple(center[:2]), centroid) <= max_km


def place_in_county(name, county_raw, state, cache, bbox, centroid=None):
    """Geocode a named place, requiring it to sit inside the county."""
    bare, _ = clean_county(county_raw)
    state_full = STATE_NAMES.get((state or '').upper(), state or '')
    for q in (f'{name}, {bare} County, {state_full}',
              f'{name}, {state_full}'):
        r = _nominatim({'q': q, 'format': 'json', 'limit': 1},
                       f'place|{q}', cache)
        if not r:
            continue
        lat, lon = float(r['lat']), float(r['lon'])
        if bbox_usable(bbox):
            s, n, w, e = bbox
            if not (s <= lat <= n and w <= lon <= e):
                continue          # outside the declared county — reject
        elif centroid is not None:
            # No usable boundary: keep the distance constraint rather than
            # accepting a same-named place in another state.
            if haversine_km((lat, lon), centroid) > DEFAULT_MAX_KM:
                continue
        return lat, lon
    return None, None


def article_places(eid, harvest_dir=HARVEST_DIR):
    """Named places from verified articles, most-mentioned first."""
    p = os.path.join(harvest_dir, f'{eid}.json')
    if not os.path.exists(p):
        return []
    try:
        rec = json.load(open(p))
    except Exception:
        return []
    counts = Counter()
    for f in rec.get('facts', []):
        if not f.get('is_about_target_event'):
            continue
        for name in f.get('affected_features') or []:
            if isinstance(name, str) and len(name.strip()) > 2:
                counts[name.strip()] += 1
    return [n for n, _ in counts.most_common(12)]


def repair_center(eid, ev, cache, max_km):
    """Return (new_center, method) or (None, reason)."""
    centroid, bbox = county_geo(ev.get('county'), ev.get('state'), cache)
    if centroid is None:
        return None, 'county not geocodable'

    # 1. Article evidence, constrained to the county
    places = article_places(eid)
    hits = []
    for name in places:
        lat, lon = place_in_county(name, ev.get('county'), ev.get('state'),
                                   cache, bbox, centroid)
        if lat is not None:
            hits.append((name, lat, lon))
        if len(hits) >= 4:
            break
    if hits:
        lat = sum(h[1] for h in hits) / len(hits)
        lon = sum(h[2] for h in hits) / len(hits)
        names = ', '.join(h[0] for h in hits)
        return [lat, lon], f'article places in county ({names})'

    # 2. Fall back to the county centroid — correct county, but no evidence
    #    about where inside it, so the chip may miss the event.
    return list(centroid), 'county centroid (no article evidence)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--cache', default=CACHE_PATH)
    ap.add_argument('--harvest-dir', default=HARVEST_DIR)
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--repair', action='store_true')
    ap.add_argument('--max-km', type=float, default=DEFAULT_MAX_KM)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    if not (args.audit or args.repair):
        args.audit = True

    events = json.load(open(args.events))
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}

    print(f'Checking centers against declared county boundary '
          f'(fallback threshold {args.max_km:.0f} km where no bbox)\n')

    broken, ok, unresolved, no_bbox = [], 0, 0, 0
    checked = 0
    ids = [e for e, v in events.items() if 'error' not in v and v.get('center')]
    for i, eid in enumerate(ids, 1):
        v = events[eid]
        centroid, bbox = county_geo(v.get('county'), v.get('state'), cache)
        if centroid is None:
            unresolved += 1
            continue
        checked += 1
        d = haversine_km(tuple(v['center'][:2]), centroid)
        # Distance alone cannot decide this: 70 km from the centroid is well
        # inside Lincoln County NM and well outside Washington County OR. Use
        # the boundary where we have one.
        if not in_county(v['center'], centroid, bbox, args.max_km):
            broken.append((eid, d))
            if not bbox_usable(bbox):
                no_bbox += 1
        else:
            ok += 1
        if i % 200 == 0:
            json.dump(cache, open(args.cache, 'w'))
            print(f'  checked {i}/{len(ids)}  broken so far: {len(broken)}',
                  flush=True)
    json.dump(cache, open(args.cache, 'w'))

    print(f'\n{checked} centers checked')
    print(f'  inside declared county:    {ok} ({100*ok/max(1,checked):.1f}%)')
    print(f'  broken:                    {len(broken)} ({100*len(broken)/max(1,checked):.1f}%)')
    print(f'    judged by distance only: {no_bbox} (county bbox degenerate)')
    print(f'  county not geocodable:     {unresolved}')
    if broken:
        print(f'  by strategy: {dict(Counter(events[e].get("strategy") for e, _ in broken))}')
        print('\n  worst offenders:')
        for eid, d in sorted(broken, key=lambda x: -x[1])[:12]:
            v = events[eid]
            print(f'    ev{eid:<6} {v.get("state")} {str(v.get("county"))[:22]:<22} '
                  f'{d:>7.0f} km  {v.get("strategy")}')

    if args.audit and not args.repair:
        print('\nRe-run with --repair to re-estimate these from article evidence.')
        return

    targets = [e for e, _ in broken][:args.limit] if args.limit else [e for e, _ in broken]
    print(f'\nRepairing {len(targets)}...')
    from collections import Counter as C
    methods = C()
    for i, eid in enumerate(targets, 1):
        v = events[eid]
        new, method = repair_center(eid, v, cache, args.max_km)
        if new is None:
            methods['failed: ' + method] += 1
            continue
        v['center_original'] = v['center']
        v['center'] = new
        v['center_repair_method'] = method
        v['strategy'] = ('article_evidence' if method.startswith('article')
                         else 'county_centroid')
        methods['article evidence' if method.startswith('article')
                else 'centroid fallback'] += 1
        if i % 20 == 0:
            json.dump(cache, open(args.cache, 'w'))
            json.dump(events, open(args.events, 'w'), indent=2)
            print(f'  {i}/{len(targets)}', flush=True)

    json.dump(cache, open(args.cache, 'w'))
    json.dump(events, open(args.events, 'w'), indent=2)
    print(f'\n{dict(methods)}')
    print(f'wrote {args.events}; originals kept in center_original')
    print('\nNOTE: centroid-fallback events have the right county but no '
          'evidence of where inside it — the chip may miss the event. They are '
          'marked strategy=county_centroid and should be treated as lower '
          'confidence, or re-harvested for article evidence first.')
    if targets:
        print('\nImagery for repaired events points at the old location. '
              'Delete those Data/images/<id> folders and re-download.')


if __name__ == '__main__':
    main()
