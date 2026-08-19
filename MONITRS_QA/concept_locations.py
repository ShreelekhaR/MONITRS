"""
Query local Nominatim for concept-type POIs inside each event's bbox.

For each event, return a dict {concept_class: [(name, lat, lon), ...]}.
Cached to Data/concept_pois.json so re-runs are fast.

Concept classes are generic categories a satellite VLM should recognize
visually — no proper-noun knowledge needed.

Usage:
    from concept_locations import get_concept_pois_for_event
    pois = get_concept_pois_for_event(event_id, center, halfwidth)
"""

import json
import os
import requests
import time
from collections import defaultdict


NOMINATIM_URL = os.environ.get('NOMINATIM_URL', 'http://nominatim.geocoder.internal:8080')
CACHE_PATH = 'Data/concept_pois.json'

# Nominatim search terms → nice display name for the concept
CONCEPT_QUERIES = [
    ('airport', 'airport'),
    ('port', 'port'),
    ('harbor', 'harbor'),
    ('stadium', 'stadium'),
    ('park', 'park'),
    ('hospital', 'hospital'),
    ('school', 'school'),
    ('university', 'university'),
    ('golf course', 'golf course'),
    ('marina', 'marina'),
    ('reservoir', 'reservoir'),
    ('dam', 'dam'),
    ('power plant', 'power plant'),
    ('cemetery', 'cemetery'),
    ('shopping center', 'shopping center'),
]

_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    return _cache


def _save_cache():
    if _cache is not None:
        os.makedirs(os.path.dirname(CACHE_PATH) or '.', exist_ok=True)
        json.dump(_cache, open(CACHE_PATH, 'w'))


def query_nominatim_bounded(q, west, south, east, north, limit=5):
    """Nominatim /search bounded to a bbox. Returns list of {name, lat, lon, class}."""
    try:
        resp = requests.get(
            f'{NOMINATIM_URL}/search',
            params={
                'q': q,
                'format': 'json',
                'limit': limit,
                'viewbox': f'{west},{north},{east},{south}',  # left, top, right, bottom
                'bounded': 1,
            },
            headers={'User-Agent': 'MONITRS/2.0'},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        return [
            {
                'name': r.get('display_name', '').split(',')[0].strip(),
                'lat': float(r['lat']),
                'lon': float(r['lon']),
                'class': r.get('class', ''),
                'type': r.get('type', ''),
            }
            for r in data
        ]
    except Exception:
        return []


def get_concept_pois_for_event(event_id, center, halfwidth=0.05):
    """Return {concept: [(name, lat, lon), ...]} for POIs inside the bbox."""
    cache = _load_cache()
    key = f'{event_id}'
    if key in cache:
        return cache[key]

    center_lat, center_lon = center
    west  = center_lon - halfwidth
    east  = center_lon + halfwidth
    south = center_lat - halfwidth
    north = center_lat + halfwidth

    pois = {}
    for query, concept in CONCEPT_QUERIES:
        results = query_nominatim_bounded(query, west, south, east, north, limit=3)
        # Only keep results actually inside the bbox
        inside = []
        for r in results:
            if south <= r['lat'] <= north and west <= r['lon'] <= east:
                inside.append((r['name'], r['lat'], r['lon']))
        if inside:
            pois[concept] = inside
    cache[key] = pois
    _save_cache()
    return pois


def build_all_events(results_file='Data/events_processed.json'):
    """Precompute concept POIs for every event. Slow the first time."""
    results = json.load(open(results_file))
    total = len(results)
    with_pois = 0
    total_pois = 0
    for i, (eid, v) in enumerate(results.items()):
        if 'error' in v:
            continue
        center = v.get('center') or v.get('fema_center', [0, 0])
        hw = v.get('halfwidth', 0.05)
        pois = get_concept_pois_for_event(eid, center, hw)
        if pois:
            with_pois += 1
            total_pois += sum(len(v) for v in pois.values())
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] {with_pois} events with POIs, {total_pois} POIs total", flush=True)
    print(f"\nDone: {with_pois}/{total} events, {total_pois} POIs")


if __name__ == '__main__':
    build_all_events()
