"""
Spatial validation: geocode extracted feature names and keep only those that
fall inside the event's satellite image bbox.

Articles about a regional disaster name features hundreds of km away (e.g. a
Klamath County event whose articles mention Mount Jefferson, 200 miles north).
Those can't appear in the image chip, so they must not become WHERE questions.

Uses local Nominatim if available, falls back to geocode.maps.co.
Cache: Data/feature_geocode.json

Usage:
    from validate_features import validate_features
    kept, dropped = validate_features(feature_names, center, halfwidth, state)
"""

import json
import os
import threading
import time
import requests

NOMINATIM_URL = os.environ.get('NOMINATIM_URL', 'http://nominatim.geocoder.internal:8080')
GEOCODE_API_KEY = os.environ.get('GEOCODE_API_KEY', '')
CACHE_PATH = 'Data/feature_geocode.json'

# Public Nominatim asks for <= 1 request/second. Local instances have no limit.
PUBLIC_NOMINATIM = 'nominatim.openstreetmap.org'
_rate_lock = threading.Lock()
_last_public_call = [0.0]


def _throttle_public(url):
    if PUBLIC_NOMINATIM not in url:
        return
    with _rate_lock:
        delta = time.time() - _last_public_call[0]
        if delta < 1.1:
            time.sleep(1.1 - delta)
        _last_public_call[0] = time.time()

STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands',
}

_cache = None
_geocoder_alive = None


def geocoder_available():
    """Probe the geocoder once. Returns True if any backend answers.

    Without this, an unreachable geocoder makes every lookup fail and silently
    deletes all features instead of surfacing the outage.
    """
    global _geocoder_alive
    if _geocoder_alive is not None:
        return _geocoder_alive
    for url, extra in _endpoints():
        try:
            params = {'q': 'Chicago', 'format': 'json', 'limit': 1}
            params.update(extra)
            r = requests.get(url, params=params,
                             headers={'User-Agent': 'MONITRS/2.0'}, timeout=8)
            if r.status_code == 200 and r.json():
                _geocoder_alive = True
                return True
        except Exception:
            continue
    _geocoder_alive = False
    return False


def _endpoints():
    eps = [(f'{NOMINATIM_URL}/search', {})]
    if GEOCODE_API_KEY:
        eps.append(('https://geocode.maps.co/search', {'api_key': GEOCODE_API_KEY}))
    # Public Nominatim as last resort (rate limited to ~1 req/s)
    eps.append(('https://nominatim.openstreetmap.org/search', {}))
    return eps


def _load():
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    return _cache


def _save():
    if _cache is not None:
        os.makedirs(os.path.dirname(CACHE_PATH) or '.', exist_ok=True)
        json.dump(_cache, open(CACHE_PATH, 'w'))


def geocode(name, state_abbr='', county=''):
    """Return (lat, lon, bbox) where bbox is (south, north, west, east) or None."""
    cache = _load()
    state = STATE_NAMES.get((state_abbr or '').upper(), state_abbr or '')
    key = f'{name}|{county}|{state}'
    if key in cache:
        v = cache[key]
        if not v:
            return None, None, None
        if len(v) == 2:            # legacy cache entry, no bbox
            return v[0], v[1], None
        return v[0], v[1], v[2]

    queries = []
    if county and state:
        queries.append(f'{name}, {county} County, {state}')
    if state:
        queries.append(f'{name}, {state}')
    queries.append(name)

    endpoints = _endpoints()

    for q in queries:
        for url, extra in endpoints:
            try:
                _throttle_public(url)
                params = {'q': q, 'format': 'json', 'limit': 1}
                params.update(extra)
                r = requests.get(url, params=params,
                                 headers={'User-Agent': 'MONITRS/2.0 (research)'},
                                 timeout=15)
                if r.status_code != 200:
                    continue
                data = r.json()
                if data:
                    lat, lon = float(data[0]['lat']), float(data[0]['lon'])
                    bb = data[0].get('boundingbox')
                    bbox = [float(x) for x in bb] if bb and len(bb) == 4 else None
                    cache[key] = [lat, lon, bbox]
                    _save()
                    return lat, lon, bbox
            except Exception:
                continue

    cache[key] = None
    _save()
    return None, None, None


def _boxes_overlap(a, b):
    """a, b = (south, north, west, east). True if they intersect."""
    return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])


def validate_features(names, center, halfwidth=0.05, state='', county='',
                      slack=1.5):
    """Split feature names into (inside_chip, outside_or_ungeocodable).

    A feature counts as inside when its OSM bounding box intersects the image
    chip. Centroid distance alone rejects long linear features — a highway or
    river crossing the chip can have a centroid 10-20 km away.

    slack widens the chip slightly to tolerate geocoding imprecision.

    Returns (kept, dropped); items carry name/lat/lon and a reason when dropped.
    """
    if not center:
        return [], [{'name': n, 'reason': 'no event center'} for n in names]

    if not geocoder_available():
        return ([{'name': n, 'lat': None, 'lon': None, 'unvalidated': True}
                 for n in names if (n or '').strip()],
                [{'name': '__geocoder__', 'reason': 'no geocoder reachable; '
                  'features passed through unvalidated'}])

    clat, clon = center[0], center[1]
    hw = halfwidth * slack
    chip = (clat - hw, clat + hw, clon - hw, clon + hw)  # s, n, w, e
    kept, dropped = [], []

    for raw in names:
        name = (raw or '').strip()
        if not name:
            continue
        lat, lon, bbox = geocode(name, state, county)
        if lat is None:
            dropped.append({'name': name, 'reason': 'not geocodable'})
            continue

        inside = False
        how = ''
        if bbox and _boxes_overlap(bbox, chip):
            inside, how = True, 'bbox overlaps chip'
        elif (abs(lat - clat) <= hw and abs(lon - clon) <= hw):
            inside, how = True, 'centroid in chip'

        if inside:
            kept.append({'name': name, 'lat': lat, 'lon': lon,
                         'bbox': bbox, 'match': how})
        else:
            km = ((abs(lat - clat) * 111.0) ** 2 +
                  (abs(lon - clon) * 111.0) ** 2) ** 0.5
            dropped.append({'name': name, 'lat': lat, 'lon': lon,
                            'reason': f'{km:.0f} km from center'})
    return kept, dropped
