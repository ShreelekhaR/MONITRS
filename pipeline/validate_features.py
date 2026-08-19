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
    """Return (lat, lon) or (None, None)."""
    cache = _load()
    state = STATE_NAMES.get((state_abbr or '').upper(), state_abbr or '')
    key = f'{name}|{county}|{state}'
    if key in cache:
        v = cache[key]
        return (v[0], v[1]) if v else (None, None)

    # Try most-specific query first
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
                    cache[key] = [lat, lon]
                    _save()
                    return lat, lon
            except Exception:
                continue

    cache[key] = None
    _save()
    return None, None


def validate_features(names, center, halfwidth=0.05, state='', county='',
                      slack=1.5):
    """Split feature names into (inside_bbox, outside_or_ungeocodable).

    slack widens the acceptance box slightly, since a river or highway is a long
    line whose geocoded centroid may sit just outside the chip while the feature
    itself still crosses it.

    Returns (kept, dropped) where each item is a dict with name/lat/lon/reason.
    """
    if not center:
        return [], [{'name': n, 'reason': 'no event center'} for n in names]

    # If no geocoder is reachable, don't silently delete every feature.
    if not geocoder_available():
        return ([{'name': n, 'lat': None, 'lon': None, 'unvalidated': True}
                 for n in names if (n or '').strip()],
                [{'name': '__geocoder__', 'reason': 'no geocoder reachable; '
                  'features passed through unvalidated'}])

    clat, clon = center[0], center[1]
    hw = halfwidth * slack
    kept, dropped = [], []

    for raw in names:
        name = (raw or '').strip()
        if not name:
            continue
        lat, lon = geocode(name, state, county)
        if lat is None:
            dropped.append({'name': name, 'reason': 'not geocodable'})
            continue
        dlat, dlon = abs(lat - clat), abs(lon - clon)
        if dlat <= hw and dlon <= hw:
            kept.append({'name': name, 'lat': lat, 'lon': lon})
        else:
            # rough km for readability
            km = ((dlat * 111.0) ** 2 + (dlon * 111.0) ** 2) ** 0.5
            dropped.append({'name': name, 'lat': lat, 'lon': lon,
                            'reason': f'{km:.0f} km from center'})
    return kept, dropped
