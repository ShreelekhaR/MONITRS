"""
Adapter to load v2 pipeline data (events_processed.json + Data/images/)
into the format the QA scripts expect.
"""

import json
import os
import re
import math
import requests
from os.path import join, isdir
from time import sleep


RESULTS_FILE = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'
GEOCODE_API_KEY = os.environ.get('GEOCODE_API_KEY', '')
NOMINATIM_URL = os.environ.get('NOMINATIM_URL', 'http://nominatim.geocoder.internal:8080')
GEOCODE_CACHE_FILE = 'Data/geocode_cache.json'

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
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'PR': 'Puerto Rico',
}

_geocode_cache = None


def load_events(results_file=RESULTS_FILE):
    with open(results_file) as f:
        return json.load(f)


def get_image_paths(event_id):
    paths = []

    # Try direct folder first (production download)
    img_dir = join(IMAGES_DIR, str(event_id))
    if not isdir(img_dir):
        # Fall back to strategy-specific folders
        for suffix in ['_firms', '_llm', '_fema']:
            candidate = join(IMAGES_DIR, f"{event_id}{suffix}")
            if isdir(candidate):
                img_dir = candidate
                break

    if not isdir(img_dir):
        return []

    for fname in sorted(os.listdir(img_dir)):
        if fname.endswith('.png') or fname.endswith('.jpg'):
            paths.append(join(img_dir, fname))

    return paths


def get_image_dates(event_id):
    paths = get_image_paths(event_id)
    dates = []
    for p in paths:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', p)
        if match:
            dates.append(match.group(1))
    return sorted(set(dates))


def parse_captions(caption_text):
    captions = {}
    if not caption_text:
        return captions
    for line in caption_text.strip().split('\n'):
        match = re.match(r'(\d{4}-\d{2}-\d{2}):\s*(.*)', line.strip())
        if match:
            captions[match.group(1)] = match.group(2)
    return captions


def _load_geocode_cache():
    global _geocode_cache
    if _geocode_cache is None:
        if os.path.exists(GEOCODE_CACHE_FILE):
            _geocode_cache = json.load(open(GEOCODE_CACHE_FILE))
        else:
            _geocode_cache = {}
    return _geocode_cache


def _save_geocode_cache():
    if _geocode_cache:
        os.makedirs(os.path.dirname(GEOCODE_CACHE_FILE) or '.', exist_ok=True)
        json.dump(_geocode_cache, open(GEOCODE_CACHE_FILE, 'w'), indent=2)


def geocode_location(loc_name, state=''):
    cache = _load_geocode_cache()
    cache_key = f"{loc_name}|{state}"
    if cache_key in cache:
        return cache[cache_key]

    state_full = STATE_NAMES.get(state, state)
    query = f"{loc_name}, {state_full}" if state_full else loc_name

    # Try local Nominatim first (no rate limit), then geocode.maps.co
    endpoints = [
        (f'{NOMINATIM_URL}/search', {'q': query, 'format': 'json', 'limit': 5}),
    ]
    if GEOCODE_API_KEY:
        endpoints.append(
            (f'https://geocode.maps.co/search', {'q': query, 'api_key': GEOCODE_API_KEY})
        )

    for url, params in endpoints:
        try:
            resp = requests.get(url, params=params,
                                headers={'User-Agent': 'MONITRS/2.0'}, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data:
                continue
            for r in data:
                if state_full.lower() in r.get('display_name', '').lower():
                    result = (float(r['lat']), float(r['lon']))
                    cache[cache_key] = result
                    return result
            if 'United States' in data[0].get('display_name', ''):
                result = (float(data[0]['lat']), float(data[0]['lon']))
                cache[cache_key] = result
                return result
        except Exception:
            continue

    cache[cache_key] = (None, None)
    return None, None


def _fuzzy_match(article_loc, osm_name):
    a = article_loc.lower().strip()
    o = osm_name.lower().strip()
    if a == o:
        return True
    if len(a) > 3 and a in o:
        return True
    if len(o) > 3 and o in a:
        return True
    filler = {'road', 'rd', 'street', 'st', 'avenue', 'ave', 'drive', 'dr',
              'lane', 'ln', 'highway', 'hwy', 'creek', 'river', 'the', 'of',
              'county', 'park', 'school', 'fire', 'station', 'north', 'south',
              'east', 'west', 'el', 'la', 'las', 'los', 'san', 'santa', 'de'}
    a_words = set(a.replace(',', ' ').split()) - filler
    o_words = set(o.replace(',', ' ').split()) - filler
    overlap = a_words & o_words
    if len(overlap) >= 2:
        return True
    if len(overlap) >= 1:
        for word in overlap:
            if len(word) > 4:
                return True
    return False


_osm_cache = {}


def osm_match_locations(event_data, halfwidth=0.05):
    """Match article locations to OSM features inside the bbox."""
    # Skip if no location names to match
    loc_names = [le.get('location', '') for le in event_data.get('location_events', []) if le.get('location')]
    if not loc_names:
        return {}

    center = event_data.get('center', event_data.get('fema_center', [0, 0]))
    cache_key = f"{center[0]:.4f},{center[1]:.4f},{halfwidth}"

    if cache_key not in _osm_cache:
        try:
            from osm_features import get_osm_features, osm_to_pixels
            osm_all = get_osm_features(center, halfwidth)
            osm_inside = osm_to_pixels(osm_all, center)
            _osm_cache[cache_key] = osm_inside
            sleep(2)  # rate limit for Overpass
        except Exception as e:
            _osm_cache[cache_key] = []

    osm_inside = _osm_cache[cache_key]
    if not osm_inside:
        return {}

    locations = {}
    for le in event_data.get('location_events', []):
        loc_name = le.get('location', '')
        if not loc_name or loc_name in locations:
            continue
        for feat in osm_inside:
            if _fuzzy_match(loc_name, feat['name']):
                locations[loc_name] = (feat['lat'], feat['lon'])
                break

    return locations


def geocode_event_locations(event_data, halfwidth=0.05):
    """Geocode article locations, return only those inside the bbox."""
    center = event_data.get('center', event_data.get('fema_center', [0, 0]))
    state = event_data.get('state', '')
    locations = {}

    loc_names = set()
    for le in event_data.get('location_events', []):
        loc = le.get('location', '')
        if loc:
            loc_names.add(loc)

    if not loc_names:
        return {}

    for loc_name in loc_names:
        lat, lon = geocode_location(loc_name, state)
        if lat is None:
            continue
        if (abs(lat - center[0]) <= halfwidth and abs(lon - center[1]) <= halfwidth):
            locations[loc_name] = (lat, lon)

    return locations


def event_to_v1_format(event_id, event_data):
    """Convert v2 event data to the format the existing QA generators expect."""
    center = event_data.get('center', event_data.get('fema_center', [0, 0]))
    base_coords = (center[0], center[1])

    # Geocode article locations and keep only those inside bbox
    halfwidth = event_data.get('halfwidth', 0.05)
    locations = {}
    if os.environ.get('ENABLE_LOCATION_QA'):
        # Nominatim runs locally, no API key needed. geocode.maps.co is fallback.
        locations = geocode_event_locations(event_data, halfwidth)

    # Build events list from location_events
    events = []
    seen = set()
    for le in event_data.get('location_events', []):
        date = le.get('date', '')
        event_text = le.get('event', '')
        key = (date, event_text)
        if key not in seen and date and event_text:
            seen.add(key)
            events.append({'date': date, 'event': event_text})

    # If no location_events, fall back to captions
    if not events:
        for date, caption in parse_captions(event_data.get('captions', '')).items():
            events.append({'date': date, 'event': caption})

    return {
        'id': str(event_id),
        'url': '',
        'base_coordinates': base_coords,
        'locations': locations,
        'events': sorted(events, key=lambda e: e['date']),
        'event_type': event_data.get('type', 'Unknown'),
        'captions': event_data.get('captions', ''),
        'start_date': event_data.get('start_date', ''),
        'end_date': event_data.get('end_date', ''),
        'image_dates': get_image_dates(event_id),
    }


def load_all_v1_format(results_file=RESULTS_FILE):
    """Load all events and convert to v1 format for QA scripts."""
    import sys
    print(f"Loading events from {results_file}...", flush=True)
    events = load_events(results_file)
    print(f"  Loaded {len(events)} events", flush=True)
    converted = {}
    n_with_locs = 0
    total = len([e for e in events.values() if 'error' not in e])
    print(f"  Processing {total} valid events (ENABLE_LOCATION_QA={os.environ.get('ENABLE_LOCATION_QA', 'not set')}, GEOCODE_API_KEY={'set' if GEOCODE_API_KEY else 'not set'})", flush=True)
    for i, (eid, edata) in enumerate(events.items()):
        if 'error' in edata:
            continue
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] {n_with_locs} with locations so far", flush=True)
        converted[eid] = event_to_v1_format(eid, edata)
        if converted[eid]['locations']:
            n_with_locs += 1
            print(f"    Event {eid}: {list(converted[eid]['locations'].keys())}", flush=True)
    if GEOCODE_API_KEY or os.environ.get('ENABLE_LOCATION_QA'):
        _save_geocode_cache()
    print(f"  {n_with_locs}/{len(converted)} events have locations inside bbox", flush=True)
    return converted
