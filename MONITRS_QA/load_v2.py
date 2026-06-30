"""
Adapter to load v2 pipeline data (events_processed.json + Data/images/)
into the format the QA scripts expect.
"""

import json
import os
import re
from os.path import join, isdir


RESULTS_FILE = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'


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


def event_to_v1_format(event_id, event_data):
    """Convert v2 event data to the format the existing QA generators expect."""
    center = event_data.get('center', event_data.get('fema_center', [0, 0]))
    base_coords = (center[0], center[1])

    # Build locations dict from location_events
    locations = {}
    for le in event_data.get('location_events', []):
        loc_name = le.get('location', '')
        if loc_name and loc_name not in locations:
            locations[loc_name] = base_coords

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
    }


def load_all_v1_format(results_file=RESULTS_FILE):
    """Load all events and convert to v1 format for QA scripts."""
    events = load_events(results_file)
    converted = {}
    for eid, edata in events.items():
        if 'error' in edata:
            continue
        converted[eid] = event_to_v1_format(eid, edata)
    return converted
