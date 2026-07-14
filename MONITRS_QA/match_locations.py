"""
Match article locations to OSM features for confirmed pixel coordinates.

Usage:
    python MONITRS_QA/match_locations.py --event 88
    python MONITRS_QA/match_locations.py --n 10
"""

import json
import os
import sys
import argparse
from time import sleep

from osm_features import get_osm_features, osm_to_pixels


RESULTS_FILE = 'Data/events_processed.json'


def fuzzy_match(article_loc, osm_name):
    """Check if article location matches OSM feature name."""
    a = article_loc.lower().strip()
    o = osm_name.lower().strip()

    # Exact match
    if a == o:
        return True

    # One fully contains the other (not single words)
    if len(a) > 3 and a in o:
        return True
    if len(o) > 3 and o in a:
        return True

    # Multi-word key overlap: require at least 2 meaningful words to match
    filler = {'road', 'rd', 'street', 'st', 'avenue', 'ave', 'drive', 'dr',
              'lane', 'ln', 'highway', 'hwy', 'creek', 'river', 'the', 'of',
              'county', 'park', 'school', 'fire', 'station', 'north', 'south',
              'east', 'west', 'el', 'la', 'las', 'los', 'san', 'santa', 'de'}
    a_words = set(a.replace(',', ' ').split()) - filler
    o_words = set(o.replace(',', ' ').split()) - filler

    overlap = a_words & o_words
    if len(overlap) >= 2:
        return True
    if len(overlap) >= 1 and (len(a_words) == 1 or len(o_words) == 1):
        # Single meaningful word match only if one side has just 1 key word
        # and it's a substantial word (>4 chars)
        for word in overlap:
            if len(word) > 4:
                return True

    return False


def match_event_locations(event_data, osm_features_inside):
    """Match article location_events to OSM features."""
    matches = []
    article_locs = set()

    for le in event_data.get('location_events', []):
        loc = le.get('location', '')
        if loc:
            article_locs.add(loc)

    for article_loc in article_locs:
        for osm_feat in osm_features_inside:
            if fuzzy_match(article_loc, osm_feat['name']):
                matches.append({
                    'article_location': article_loc,
                    'osm_name': osm_feat['name'],
                    'osm_type': osm_feat['type'],
                    'lat': osm_feat['lat'],
                    'lon': osm_feat['lon'],
                    'pixel_x': osm_feat['pixel_x'],
                    'pixel_y': osm_feat['pixel_y'],
                })
                break

    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event', nargs='+', type=int, default=None)
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    if args.event:
        event_ids = [str(e) for e in args.event]
    else:
        candidates = [eid for eid, d in results.items() if 'error' not in d]
        event_ids = candidates[:args.n]

    total_article_locs = 0
    total_matched = 0

    for eid in event_ids:
        if eid not in results or 'error' in results[eid]:
            continue

        data = results[eid]
        center = data['center']
        halfwidth = data.get('halfwidth', 0.05)

        article_locs = set(le.get('location', '') for le in data.get('location_events', []) if le.get('location'))

        print(f"\nEvent {eid}: {data.get('event', '?')} ({data.get('type', '?')})")
        print(f"  Article locations: {list(article_locs)[:5]}{'...' if len(article_locs) > 5 else ''}")

        osm_features = get_osm_features(center, halfwidth)
        inside = osm_to_pixels(osm_features, center)
        print(f"  OSM features inside chip: {len(inside)}")

        matches = match_event_locations(data, inside)
        print(f"  Matches: {len(matches)}/{len(article_locs)}")

        for m in matches:
            print(f"    '{m['article_location']}' → '{m['osm_name']}' ({m['osm_type']}) at pixel ({m['pixel_x']}, {m['pixel_y']})")

        total_article_locs += len(article_locs)
        total_matched += len(matches)

        sleep(1)

    print(f"\n{'='*50}")
    print(f"Total: {total_matched}/{total_article_locs} article locations matched to OSM ({100*total_matched/max(1,total_article_locs):.0f}%)")


if __name__ == '__main__':
    main()
