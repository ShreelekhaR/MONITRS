"""
Quick test of OSM Overpass API for a location.

Usage:
    python test_osm.py --lat 34.4352 --lon -119.7372
    python test_osm.py --event 88
"""

import requests
import json
import argparse
import os

RESULTS_FILE = 'Data/events_processed.json'

def test_overpass(lat, lon, halfwidth=0.05):
    bbox = f"{lat-halfwidth},{lon-halfwidth},{lat+halfwidth},{lon+halfwidth}"

    query = (
        '[out:json][timeout:30];'
        '('
        f'way["highway"]({bbox});'
        f'way["waterway"]({bbox});'
        f'node["place"]({bbox});'
        f'way["natural"="water"]({bbox});'
        ')'
        ';out center;'
    )

    print(f"Querying OSM at ({lat:.4f}, {lon:.4f}), bbox={bbox}")
    print(f"Query length: {len(query)} chars")

    # Try multiple Overpass endpoints
    endpoints = [
        'https://overpass.kumi.systems/api/interpreter',
        'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
        'https://overpass-api.de/api/interpreter',
    ]
    resp = None
    for url in endpoints:
        try:
            resp = requests.get(url, params={'data': query}, timeout=30,
                               headers={'Accept': 'application/json'})
            if resp.status_code == 200:
                print(f"  Using: {url}")
                break
            print(f"  {url}: {resp.status_code}")
        except Exception as e:
            print(f"  {url}: {e}")
            continue

    if not resp or resp.status_code != 200:
        print(f"All endpoints failed")
        return

    print(f"Status: {resp.status_code}")
    print(f"Response length: {len(resp.text)} chars")

    if resp.status_code != 200:
        print(f"Error: {resp.text[:500]}")
        return

    data = resp.json()
    elements = data.get('elements', [])
    print(f"Elements: {len(elements)}")

    # Count by type
    by_type = {}
    for e in elements:
        tags = e.get('tags', {})
        if 'highway' in tags:
            t = f"road ({tags['highway']})"
        elif 'waterway' in tags:
            t = f"waterway ({tags['waterway']})"
        elif 'place' in tags:
            t = f"place ({tags['place']})"
        elif 'natural' in tags:
            t = f"natural ({tags['natural']})"
        else:
            t = 'other'
        by_type[t] = by_type.get(t, 0) + 1

    print(f"\nBy type:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # Show named features
    named = [e for e in elements if 'name' in e.get('tags', {})]
    print(f"\nNamed features: {len(named)}")
    for e in named[:15]:
        tags = e.get('tags', {})
        print(f"  {tags['name']} ({tags.get('highway', tags.get('waterway', tags.get('place', '?')))})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat', type=float, default=None)
    parser.add_argument('--lon', type=float, default=None)
    parser.add_argument('--event', type=int, default=None)
    parser.add_argument('--halfwidth', type=float, default=0.05)
    args = parser.parse_args()

    if args.event is not None:
        r = json.load(open(RESULTS_FILE))
        e = r[str(args.event)]
        lat, lon = e['center']
        print(f"Event {args.event}: {e['event']}")
    else:
        lat, lon = args.lat, args.lon

    test_overpass(lat, lon, args.halfwidth)


if __name__ == '__main__':
    main()
