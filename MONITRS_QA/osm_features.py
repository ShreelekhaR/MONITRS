"""
Query OpenStreetMap for features inside an event's bbox.
Returns named roads, rivers, towns, landmarks with pixel coordinates.

Usage:
    from osm_features import get_osm_features, osm_to_pixels
"""

import requests
import math
from time import sleep


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def get_osm_features(center, halfwidth=0.05):
    """Query OSM Overpass for named features inside the bbox."""
    lat, lon = center
    south = lat - halfwidth
    north = lat + halfwidth
    west = lon - halfwidth
    east = lon + halfwidth
    bbox = f"{south},{west},{north},{east}"

    query = f"""
    [out:json][timeout:30];
    (
      way["highway"]["name"]({bbox});
      way["waterway"]["name"]({bbox});
      node["place"]({bbox});
      way["natural"="water"]["name"]({bbox});
      way["landuse"]["name"]({bbox});
      node["amenity"]["name"]({bbox});
      way["building"]["name"]({bbox});
      relation["boundary"="administrative"]["name"]({bbox});
    );
    out center;
    """

    try:
        resp = requests.post(OVERPASS_URL, data={'data': query}, timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []

    features = []
    seen_names = set()

    for element in data.get('elements', []):
        tags = element.get('tags', {})
        name = tags.get('name', '')
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        # Get coordinates
        if element['type'] == 'node':
            feat_lat = element['lat']
            feat_lon = element['lon']
        elif 'center' in element:
            feat_lat = element['center']['lat']
            feat_lon = element['center']['lon']
        else:
            continue

        # Classify feature type
        if 'highway' in tags:
            feat_type = 'road'
            feat_subtype = tags.get('highway', '')
        elif 'waterway' in tags:
            feat_type = 'waterway'
            feat_subtype = tags.get('waterway', '')
        elif 'natural' in tags and tags['natural'] == 'water':
            feat_type = 'water'
            feat_subtype = 'lake/pond'
        elif 'place' in tags:
            feat_type = 'place'
            feat_subtype = tags.get('place', '')
        elif 'landuse' in tags:
            feat_type = 'landuse'
            feat_subtype = tags.get('landuse', '')
        elif 'amenity' in tags:
            feat_type = 'amenity'
            feat_subtype = tags.get('amenity', '')
        elif 'building' in tags:
            feat_type = 'building'
            feat_subtype = tags.get('building', '')
        elif 'boundary' in tags:
            feat_type = 'boundary'
            feat_subtype = tags.get('admin_level', '')
        else:
            feat_type = 'other'
            feat_subtype = ''

        features.append({
            'name': name,
            'lat': feat_lat,
            'lon': feat_lon,
            'type': feat_type,
            'subtype': feat_subtype,
        })

    return features


def geo_to_pixel(lat, lon, center, gsd=10.0, img_size=512):
    """Convert lat/lon to pixel coordinates using Sentinel-2 GSD."""
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(center[0]))
    x_meters = (lon - center[1]) * meters_per_degree_lon
    y_meters = (lat - center[0]) * meters_per_degree_lat
    x_pixel = int((x_meters / gsd) + img_size / 2)
    y_pixel = int((-y_meters / gsd) + img_size / 2)
    return x_pixel, y_pixel


def osm_to_pixels(features, center):
    """Convert OSM features to pixel coordinates, filter to inside image."""
    result = []
    for feat in features:
        px, py = geo_to_pixel(feat['lat'], feat['lon'], center)
        if 0 <= px < 512 and 0 <= py < 512:
            feat['pixel_x'] = px
            feat['pixel_y'] = py
            result.append(feat)
    return result


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--halfwidth', type=float, default=0.05)
    args = parser.parse_args()

    center = (args.lat, args.lon)
    features = get_osm_features(center, args.halfwidth)
    inside = osm_to_pixels(features, center)

    print(f"OSM features: {len(features)} total, {len(inside)} inside image")
    print(f"\n{'Name':<30} {'Type':<12} {'Pixel':>12}")
    print('-' * 56)
    for f in inside:
        print(f"{f['name'][:30]:<30} {f['type']:<12} ({f['pixel_x']:>3}, {f['pixel_y']:>3})")
