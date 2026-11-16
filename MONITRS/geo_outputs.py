"""
MONITRS v2 — Dataset output formats

  GeoJSON     locations: one Feature per location (image folder), its image
              footprint as a Polygon; written by scrape_articles.py, read by
              generate_article_captions.py
  .npz        captions: {images_dir}/{image_dir_id}/captions.npz, arrays
              aligned by index (dates[i] <-> captions[i])
  GeoParquet  dataset index: one row per captioned location with its
              footprint, metadata, image paths and captions
  GeoTIFF     imagery: written by scrape_articles.py (not here)

Coordinates are WGS84 longitude/latitude (EPSG:4326).
"""

import os
import json

import numpy as np
import geopandas as gpd
from shapely.geometry import box

CAPTIONS_FILE = 'captions.npz'


# --- Locations (GeoJSON) ---

def location_places(loc):
    """Place names of the location-events aligned to this location's images."""
    places = []
    for events in (loc.get('event_image_alignment') or {}).values():
        for le in events:
            p = le.get('location')
            if p and p not in places:
                places.append(p)
    return places


def location_articles(loc, ev):
    """URLs of the articles this location's events came from (all scraped articles if none)."""
    ids = {str(x) for x in loc.get('event_ids') or []}
    urls = []
    for le in ev.get('location_events') or []:
        url = le.get('source_url')
        if url and str(le.get('id')) in ids and url not in urls:
            urls.append(url)
    return urls or [a['url'] for a in ev.get('articles') or [] if 'skipped' not in a]


def footprint(lat, lon, halfwidth):
    """Image footprint polygon (the region scrape_articles.py downloads)."""
    return box(lon - halfwidth, lat - halfwidth, lon + halfwidth, lat + halfwidth)


def location_rows(results):
    """One dict per location (properties + shapely 'geometry'), from scrape_articles.py
    results ({query_id: event})."""
    rows = []
    for query_id, ev in results.items():
        for loc in ev.get('locations') or []:
            if not loc.get('image_dir_id'):
                continue
            lat, lon = float(loc['lat']), float(loc['lon'])
            halfwidth = float(loc.get('halfwidth') or 0.05)
            rows.append({
                'image_dir_id': loc['image_dir_id'],
                'query_id': str(query_id),
                'country': ev.get('country', ''),
                'disaster_type': loc.get('disaster_type'),
                'start_date': loc.get('start_date'),
                'end_date': loc.get('end_date'),
                'lat': lat,
                'lon': lon,
                'halfwidth': halfwidth,
                'radius_km': loc.get('radius_km'),
                'places': location_places(loc),
                'image_dates': sorted(loc.get('image_dates') or []),
                'article_urls': location_articles(loc, ev),
                'geometry': footprint(lat, lon, halfwidth),
            })
    return rows


def write_locations_geojson(results, path):
    """Write the locations in results as a GeoJSON FeatureCollection (atomically)."""
    rows = location_rows(results)
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326') if rows else \
        gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        # to_json keeps the list properties (places, image_dates, article_urls) as JSON arrays
        f.write(gdf.to_json(ensure_ascii=False, drop_id=True))
    os.replace(tmp, path)
    return len(rows)


def read_locations(path):
    """Location property dicts from a locations GeoJSON, or from a scrape_articles.py
    events JSON (converted on the fly, for files made before the GeoJSON output)."""
    with open(path) as f:
        data = json.load(f)
    if data.get('type') == 'FeatureCollection':
        return [feat['properties'] for feat in data['features']]
    return [{k: v for k, v in row.items() if k != 'geometry'} for row in location_rows(data)]


# --- Captions (.npz) ---

def captions_path(images_dir, image_dir_id):
    return os.path.join(images_dir, image_dir_id, CAPTIONS_FILE)


def save_captions(path, captions, articles):
    """captions: {date: caption}; articles: [{'url', 'published'}]. Written atomically."""
    dates = sorted(captions)
    tmp = path[:-len('.npz')] + '.tmp.npz'
    np.savez(tmp,
             dates=np.array(dates, dtype=str),
             captions=np.array([captions[d] for d in dates], dtype=str),
             article_urls=np.array([a['url'] for a in articles], dtype=str),
             article_published=np.array([a.get('published') or '' for a in articles], dtype=str))
    os.replace(tmp, path)


def load_captions(path):
    """{'dates', 'captions', 'article_urls', 'article_published'} as lists of str."""
    with np.load(path) as data:
        return {k: data[k].tolist() for k in data.files}


# --- Dataset index (GeoParquet) ---

def write_geoparquet(locations, images_dir, path):
    """One row per location with a captions.npz, written as GeoParquet. Returns the row count."""
    rows = []
    for loc in locations:
        cap_path = captions_path(images_dir, loc['image_dir_id'])
        if not os.path.exists(cap_path):
            continue
        cap = load_captions(cap_path)
        folder = os.path.dirname(cap_path)
        files = sorted(os.listdir(folder))
        rows.append({
            'image_dir_id': loc['image_dir_id'],
            'query_id': str(loc['query_id']),
            'country': loc.get('country') or '',
            'disaster_type': loc.get('disaster_type') or '',
            'start_date': loc.get('start_date') or '',
            'end_date': loc.get('end_date') or '',
            'lat': loc['lat'],
            'lon': loc['lon'],
            'halfwidth': loc['halfwidth'],
            'places': loc.get('places') or [],
            'image_dates': loc.get('image_dates') or [],
            'tif_paths': [os.path.join(folder, f) for f in files if f.endswith('.tif')],
            'png_paths': [os.path.join(folder, f) for f in files if f.endswith('.png')],
            'captions_path': cap_path,
            'caption_dates': cap['dates'],
            'captions': cap['captions'],
            'article_urls': cap['article_urls'],
            'geometry': footprint(loc['lat'], loc['lon'], loc['halfwidth']),
        })
    if not rows:
        return 0

    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')
    tmp = path + '.tmp'
    gdf.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return len(rows)
