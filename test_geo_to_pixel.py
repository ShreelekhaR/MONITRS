"""
Test and visualize geo_to_pixel by comparing satellite image with OSM map
at the same location, with location pins overlaid on both.

Usage:
    python test_geo_to_pixel.py
    python test_geo_to_pixel.py --event 0
"""

import os
import re
import json
import argparse
import math
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image as PILImage

RESULTS_FILE = 'Data/events_processed.json'
ODIR = 'Data/images'
OUT_DIR = 'Data/visualizations'


def geo_to_pixel_correct(locations, center):
    """Correct geo_to_pixel using Sentinel-2 GSD (10m/px at 512x512)."""
    height = 512
    width = 512
    gsd_meters = 10.0
    center_lat, center_lon = center

    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(center_lat))

    pixel_locations = {}
    for loc_name, coords in locations.items():
        lat, lon = coords
        x_meters = (lon - center_lon) * meters_per_degree_lon
        y_meters = (lat - center_lat) * meters_per_degree_lat
        x_pixel = int((x_meters / gsd_meters) + width / 2)
        y_pixel = int((-y_meters / gsd_meters) + height / 2)
        pixel_locations[loc_name] = (x_pixel, y_pixel)
    return pixel_locations


def geo_to_pixel_old(locations, center):
    """Old (incorrect) geo_to_pixel from templated_mcq.py."""
    height = 512
    width = 512
    center_lat, center_lon = center

    pixel_locations = {}
    for loc_name, coords in locations.items():
        lat, lon = coords
        x_offset = int((lon - center_lon) * (width / 360.0) + width / 2)
        y_offset = int((lat - center_lat) * (height / 180.0) + height / 2)
        pixel_locations[loc_name] = (x_offset, y_offset)
    return pixel_locations


def get_osm_tile(center, halfwidth=0.05, size=512):
    """Download an OSM static map image for comparison."""
    lat, lon = center
    bbox = f"{lon-halfwidth},{lat-halfwidth},{lon+halfwidth},{lat+halfwidth}"
    url = (f"https://render.openstreetmap.org/cgi-bin/export?"
           f"bbox={bbox}&scale=5000&format=png")

    # Alternative: use a static tile approach
    zoom = 12
    url = (f"https://static-maps.yandex.ru/v1?"
           f"ll={lon},{lat}&z={zoom}&size=512,512&l=map")

    # Simpler: use OSM tile server directly
    # Calculate tile coordinates
    n = 2 ** zoom
    xtile = int((lon + 180) / 360 * n)
    ytile = int((1 - math.log(math.tan(math.radians(lat)) + 1/math.cos(math.radians(lat))) / math.pi) / 2 * n)

    # Download 3x3 grid of tiles and crop to center
    tile_size = 256
    tiles = np.zeros((tile_size * 3, tile_size * 3, 3), dtype=np.uint8)

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            tile_url = f"https://tile.openstreetmap.org/{zoom}/{xtile+dx}/{ytile+dy}.png"
            tmp_file = f'/tmp/osm_tile_{zoom}_{xtile+dx}_{ytile+dy}.png'
            try:
                if not os.path.exists(tmp_file):
                    req = urllib.request.Request(tile_url, headers={'User-Agent': 'MONITRS/2.0'})
                    urllib.request.urlretrieve(tile_url, tmp_file)
                tile_img = np.array(PILImage.open(tmp_file).convert('RGB'))
                ox = (dx + 1) * tile_size
                oy = (dy + 1) * tile_size
                tiles[oy:oy+tile_size, ox:ox+tile_size] = tile_img
            except Exception:
                continue

    # Crop center 512x512
    cy, cx = tile_size * 3 // 2, tile_size * 3 // 2
    cropped = tiles[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
    return cropped


def find_event_image(event_idx):
    """Find the first during-event image for an event."""
    for suffix in ['', '_firms', '_llm', '_fema']:
        img_dir = os.path.join(ODIR, f"{event_idx}{suffix}")
        if os.path.isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if ('during' in fname or 'event' in fname) and (fname.endswith('.png') or fname.endswith('.jpg')):
                    return os.path.join(img_dir, fname)
                # Also match caption-date images (no phase prefix)
                if re.match(r'\d+_\d{4}-\d{2}-\d{2}\.png', fname):
                    return os.path.join(img_dir, fname)
    return None


def create_comparison(event_idx, event_data):
    center = event_data['center']
    halfwidth = event_data.get('halfwidth', 0.05)

    # Get satellite image
    sat_path = find_event_image(event_idx)
    if not sat_path:
        print(f"  No satellite image found for event {event_idx}")
        return None

    # Build locations from location_events
    locations = {}
    for le in event_data.get('location_events', []):
        loc = le.get('location', '')
        if loc and loc not in locations:
            locations[loc] = (center[0], center[1])

    # Also add FEMA center and chosen center as reference points
    ref_points = {
        'FEMA center': tuple(event_data.get('fema_center', center)),
        'Chosen center': tuple(center),
    }

    # Compute pixel positions with both methods
    pixels_correct = geo_to_pixel_correct(ref_points, center)
    pixels_old = geo_to_pixel_old(ref_points, center)

    # Add some test points at known offsets
    test_points = {}
    lat, lon = center
    meters_per_deg_lon = 111320 * math.cos(math.radians(lat))
    offsets_km = [1, 2, 3]
    for km in offsets_km:
        test_points[f'{km}km E'] = (lat, lon + km * 1000 / meters_per_deg_lon)
        test_points[f'{km}km N'] = (lat + km * 1000 / 111320, lon)

    test_px_correct = geo_to_pixel_correct(test_points, center)
    test_px_old = geo_to_pixel_old(test_points, center)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: satellite image with CORRECT geo_to_pixel
    sat_img = mpimg.imread(sat_path)
    axes[0].imshow(sat_img)
    axes[0].set_title(f'Sentinel-2 + correct geo_to_pixel\n(GSD-based, 10m/px)', fontsize=11)

    for name, (px, py) in pixels_correct.items():
        if 0 <= px < 512 and 0 <= py < 512:
            color = 'yellow' if 'FEMA' in name else 'red'
            axes[0].plot(px, py, 'o', color=color, markersize=8)
            axes[0].annotate(name, (px, py), fontsize=7, color=color,
                           xytext=(5, 5), textcoords='offset points')

    for name, (px, py) in test_px_correct.items():
        if 0 <= px < 512 and 0 <= py < 512:
            axes[0].plot(px, py, 'x', color='cyan', markersize=6)
            axes[0].annotate(name, (px, py), fontsize=6, color='cyan',
                           xytext=(3, 3), textcoords='offset points')

    # Right: satellite image with OLD geo_to_pixel
    axes[1].imshow(sat_img)
    axes[1].set_title(f'Sentinel-2 + OLD geo_to_pixel\n(360°/180° projection — WRONG)', fontsize=11)

    for name, (px, py) in pixels_old.items():
        if 0 <= px < 512 and 0 <= py < 512:
            color = 'yellow' if 'FEMA' in name else 'red'
            axes[1].plot(px, py, 'o', color=color, markersize=8)
            axes[1].annotate(name, (px, py), fontsize=7, color=color,
                           xytext=(5, 5), textcoords='offset points')

    for name, (px, py) in test_px_old.items():
        if 0 <= px < 512 and 0 <= py < 512:
            axes[1].plot(px, py, 'x', color='cyan', markersize=6)
            axes[1].annotate(name, (px, py), fontsize=6, color='cyan',
                           xytext=(3, 3), textcoords='offset points')

    event_name = event_data.get('event', '?')
    fig.suptitle(f"{event_name}\nCenter: ({center[0]:.4f}, {center[1]:.4f}) | halfwidth: {halfwidth}°",
                 fontsize=13, fontweight='bold')

    # Print comparison
    print(f"\n  Pixel comparison for reference points:")
    print(f"  {'Point':<20} {'Correct (GSD)':>15} {'Old (360/180)':>15} {'Difference':>12}")
    print(f"  {'-'*65}")
    for name in ref_points:
        cx, cy = pixels_correct[name]
        ox, oy = pixels_old[name]
        diff = math.sqrt((cx-ox)**2 + (cy-oy)**2)
        print(f"  {name:<20} ({cx:>3}, {cy:>3})      ({ox:>3}, {oy:>3})      {diff:>8.1f} px")

    print(f"\n  Test points (should be evenly spaced):")
    for name in test_points:
        cx, cy = test_px_correct[name]
        ox, oy = test_px_old[name]
        print(f"  {name:<20} correct=({cx:>4}, {cy:>4})  old=({ox:>4}, {oy:>4})")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event', type=int, default=None)
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(RESULTS_FILE):
        print(f"No results at {RESULTS_FILE}")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.event is not None:
        event_ids = [str(args.event)]
    else:
        # Pick events with images
        candidates = []
        for eid, data in results.items():
            if 'error' in data:
                continue
            if find_event_image(eid):
                candidates.append(eid)
        event_ids = candidates[:args.n]

    for eid in event_ids:
        if eid not in results:
            print(f"Event {eid}: not found")
            continue
        data = results[eid]
        if 'error' in data:
            continue

        print(f"\nEvent {eid}: {data.get('event', '?')}")
        fig = create_comparison(eid, data)
        if fig:
            out_path = os.path.join(OUT_DIR, f'geo_to_pixel_{eid}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
