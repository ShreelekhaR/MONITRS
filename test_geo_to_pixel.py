"""
Visualize geocoded article locations on satellite imagery.
Geocodes each location from the event, converts to pixel coordinates,
and plots on the satellite image.

Usage:
    python test_geo_to_pixel.py --event 0
    python test_geo_to_pixel.py --n 3
"""

import os
import re
import json
import argparse
import math
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from time import sleep

RESULTS_FILE = 'Data/events_processed.json'
ODIR = 'Data/images'
OUT_DIR = 'Data/visualizations'
GEOCODE_API_KEY = os.environ.get('GEOCODE_API_KEY', '')

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


def geocode(loc_name, state=''):
    if not GEOCODE_API_KEY:
        return None, None
    state_full = STATE_NAMES.get(state, state)
    query = f"{loc_name}, {state_full}" if state_full else loc_name
    try:
        resp = requests.get(
            f'https://geocode.maps.co/search?q={query}&api_key={GEOCODE_API_KEY}',
            timeout=10)
        data = resp.json()
        if data:
            for r in data:
                if state_full.lower() in r.get('display_name', '').lower():
                    return float(r['lat']), float(r['lon'])
            if 'United States' in data[0].get('display_name', ''):
                return float(data[0]['lat']), float(data[0]['lon'])
    except Exception:
        pass
    return None, None


def geo_to_pixel(lat, lon, center, halfwidth=0.05):
    gsd_meters = 10.0
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(center[0]))
    x_meters = (lon - center[1]) * meters_per_degree_lon
    y_meters = (lat - center[0]) * meters_per_degree_lat
    x_pixel = int((x_meters / gsd_meters) + 256)
    y_pixel = int((-y_meters / gsd_meters) + 256)
    return x_pixel, y_pixel


def find_event_image(event_idx):
    for suffix in ['', '_firms', '_llm', '_fema']:
        img_dir = os.path.join(ODIR, f"{event_idx}{suffix}")
        if os.path.isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    if 'during' in fname or 'event' in fname or re.match(r'\d+_\d{4}', fname):
                        return os.path.join(img_dir, fname)
    return None


def create_viz(event_idx, event_data):
    center = event_data['center']
    halfwidth = event_data.get('halfwidth', 0.05)
    state = event_data.get('state', '')

    img_path = find_event_image(event_idx)
    if not img_path:
        return None

    # Get unique location names
    loc_names = list(set(
        le.get('location', '') for le in event_data.get('location_events', [])
        if le.get('location')
    ))

    # Geocode each
    geocoded = {}
    for loc in loc_names:
        lat, lon = geocode(loc, state)
        if lat is not None:
            dist_km = math.sqrt((lat - center[0])**2 + (lon - center[1])**2) * 111
            geocoded[loc] = {'lat': lat, 'lon': lon, 'dist_km': dist_km}
            sleep(1.1)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    sat_img = mpimg.imread(img_path)

    # Left: satellite image with all geocoded points
    axes[0].imshow(sat_img)
    axes[0].set_title('Geocoded locations on satellite image', fontsize=11)

    # Plot center
    cx, cy = geo_to_pixel(center[0], center[1], center, halfwidth)
    axes[0].plot(cx, cy, '+', color='yellow', markersize=15, markeredgewidth=2)
    axes[0].annotate('CENTER', (cx, cy), fontsize=7, color='yellow',
                     xytext=(5, -15), textcoords='offset points', fontweight='bold')

    # Plot FEMA center
    fema = event_data.get('fema_center', center)
    fx, fy = geo_to_pixel(fema[0], fema[1], center, halfwidth)
    if 0 <= fx < 512 and 0 <= fy < 512:
        axes[0].plot(fx, fy, 's', color='yellow', markersize=8, markerfacecolor='none', markeredgewidth=2)
        axes[0].annotate('FEMA', (fx, fy), fontsize=7, color='yellow',
                         xytext=(5, 5), textcoords='offset points')

    colors_inside = {'visual': '#FF4444', 'contextual': '#4488FF'}
    colors_outside = '#888888'

    for loc, data in geocoded.items():
        px, py = geo_to_pixel(data['lat'], data['lon'], center, halfwidth)
        inside = 0 <= px < 512 and 0 <= py < 512

        # Get type from location_events
        loc_type = 'visual'
        for le in event_data.get('location_events', []):
            if le.get('location') == loc:
                loc_type = le.get('type', 'visual')
                break

        if inside:
            color = colors_inside.get(loc_type, '#FF4444')
            axes[0].plot(px, py, 'o', color=color, markersize=7, markeredgewidth=1.5, markerfacecolor='none')
            axes[0].annotate(loc[:20], (px, py), fontsize=6, color=color,
                             xytext=(5, 5), textcoords='offset points')

    axes[0].set_xlim(0, 512)
    axes[0].set_ylim(512, 0)

    # Right: zoomed out view showing all points including outside bbox
    axes[1].imshow(sat_img, extent=[-halfwidth, halfwidth, -halfwidth, halfwidth])
    axes[1].set_title('All geocoded locations (zoomed out)', fontsize=11)

    # Compute extent to show all points
    all_lats = [center[0]] + [d['lat'] for d in geocoded.values()]
    all_lons = [center[1]] + [d['lon'] for d in geocoded.values()]
    lat_range = max(all_lats) - min(all_lats)
    lon_range = max(all_lons) - min(all_lons)
    extent = max(lat_range, lon_range, halfwidth * 2) * 1.2

    axes[1].set_xlim(-extent/2, extent/2)
    axes[1].set_ylim(-extent/2, extent/2)

    # Plot all points relative to center
    for loc, data in geocoded.items():
        dx = data['lon'] - center[1]
        dy = data['lat'] - center[0]
        inside = abs(dx) <= halfwidth and abs(dy) <= halfwidth

        loc_type = 'visual'
        for le in event_data.get('location_events', []):
            if le.get('location') == loc:
                loc_type = le.get('type', 'visual')
                break

        color = colors_inside.get(loc_type, '#FF4444') if inside else colors_outside
        marker = 'o' if inside else 'x'
        axes[1].plot(dx, dy, marker, color=color, markersize=8)
        axes[1].annotate(f"{loc[:15]} ({data['dist_km']:.0f}km)", (dx, dy),
                         fontsize=6, color=color, xytext=(3, 3), textcoords='offset points')

    # Draw bbox
    rect = plt.Rectangle((-halfwidth, -halfwidth), halfwidth*2, halfwidth*2,
                          linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
    axes[1].add_patch(rect)
    axes[1].plot(0, 0, '+', color='yellow', markersize=12, markeredgewidth=2)
    axes[1].set_xlabel('Longitude offset (degrees)')
    axes[1].set_ylabel('Latitude offset (degrees)')

    event_name = event_data.get('event', '?')
    strategy = event_data.get('strategy', '?')
    n_inside = sum(1 for d in geocoded.values()
                   if abs(d['lon'] - center[1]) <= halfwidth and abs(d['lat'] - center[0]) <= halfwidth)
    fig.suptitle(f"{event_name}\n{strategy} center | {len(geocoded)} geocoded, {n_inside} inside bbox",
                 fontsize=13, fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='#FF4444', markersize=8, label='Visual (inside)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='#4488FF', markersize=8, label='Contextual (inside)'),
        Line2D([0], [0], marker='x', color='#888888', markersize=8, label='Outside bbox', linestyle='None'),
        Line2D([0], [0], marker='+', color='yellow', markersize=10, label='Center', linestyle='None'),
        Line2D([0], [0], marker='s', color='yellow', markerfacecolor='none', markersize=8, label='FEMA center', linestyle='None'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=7)

    plt.tight_layout()

    # Print summary
    print(f"  Geocoded {len(geocoded)}/{len(loc_names)} locations:")
    for loc, data in sorted(geocoded.items(), key=lambda x: x[1]['dist_km']):
        inside = abs(data['lon'] - center[1]) <= halfwidth and abs(data['lat'] - center[0]) <= halfwidth
        tag = "IN" if inside else "OUT"
        print(f"    [{tag}] {loc}: ({data['lat']:.4f}, {data['lon']:.4f}) — {data['dist_km']:.1f}km")

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
        candidates = [eid for eid, d in results.items()
                      if 'error' not in d and find_event_image(eid)]
        event_ids = candidates[:args.n]

    if not GEOCODE_API_KEY:
        print("Warning: GEOCODE_API_KEY not set. Set it to geocode locations.")

    for eid in event_ids:
        if eid not in results or 'error' in results[eid]:
            continue
        print(f"\nEvent {eid}: {results[eid].get('event', '?')}")
        fig = create_viz(eid, results[eid])
        if fig:
            out_path = os.path.join(OUT_DIR, f'geo_to_pixel_{eid}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
