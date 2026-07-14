"""
Visualize OSM features overlaid on satellite imagery.
Shows what named features (roads, rivers, towns) are actually inside each image chip.

Usage:
    python visualize_osm.py --event 0
    python visualize_osm.py --event 0 1 2 3 4
    python visualize_osm.py --n 5
"""

import os
import re
import json
import argparse
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import sleep

sys.path.insert(0, 'MONITRS_QA')
from osm_features import get_osm_features, osm_to_pixels

ODIR = 'Data/images'
RESULTS_FILE = 'Data/events_processed.json'
OUT_DIR = 'Data/visualizations'

TYPE_COLORS = {
    'road': '#FF4444',
    'waterway': '#4488FF',
    'water': '#2266CC',
    'place': '#FFAA00',
    'landuse': '#44AA44',
    'amenity': '#FF88FF',
    'building': '#AAAAAA',
    'boundary': '#FFFFFF',
    'other': '#CCCCCC',
}

TYPE_MARKERS = {
    'road': '-',
    'waterway': 'v',
    'water': 'D',
    'place': '*',
    'landuse': 's',
    'amenity': 'P',
    'building': '^',
    'boundary': 'h',
    'other': 'o',
}


def find_event_image(event_idx):
    for suffix in ['', '_firms', '_llm', '_fema']:
        img_dir = os.path.join(ODIR, f"{event_idx}{suffix}")
        if os.path.isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if (fname.endswith('.png') or fname.endswith('.jpg')):
                    if 'during' in fname or 'event' in fname or re.match(r'\d+_\d{4}', fname):
                        return os.path.join(img_dir, fname)
    return None


def create_osm_viz(event_idx, event_data, features_inside, features_all):
    img_path = find_event_image(event_idx)
    if not img_path:
        return None

    center = event_data['center']
    halfwidth = event_data.get('halfwidth', 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: satellite image with OSM features
    sat_img = mpimg.imread(img_path)
    axes[0].imshow(sat_img)
    axes[0].set_title(f'OSM features inside chip ({len(features_inside)} features)', fontsize=11)

    for feat in features_inside:
        color = TYPE_COLORS.get(feat['type'], '#CCCCCC')
        marker = TYPE_MARKERS.get(feat['type'], 'o')
        if marker == '-':
            marker = 'o'
        axes[0].plot(feat['pixel_x'], feat['pixel_y'], marker,
                     color=color, markersize=8, markeredgewidth=1.5, markerfacecolor='none')
        axes[0].annotate(feat['name'][:20], (feat['pixel_x'], feat['pixel_y']),
                         fontsize=5, color=color, xytext=(4, 4), textcoords='offset points',
                         fontweight='bold')

    axes[0].set_xlim(0, 512)
    axes[0].set_ylim(512, 0)

    # Right: feature list grouped by type
    axes[1].axis('off')
    axes[1].set_title(f'All OSM features ({len(features_all)} total)', fontsize=11)

    text_lines = []
    by_type = {}
    for f in features_inside:
        t = f['type']
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(f)

    for feat_type, feats in sorted(by_type.items()):
        text_lines.append(f"\n{feat_type.upper()} ({len(feats)}):")
        for f in feats[:10]:
            text_lines.append(f"  {f['name'][:35]} ({f['pixel_x']}, {f['pixel_y']})")
        if len(feats) > 10:
            text_lines.append(f"  ... and {len(feats)-10} more")

    outside_count = len(features_all) - len(features_inside)
    if outside_count > 0:
        text_lines.append(f"\n{outside_count} features outside image chip")

    axes[1].text(0.02, 0.98, '\n'.join(text_lines), transform=axes[1].transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#cccccc'))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for feat_type in sorted(set(f['type'] for f in features_inside)):
        color = TYPE_COLORS.get(feat_type, '#CCCCCC')
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color, markersize=8, label=feat_type))
    if legend_elements:
        axes[0].legend(handles=legend_elements, loc='lower right', fontsize=7)

    event_name = event_data.get('event', '?')
    strategy = event_data.get('strategy', '?')
    fig.suptitle(f"{event_name}\n{strategy} center | {len(features_inside)} OSM features in chip",
                 fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event', nargs='+', type=int, default=None)
    parser.add_argument('--n', type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(RESULTS_FILE):
        print(f"No results at {RESULTS_FILE}")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.event:
        event_ids = [str(e) for e in args.event]
    else:
        candidates = [eid for eid, d in results.items()
                      if 'error' not in d and find_event_image(eid)]
        event_ids = random.sample(candidates, min(args.n, len(candidates)))

    for eid in event_ids:
        if eid not in results or 'error' in results[eid]:
            continue

        data = results[eid]
        center = data['center']
        halfwidth = data.get('halfwidth', 0.05)

        print(f"\nEvent {eid}: {data.get('event', '?')}")
        print(f"  Querying OSM at ({center[0]:.4f}, {center[1]:.4f})...")

        features = get_osm_features(center, halfwidth)
        inside = osm_to_pixels(features, center)

        print(f"  OSM: {len(features)} total, {len(inside)} inside chip")

        by_type = {}
        for f in inside:
            t = f['type']
            by_type[t] = by_type.get(t, 0) + 1
        if by_type:
            print(f"  Types: {by_type}")

        fig = create_osm_viz(eid, data, inside, features)
        if fig:
            out_path = os.path.join(OUT_DIR, f'osm_{eid}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  Saved: {out_path}")

        sleep(1)  # rate limit overpass


if __name__ == '__main__':
    main()
