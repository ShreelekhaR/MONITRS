"""
Visualize MONITRS v2 samples: image sequences with captions as matplotlib figures.

Usage:
    python visualize_samples.py                    # 5 random samples
    python visualize_samples.py --n 10             # 10 samples
    python visualize_samples.py --events 0 1 5 9   # specific events
    python visualize_samples.py --type Fire        # only fire events
"""

import os
import re
import json
import argparse
import random
import textwrap
from os.path import join, isdir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


ODIR = 'Data/images'
RESULTS_FILE = 'Data/events_processed.json'
OUT_DIR = 'Data/visualizations'

PHASE_COLORS = {'pre': '#64B4FF', 'during': '#FF6464', 'post': '#64FF64'}


def parse_captions(caption_text):
    captions = {}
    if not caption_text:
        return captions
    for line in caption_text.strip().split('\n'):
        match = re.match(r'(\d{4}-\d{2}-\d{2}):\s*(.*)', line.strip())
        if match:
            captions[match.group(1)] = match.group(2)
    return captions


def get_event_images(event_idx):
    img_dir = join(ODIR, str(event_idx))
    if not isdir(img_dir):
        return []
    images = []
    for fname in sorted(os.listdir(img_dir)):
        if not (fname.endswith('.png') or fname.endswith('.jpg')):
            continue
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
        if not date_match:
            continue
        date = date_match.group(1)
        if '_pre_' in fname:
            phase = 'pre'
        elif '_post_' in fname:
            phase = 'post'
        else:
            phase = 'during'
        images.append({'path': join(img_dir, fname), 'date': date, 'phase': phase})
    return images


def create_sample_viz(event_idx, event_data, max_frames=8):
    images = get_event_images(event_idx)
    if not images:
        return None

    captions = parse_captions(event_data.get('captions', ''))

    pre = [img for img in images if img['phase'] == 'pre']
    during = [img for img in images if img['phase'] == 'during']
    post = [img for img in images if img['phase'] == 'post']

    selected = []
    if pre:
        selected.append(pre[-1])
    if during:
        step = max(1, len(during) // min(len(during), max_frames - 2))
        selected.extend(during[::step][:max_frames - 2])
    if post:
        selected.append(post[0])

    # Filter out images that can't be loaded
    valid = []
    for img_info in selected:
        try:
            img = mpimg.imread(img_info['path'])
            if img is not None and img.size > 0:
                valid.append(img_info)
        except Exception:
            continue
    selected = valid

    if not selected:
        return None

    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5.5))
    if n == 1:
        axes = [axes]

    event_name = event_data.get('event', '?')
    event_type = event_data.get('type', '?')
    strategy = event_data.get('strategy', '?')
    state = event_data.get('state', '')
    county = event_data.get('county', '')

    fig.suptitle(f"{event_name}\n{event_type}  |  {county}, {state}  |  Strategy: {strategy}",
                 fontsize=13, fontweight='bold', y=0.98)

    for i, (ax, img_info) in enumerate(zip(axes, selected)):
        img = mpimg.imread(img_info['path'])
        ax.imshow(img)

        ax.set_xticks([])
        ax.set_yticks([])

        color = PHASE_COLORS.get(img_info['phase'], '#CCCCCC')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        ax.set_title(f"{img_info['date']}  ({img_info['phase']})",
                     fontsize=10, color=color, fontweight='bold', pad=6)

        # Find caption: exact match, then nearest earlier, then nearest overall
        caption = captions.get(img_info['date'], '')
        if not caption and captions:
            from datetime import datetime
            img_dt = datetime.strptime(img_info['date'], '%Y-%m-%d')
            # Prefer the latest caption that's on or before this image date
            earlier = {d: c for d, c in captions.items()
                       if datetime.strptime(d, '%Y-%m-%d') <= img_dt}
            if earlier:
                best = max(earlier.keys())
                caption = earlier[best]
            else:
                # Fall back to nearest overall
                closest = min(captions.keys(),
                              key=lambda d: abs((datetime.strptime(d, '%Y-%m-%d') - img_dt).days))
                caption = captions[closest]
        if caption:
            wrapped = textwrap.fill(caption, width=40)
            ax.set_xlabel(wrapped, fontsize=7, color='#444444',
                          ha='center', labelpad=8, wrap=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--events', nargs='+', type=int, default=None)
    parser.add_argument('--type', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(RESULTS_FILE):
        print(f"No results at {RESULTS_FILE}")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.events:
        event_ids = [str(e) for e in args.events]
    else:
        candidates = []
        for eid, data in results.items():
            if 'error' in data:
                continue
            if args.type and data.get('type') != args.type:
                continue
            img_dir = join(ODIR, str(eid))
            if isdir(img_dir) and len(os.listdir(img_dir)) >= 3:
                candidates.append(eid)
        if not candidates:
            print("No events with images found")
            return
        event_ids = random.sample(candidates, min(args.n, len(candidates)))

    print(f"Visualizing {len(event_ids)} events...")

    for eid in event_ids:
        if eid not in results:
            print(f"  Event {eid}: not in results")
            continue
        data = results[eid]
        if 'error' in data:
            print(f"  Event {eid}: has error, skipping")
            continue

        fig = create_sample_viz(eid, data)
        if fig:
            out_path = join(OUT_DIR, f'sample_{eid}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            print(f"  Event {eid} ({data['event'][:30]}): saved to {out_path}")
        else:
            print(f"  Event {eid}: no images found")

    print(f"\nVisualizations saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
