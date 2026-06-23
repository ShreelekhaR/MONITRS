"""
Visualize MONITRS v2 samples: image sequences with captions overlaid.
Creates a horizontal strip of images with captions below each frame.

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
from os.path import join, isdir
from PIL import Image, ImageDraw, ImageFont
import textwrap


ODIR = 'Data/images'
RESULTS_FILE = 'Data/events_processed.json'
OUT_DIR = 'Data/visualizations'


def get_font(size=12):
    for path in [
        '/System/Library/Fonts/Helvetica.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


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
        images.append({
            'path': join(img_dir, fname),
            'date': date,
            'phase': phase,
        })
    return images


def create_sample_viz(event_idx, event_data, max_frames=8):
    images = get_event_images(event_idx)
    if not images:
        return None

    captions = parse_captions(event_data.get('captions', ''))

    # Select frames: 1 pre + up to max_frames-2 during + 1 post
    pre = [img for img in images if img['phase'] == 'pre']
    during = [img for img in images if img['phase'] == 'during']
    post = [img for img in images if img['phase'] == 'post']

    selected = []
    if pre:
        selected.append(pre[-1])  # last pre-event (closest to event)
    if during:
        step = max(1, len(during) // min(len(during), max_frames - 2))
        selected.extend(during[::step][:max_frames - 2])
    if post:
        selected.append(post[0])  # first post-event

    if not selected:
        return None

    # Layout
    thumb_size = 256
    caption_height = 80
    padding = 5
    n_frames = len(selected)

    total_w = n_frames * (thumb_size + padding) - padding + 20
    header_h = 60
    total_h = header_h + thumb_size + caption_height + 20

    canvas = Image.new('RGB', (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font_title = get_font(16)
    font_caption = get_font(10)
    font_date = get_font(12)

    # Header
    event_name = event_data.get('event', '?')
    event_type = event_data.get('type', '?')
    strategy = event_data.get('strategy', '?')
    state = event_data.get('state', '')
    county = event_data.get('county', '')
    draw.text((10, 8), f"Event {event_idx}: {event_name}", fill=(255, 255, 255), font=font_title)
    draw.text((10, 30), f"{event_type} | {county}, {state} | Strategy: {strategy}",
              fill=(180, 180, 180), font=font_caption)

    # Frames
    phase_colors = {'pre': (100, 180, 255), 'during': (255, 100, 100), 'post': (100, 255, 100)}

    for i, img_info in enumerate(selected):
        x = 10 + i * (thumb_size + padding)
        y = header_h

        # Image
        try:
            img = Image.open(img_info['path']).resize((thumb_size, thumb_size))
            canvas.paste(img, (x, y))
        except Exception:
            continue

        # Phase indicator bar
        color = phase_colors.get(img_info['phase'], (200, 200, 200))
        draw.rectangle([x, y, x + thumb_size, y + 3], fill=color)

        # Date label
        date_label = f"{img_info['date']} ({img_info['phase']})"
        draw.text((x + 2, y + thumb_size + 4), date_label, fill=color, font=font_date)

        # Caption
        caption = captions.get(img_info['date'], '')
        if caption:
            wrapped = textwrap.fill(caption, width=35)
            lines = wrapped.split('\n')[:3]
            for j, line in enumerate(lines):
                draw.text((x + 2, y + thumb_size + 20 + j * 14), line,
                          fill=(220, 220, 220), font=font_caption)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5, help='Number of samples')
    parser.add_argument('--events', nargs='+', type=int, default=None, help='Specific event IDs')
    parser.add_argument('--type', type=str, default=None, help='Filter by event type (Fire, Hurricane, etc)')
    args = parser.parse_args()

    if not os.path.exists(RESULTS_FILE):
        print(f"No results at {RESULTS_FILE}")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Select events
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

        viz = create_sample_viz(eid, data)
        if viz:
            out_path = join(OUT_DIR, f'sample_{eid}.png')
            viz.save(out_path)
            print(f"  Event {eid} ({data['event'][:30]}): saved to {out_path}")
        else:
            print(f"  Event {eid}: no images found")

    print(f"\nVisualizations saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
