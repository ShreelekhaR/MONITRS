"""
Overlay geocoded locations on satellite images for visual verification.
Picks the first during-event image for each strategy and draws location markers.

Usage:
    python visualize_points.py
"""

import os
import json
import re
from os.path import join, isdir
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ODIR = 'Data/images'
RESULTS_FILE = 'Data/test_pipeline_results.json'
OUT_DIR = 'Data/visualizations'

COLORS = {
    'visual': (255, 50, 50),      # red for visual events
    'contextual': (50, 150, 255),  # blue for contextual events
    'fema': (255, 255, 0),         # yellow for FEMA center
}


def latlon_to_pixel(lat, lon, center, halfwidth, img_size=512):
    x = int((lon - (center[1] - halfwidth)) / (2 * halfwidth) * img_size)
    y = int(((center[0] + halfwidth) - lat) / (2 * halfwidth) * img_size)
    return x, y


def draw_point(draw, x, y, label, color, radius=6):
    draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                 outline=color, width=2)
    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x + radius + 3, y - 6), label, fill=color, font=font)


def visualize_event(event_idx, event_data, strategy='llm'):
    # Determine center and halfwidth for this strategy
    if strategy == 'fema':
        center = tuple(event_data['fema_center'])
        halfwidth = 0.05
    elif strategy == 'llm':
        center = tuple(event_data.get('llm_center') or event_data['fema_center'])
        llm_est = event_data.get('llm_estimate') or {}
        radius = llm_est.get('radius_km', 10)
        halfwidth = min(max(0.05, radius / 111 / 2), 0.15)
    elif strategy == 'firms':
        center = tuple(event_data.get('firms_center') or event_data['fema_center'])
        halfwidth = 0.05
    else:  # bbox
        center = tuple(event_data.get('geocoded_center') or event_data['fema_center'])
        halfwidth = 0.05

    img_dir = join(ODIR, f"{event_idx}_{strategy}")
    if not isdir(img_dir):
        return None

    # Find first during/event image (handles both old _event_ and new _during_ naming)
    during_imgs = sorted([f for f in os.listdir(img_dir)
                          if ('_during_' in f or '_event_' in f)
                          and (f.endswith('.png') or f.endswith('.jpg'))])
    if not during_imgs:
        return None
    img_path = join(img_dir, during_imgs[0])

    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Draw FEMA center
    fema = tuple(event_data['fema_center'])
    fx, fy = latlon_to_pixel(fema[0], fema[1], center, halfwidth)
    if 0 <= fx < 512 and 0 <= fy < 512:
        draw_point(draw, fx, fy, "FEMA", COLORS['fema'], radius=8)

    # Draw ALL geocoded locations
    geocoded = event_data.get('locations_geocoded', {})
    inside = event_data.get('locations_inside_bbox', [])
    loc_events = event_data.get('location_events', [])

    loc_types = {}
    for le in loc_events:
        loc = le.get('location', '')
        loc_types[loc] = le.get('type', 'visual')

    drawn = 0
    for loc_name, coords in geocoded.items():
        lat, lon = coords[0], coords[1]
        px, py = latlon_to_pixel(lat, lon, center, halfwidth)
        # Draw even if outside image bounds (clamp label)
        if -50 <= px < 562 and -50 <= py < 562:
            ltype = loc_types.get(loc_name, 'visual')
            if loc_name in inside:
                color = COLORS.get(ltype, COLORS['visual'])
            else:
                color = (150, 150, 150)  # grey for outside bbox
            draw_point(draw, max(0, min(px, 511)), max(0, min(py, 511)), loc_name, color)
            drawn += 1

    # If no geocoded points visible, draw LLM center as a crosshair
    llm_c = event_data.get('llm_center')
    if llm_c and drawn == 0:
        lx, ly = latlon_to_pixel(llm_c[0], llm_c[1], center, halfwidth)
        if 0 <= lx < 512 and 0 <= ly < 512:
            draw.line([lx - 10, ly, lx + 10, ly], fill=(0, 255, 0), width=2)
            draw.line([lx, ly - 10, lx, ly + 10], fill=(0, 255, 0), width=2)
            draw_point(draw, lx, ly, "LLM center", (0, 255, 0))

    # Draw FIRMS center if available
    firms_c = event_data.get('firms_center')
    if firms_c:
        frx, fry = latlon_to_pixel(firms_c[0], firms_c[1], center, halfwidth)
        if 0 <= frx < 512 and 0 <= fry < 512:
            draw.line([frx - 10, fry, frx + 10, fry], fill=(255, 165, 0), width=2)
            draw.line([frx, fry - 10, frx, fry + 10], fill=(255, 165, 0), width=2)
            draw_point(draw, frx, fry, "FIRMS", (255, 165, 0))

    # Add legend
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
    legend_y = 10
    for label, color in [("Visual (inside)", COLORS['visual']),
                         ("Contextual (inside)", COLORS['contextual']),
                         ("Outside bbox", (150, 150, 150)),
                         ("FEMA center", COLORS['fema']),
                         ("LLM center", (0, 255, 0)),
                         ("FIRMS center", (255, 165, 0))]:
        draw.rectangle([10, legend_y, 22, legend_y + 12], fill=color)
        draw.text((26, legend_y - 1), label, fill=(255, 255, 255), font=font)
        legend_y += 16

    # Add title
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', during_imgs[0])
    date_str = date_match.group(1) if date_match else '?'
    title = f"Event {event_idx} | {strategy} | {date_str}"
    draw.text((10, 490), title, fill=(255, 255, 255), font=font)

    return img


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"No results at {RESULTS_FILE}")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Pick first 5 events that have results
    count = 0
    for event_idx, event_data in sorted(results.items(), key=lambda x: int(x[0])):
        if 'error' in event_data:
            continue
        if count >= 5:
            break

        print(f"\nEvent {event_idx}: {event_data['event']} ({event_data['type']})")

        strategies = ['fema', 'llm', 'bbox', 'firms']
        imgs = []
        labels = []

        for strat in strategies:
            img = visualize_event(event_idx, event_data, strat)
            if img:
                imgs.append(img)
                labels.append(strat)
                out_path = join(OUT_DIR, f"{event_idx}_{strat}_annotated.png")
                img.save(out_path)
                print(f"  Saved {out_path}")

        # Create side-by-side comparison if we have multiple
        if len(imgs) >= 2:
            total_w = 512 * len(imgs) + 10 * (len(imgs) - 1)
            canvas = Image.new('RGB', (total_w, 540), (30, 30, 30))
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            except Exception:
                font = ImageFont.load_default()
            for i, (img, label) in enumerate(zip(imgs, labels)):
                x_offset = i * (512 + 10)
                canvas.paste(img, (x_offset, 20))
                draw = ImageDraw.Draw(canvas)
                draw.text((x_offset + 200, 2), label.upper(), fill=(255, 255, 255), font=font)

            comp_path = join(OUT_DIR, f"{event_idx}_comparison.png")
            canvas.save(comp_path)
            print(f"  Comparison: {comp_path}")

        count += 1

    print(f"\nVisualizations saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
