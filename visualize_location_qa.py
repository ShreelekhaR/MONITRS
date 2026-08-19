"""
Visualize location_identification test samples: image + all option pixel coords +
ground truth marker + each model's prediction.

Usage:
    python visualize_location_qa.py                       # 20 random samples
    python visualize_location_qa.py --n 40 --html
    python visualize_location_qa.py --sample 5            # specific sample index
"""

import json
import os
import re
import argparse
import random
import base64
from os.path import join, isdir
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Rectangle
import math
import io
import urllib.request
from PIL import Image as PILImage


CKPT_DIR = 'benchmark_ckpts'
IMAGES_DIR = 'Data/images'
TEST_FILE = 'test_total.json'
OUT_DIR = 'Data/loc_viz'

MODEL_ORDER = ['qwen-base', 'qwen-ft', 'gemini']
MODEL_COLORS = {
    'qwen-base': '#e74c3c',
    'qwen-ft':   '#2ecc71',
    'gemini':    '#3498db',
}


def latlon_to_tile(lat, lon, zoom):
    """OSM slippy tile x, y (float) for lat/lon at zoom."""
    n = 2 ** zoom
    xt = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    yt = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n
    return xt, yt


def tile_to_latlon(x, y, zoom):
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad), lon


def fetch_osm_map(center_lat, center_lon, zoom=13, tile_size=256, grid=3):
    """Fetch a grid x grid tile mosaic centered on (lat, lon). Return (PIL image, extent_latlon).
    extent_latlon = (west_lon, east_lon, south_lat, north_lat)
    """
    xt, yt = latlon_to_tile(center_lat, center_lon, zoom)
    x0 = int(xt) - grid // 2
    y0 = int(yt) - grid // 2

    canvas = PILImage.new('RGB', (tile_size * grid, tile_size * grid), 'white')
    for dy in range(grid):
        for dx in range(grid):
            tx, ty = x0 + dx, y0 + dy
            url = f'https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png'
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'MONITRS-viz/1.0'})
                with urllib.request.urlopen(req, timeout=10) as r:
                    tile = PILImage.open(io.BytesIO(r.read())).convert('RGB')
                canvas.paste(tile, (dx * tile_size, dy * tile_size))
            except Exception:
                pass

    # Compute lat/lon extent
    nw_lat, nw_lon = tile_to_latlon(x0, y0, zoom)
    se_lat, se_lon = tile_to_latlon(x0 + grid, y0 + grid, zoom)
    extent = (nw_lon, se_lon, se_lat, nw_lat)
    return canvas, extent


def parse_location_latlon(q):
    """Extract (lat, lon) from question like 'Where is X (18.22, -66.43)?'"""
    m = re.search(r'Where is .+?\s*\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', q)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def get_first_event_image(eid):
    for suffix in ['', '_firms', '_llm', '_fema']:
        d = join(IMAGES_DIR, f"{eid}{suffix}")
        if isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    return join(d, fname)
    return None


def extract_letter(text):
    m = re.search(r'\b([A-Da-d])\b', text)
    return m.group(1).lower() if m else None


def parse_options_from_question(q):
    """Parse 'a. (x, y)  b. (x, y) ...' style options → {letter: (x, y)}"""
    opts = {}
    for m in re.finditer(r'\b([a-dA-D])[\.\)]\s*\((-?\d+),\s*(-?\d+)\)', q):
        letter = m.group(1).lower()
        opts[letter] = (int(m.group(2)), int(m.group(3)))
    return opts


def parse_location_name(q):
    m = re.search(r'Where is (.+?)\s*\(', q)
    return m.group(1).strip() if m else 'unknown'


def load_predictions():
    preds = {}
    for name in MODEL_ORDER:
        p = join(CKPT_DIR, f'{name}.json')
        if os.path.exists(p):
            preds[name] = json.load(p if False else open(p))
    return preds


def load_test_samples():
    return json.load(open(TEST_FILE))


def sample_by_task_at_seed(test_data, seed=42):
    """Match benchmark.py's ordering."""
    by_task = defaultdict(list)
    for item in test_data:
        by_task[item.get('task', '?')].append(item)
    random.seed(seed)
    for t in by_task:
        random.shuffle(by_task[t])
    return dict(by_task)


def visualize_sample(sample, preds_at_index, out_path):
    convos = sample['conversations']
    q = convos[0]['value']
    gt = convos[1]['value'].strip()
    gt_letter = extract_letter(gt)

    options = parse_options_from_question(q)
    if not options or gt_letter not in options:
        return False

    loc_name = parse_location_name(q)
    loc_lat, loc_lon = parse_location_latlon(q)
    eid = str(sample.get('folder_id', ''))
    img_path = get_first_event_image(eid)
    if not img_path:
        return False

    fig, (ax_sat, ax_map) = plt.subplots(1, 2, figsize=(18, 9))

    # ── LEFT: satellite chip with option pixels + GT + predictions ──
    img = mpimg.imread(img_path)
    ax_sat.imshow(img, extent=[0, 512, 512, 0])
    all_xs = [x for x, _ in options.values()]
    all_ys = [y for _, y in options.values()]
    pad = 20
    ax_sat.set_xlim(min(0, min(all_xs)) - pad, max(512, max(all_xs)) + pad)
    ax_sat.set_ylim(max(512, max(all_ys)) + pad, min(0, min(all_ys)) - pad)

    ax_sat.add_patch(Rectangle((0, 0), 512, 512, fill=False, edgecolor='gray',
                                linewidth=1, linestyle='--'))
    for letter, (x, y) in options.items():
        ax_sat.plot(x, y, 'o', color='#888', markersize=6, alpha=0.6)
        ax_sat.annotate(letter.upper(), (x, y), fontsize=10, color='#333',
                        xytext=(6, -6), textcoords='offset points', fontweight='bold')

    gx, gy = options[gt_letter]
    ax_sat.plot(gx, gy, 'o', color='#0a0', markersize=20, markerfacecolor='none',
                markeredgewidth=3)

    legend_items = []
    for name in MODEL_ORDER:
        if name not in preds_at_index:
            continue
        pred_text = preds_at_index[name]
        pl = extract_letter(pred_text)
        color = MODEL_COLORS.get(name, '#666')
        if pl and pl in options:
            px, py = options[pl]
            marker = 'X' if pl == gt_letter else 'x'
            ax_sat.plot(px, py, marker, color=color, markersize=16, markeredgewidth=3)
            legend_items.append(f'{name}: {pl.upper()} {"✓" if pl == gt_letter else "✗"}')
        else:
            legend_items.append(f'{name}: no answer ({pred_text[:30]})')

    ax_sat.set_title('Satellite chip + option pixels', fontsize=11)
    ax_sat.set_xlabel('pixel x')
    ax_sat.set_ylabel('pixel y')
    ax_sat.grid(True, alpha=0.2)

    # ── RIGHT: OSM map showing lat/lon of the location ──
    if loc_lat is not None and loc_lon is not None:
        try:
            osm_img, extent = fetch_osm_map(loc_lat, loc_lon, zoom=12, grid=3)
            ax_map.imshow(osm_img, extent=extent, origin='upper')
            ax_map.plot(loc_lon, loc_lat, 'o', color='#0a0', markersize=20,
                        markerfacecolor='none', markeredgewidth=3)
            ax_map.plot(loc_lon, loc_lat, 'x', color='#0a0', markersize=12, markeredgewidth=2)
            ax_map.set_title(f'OSM map @ ({loc_lat:.4f}, {loc_lon:.4f})', fontsize=11)
            ax_map.set_xlabel('longitude')
            ax_map.set_ylabel('latitude')
        except Exception as e:
            ax_map.text(0.5, 0.5, f'Map fetch failed:\n{e}',
                        ha='center', va='center', transform=ax_map.transAxes)
    else:
        ax_map.text(0.5, 0.5, 'No lat/lon in question',
                    ha='center', va='center', transform=ax_map.transAxes)

    fig.suptitle(f'Event {eid}: "{loc_name}"\n{" | ".join(legend_items)}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--sample', type=int, default=None, help='Specific sample index')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--html', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    test_data = load_test_samples()
    by_task = sample_by_task_at_seed(test_data, seed=args.seed)
    if 'location_identification' not in by_task:
        print("No location_identification samples found")
        return
    samples = by_task['location_identification']
    preds_all = load_predictions()

    if args.sample is not None:
        indices = [args.sample]
    else:
        indices = list(range(min(args.n, len(samples))))

    print(f"Rendering {len(indices)} location_identification samples...")
    generated = []
    for i in indices:
        s = samples[i]
        preds_here = {}
        for name, raw in preds_all.items():
            arr = raw.get('location_identification', [])
            if i < len(arr):
                preds_here[name] = arr[i]['pred']
        out = join(OUT_DIR, f'loc_{i:03d}.png')
        if visualize_sample(s, preds_here, out):
            generated.append(out)
            print(f"  {out}")

    if args.html:
        html_path = join(OUT_DIR, 'index.html')
        parts = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
                 '<title>Location QA visualization</title>',
                 '<style>body{font-family:sans-serif;background:#f4f4f4;padding:20px}',
                 '.card{background:#fff;margin:12px 0;padding:12px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08)}',
                 'img{max-width:800px}</style></head><body>',
                 f'<h1>Location QA: {len(generated)} samples</h1>']
        for g in generated:
            with open(g, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            parts.append(f'<div class="card"><img src="data:image/png;base64,{b64}"></div>')
        parts.append('</body></html>')
        with open(html_path, 'w') as f:
            f.write('\n'.join(parts))
        print(f"HTML: {html_path}")


if __name__ == '__main__':
    main()
