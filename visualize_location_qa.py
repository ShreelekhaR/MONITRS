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
    eid = str(sample.get('folder_id', ''))
    img_path = get_first_event_image(eid)
    if not img_path:
        return False

    fig, ax = plt.subplots(figsize=(10, 10))
    img = mpimg.imread(img_path)
    ax.imshow(img, extent=[0, 512, 512, 0])
    ax.set_xlim(-100, 612)  # room to show off-image options
    ax.set_ylim(612, -100)

    # Image boundary
    ax.add_patch(Rectangle((0, 0), 512, 512, fill=False, edgecolor='gray',
                            linewidth=1, linestyle='--'))

    # Draw all 4 options as small gray dots with letter labels
    for letter, (x, y) in options.items():
        ax.plot(x, y, 'o', color='#888', markersize=6, alpha=0.6)
        ax.annotate(letter.upper(), (x, y), fontsize=10, color='#333',
                    xytext=(6, -6), textcoords='offset points', fontweight='bold')

    # Ground truth — big green ring
    gx, gy = options[gt_letter]
    ax.plot(gx, gy, 'o', color='#0a0', markersize=20, markerfacecolor='none',
            markeredgewidth=3, label=f'GT ({gt_letter.upper()})')

    # Model predictions
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
            ax.plot(px, py, marker, color=color, markersize=16, markeredgewidth=3)
            legend_items.append(f'{name}: {pl.upper()} {"✓" if pl == gt_letter else "✗"}')
        else:
            legend_items.append(f'{name}: no answer ({pred_text[:30]})')

    ax.set_title(f'Event {eid}: "{loc_name}"\n{" | ".join(legend_items)}',
                  fontsize=11)
    ax.set_xlabel('pixel x')
    ax.set_ylabel('pixel y')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
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
