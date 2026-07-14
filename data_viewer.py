"""
Interactive data viewer for MONITRS v2 dataset.
Shows event info, images, captions, and QA pairs in a clean format.

Usage:
    python data_viewer.py                  # random 5 events
    python data_viewer.py --event 0 1 88   # specific events
    python data_viewer.py --type Fire      # filter by type
    python data_viewer.py --html           # generate HTML viewer
"""

import json
import os
import re
import argparse
import random
import base64
from os.path import join, isdir


RESULTS_FILE = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'


def get_event_images(eid):
    images = []
    for suffix in ['', '_firms', '_llm', '_fema']:
        img_dir = join(IMAGES_DIR, f"{eid}{suffix}")
        if isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    m = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
                    if m:
                        images.append({
                            'path': join(img_dir, fname),
                            'date': m.group(1),
                            'fname': fname,
                        })
            break
    return sorted(images, key=lambda x: x['date'])


def parse_captions(text):
    caps = {}
    if not text:
        return caps
    for line in text.strip().split('\n'):
        m = re.match(r'(\d{4}-\d{2}-\d{2}):\s*(.*)', line.strip())
        if m:
            caps[m.group(1)] = m.group(2)
    return caps


def load_qa():
    qa = {}
    for f in ['train_total.json', 'test_total.json']:
        if os.path.exists(f):
            for item in json.load(open(f)):
                eid = str(item.get('folder_id', ''))
                if eid not in qa:
                    qa[eid] = []
                qa[eid].append(item)
    return qa


def print_event(eid, data, qa_items):
    images = get_event_images(eid)
    captions = parse_captions(data.get('captions', ''))

    print(f"\n{'='*70}")
    print(f"EVENT {eid}: {data.get('event', '?')}")
    print(f"  Type: {data.get('type', '?')} | {data.get('county', '')}, {data.get('state', '')}")
    print(f"  Dates: {data.get('start_date', '?')} to {data.get('end_date', '?')}")
    print(f"  Strategy: {data.get('strategy', '?')}")
    print(f"  Center: ({data['center'][0]:.4f}, {data['center'][1]:.4f})")
    print(f"  Images: {len(images)} | QA pairs: {len(qa_items)}")

    print(f"\n  IMAGES:")
    for img in images:
        cap = captions.get(img['date'], '')
        cap_preview = cap[:80] + '...' if len(cap) > 80 else cap
        print(f"    {img['date']}: {img['fname']}")
        if cap_preview:
            print(f"      Caption: {cap_preview}")

    print(f"\n  QA PAIRS:")
    for i, item in enumerate(qa_items[:6]):
        convos = item.get('conversations', [])
        q = convos[0]['value'][:100] if len(convos) > 0 else ''
        a = convos[1]['value'][:100] if len(convos) > 1 else ''
        q = q.replace('<video>', '').replace('This is a sequence of satellite images:\n', '').strip()
        task = item.get('task', '?')
        print(f"    Q{i+1} [{task}]: {q}")
        print(f"    A{i+1}: {a}")
        print()


def generate_html(events_to_show, results, qa_by_event, output='Data/viewer.html'):
    html = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
            '<title>MONITRS v2 Dataset Viewer</title>',
            '<style>',
            'body { font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }',
            '.event { background: white; border-radius: 12px; padding: 24px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }',
            '.event-header { font-size: 20px; font-weight: bold; margin-bottom: 4px; }',
            '.event-meta { color: #666; font-size: 14px; margin-bottom: 16px; }',
            '.images { display: flex; gap: 8px; overflow-x: auto; padding: 8px 0; }',
            '.img-card { text-align: center; min-width: 180px; }',
            '.img-card img { width: 180px; height: 180px; object-fit: cover; border-radius: 8px; border: 2px solid #ddd; }',
            '.img-card .date { font-size: 12px; font-weight: bold; color: #e44; margin-top: 4px; }',
            '.img-card .caption { font-size: 10px; color: #666; margin-top: 2px; max-width: 180px; }',
            '.qa { background: #f9f9f9; border-radius: 8px; padding: 16px; margin-top: 16px; }',
            '.qa-item { margin-bottom: 12px; }',
            '.qa-q { font-weight: bold; font-size: 13px; }',
            '.qa-a { font-size: 13px; color: #333; margin-top: 2px; }',
            '.qa-task { font-size: 11px; color: white; background: #4488cc; padding: 2px 8px; border-radius: 10px; }',
            'h1 { text-align: center; }',
            '.stats { text-align: center; color: #666; margin-bottom: 30px; }',
            '</style></head><body>',
            '<h1>MONITRS v2 Dataset Viewer</h1>',
            f'<div class="stats">{len(events_to_show)} events shown</div>']

    for eid in events_to_show:
        if eid not in results or 'error' in results[eid]:
            continue
        data = results[eid]
        images = get_event_images(eid)
        captions = parse_captions(data.get('captions', ''))
        qa_items = qa_by_event.get(eid, [])

        html.append('<div class="event">')
        html.append(f'<div class="event-header">{data.get("event", "?")}</div>')
        html.append(f'<div class="event-meta">{data.get("type", "")} | {data.get("county", "")}, {data.get("state", "")} | {data.get("start_date", "")} to {data.get("end_date", "")} | {len(images)} images | {len(qa_items)} QA</div>')

        # Images
        html.append('<div class="images">')
        for img in images[:8]:
            cap = captions.get(img['date'], '')[:60]
            try:
                with open(img['path'], 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                ext = 'png' if img['path'].endswith('.png') else 'jpeg'
                html.append(f'<div class="img-card"><img src="data:image/{ext};base64,{b64}"><div class="date">{img["date"]}</div><div class="caption">{cap}</div></div>')
            except Exception:
                html.append(f'<div class="img-card"><div style="width:180px;height:180px;background:#eee;border-radius:8px"></div><div class="date">{img["date"]}</div></div>')
        html.append('</div>')

        # QA
        if qa_items:
            html.append('<div class="qa">')
            for item in qa_items[:4]:
                convos = item.get('conversations', [])
                q = convos[0]['value'][:200] if len(convos) > 0 else ''
                a = convos[1]['value'][:200] if len(convos) > 1 else ''
                q = q.replace('<video>', '').replace('This is a sequence of satellite images:\n', '').strip()
                task = item.get('task', '?')
                html.append(f'<div class="qa-item"><span class="qa-task">{task}</span><div class="qa-q">Q: {q}</div><div class="qa-a">A: {a}</div></div>')
            html.append('</div>')

        html.append('</div>')

    html.append('</body></html>')

    with open(output, 'w') as f:
        f.write('\n'.join(html))
    print(f"HTML viewer saved to {output} ({os.path.getsize(output)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event', nargs='+', type=int, default=None)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--type', type=str, default=None)
    parser.add_argument('--html', action='store_true')
    args = parser.parse_args()

    results = json.load(open(RESULTS_FILE))
    qa_by_event = load_qa()

    if args.event:
        event_ids = [str(e) for e in args.event]
    else:
        candidates = [eid for eid, d in results.items()
                      if 'error' not in d and get_event_images(eid)]
        if args.type:
            candidates = [eid for eid in candidates if results[eid].get('type') == args.type]
        event_ids = random.sample(candidates, min(args.n, len(candidates)))

    if args.html:
        generate_html(event_ids, results, qa_by_event)
    else:
        for eid in event_ids:
            if eid in results and 'error' not in results[eid]:
                print_event(eid, results[eid], qa_by_event.get(eid, []))


if __name__ == '__main__':
    main()
