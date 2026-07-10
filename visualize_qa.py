"""
Visualize MONITRS v2 QA samples: satellite image sequence + question/answer pairs.

Usage:
    python visualize_qa.py                    # 5 random samples
    python visualize_qa.py --n 10             # 10 samples
    python visualize_qa.py --event 0          # specific event
    python visualize_qa.py --type Fire        # filter by type
    python visualize_qa.py --task temporal_grounding  # filter by QA task
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
import matplotlib.gridspec as gridspec


ODIR = 'Data/images'
RESULTS_FILE = 'Data/events_processed.json'
QA_FILES = ['train_total.json', 'test_total.json',
            'new_train_multiple_choice.json', 'new_test_multiple_choice.json',
            'train_generated_multiple_choice_q_a.json', 'test_generated_multiple_choice_q_a.json',
            'train_generated_q_a.json', 'test_generated_q_a.json']
OUT_DIR = 'Data/visualizations'


def load_qa_by_event():
    qa_by_event = {}
    for qa_file in QA_FILES:
        if not os.path.exists(qa_file):
            continue
        data = json.load(open(qa_file))
        for item in data:
            eid = str(item.get('folder_id', ''))
            if eid not in qa_by_event:
                qa_by_event[eid] = []
            qa_by_event[eid].append(item)
    return qa_by_event


def get_event_images(event_idx):
    images = []
    for suffix in ['', '_firms', '_llm', '_fema']:
        img_dir = join(ODIR, f"{event_idx}{suffix}")
        if isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
                    if date_match:
                        images.append({
                            'path': join(img_dir, fname),
                            'date': date_match.group(1),
                        })
            break
    return sorted(images, key=lambda x: x['date'])


def create_qa_viz(event_idx, event_data, qa_items, max_images=6, max_qa=4):
    images = get_event_images(event_idx)
    if not images:
        return None

    # Select images evenly
    if len(images) > max_images:
        step = len(images) // max_images
        images = images[::step][:max_images]

    # Select QA items (mix of types if possible)
    tasks_seen = set()
    selected_qa = []
    for qa in qa_items:
        task = qa.get('task', '')
        if task not in tasks_seen and len(selected_qa) < max_qa:
            tasks_seen.add(task)
            selected_qa.append(qa)
    # Fill remaining with any
    for qa in qa_items:
        if len(selected_qa) >= max_qa:
            break
        if qa not in selected_qa:
            selected_qa.append(qa)

    if not selected_qa:
        return None

    n_imgs = len(images)
    n_qa = len(selected_qa)

    # Layout: images on top, QA below
    fig = plt.figure(figsize=(max(3.5 * n_imgs, 12), 5 + 2.5 * n_qa))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, max(1, n_qa * 1.2)], hspace=0.15)

    # Top: image row
    gs_imgs = gridspec.GridSpecFromSubplotSpec(1, n_imgs, subplot_spec=gs[0], wspace=0.05)

    event_name = event_data.get('event', '?')
    event_type = event_data.get('type', '?')
    state = event_data.get('state', '')
    county = event_data.get('county', '')

    fig.suptitle(f"{event_name}\n{event_type}  |  {county}, {state}  |  {n_imgs} images, {len(qa_items)} QA pairs",
                 fontsize=12, fontweight='bold', y=0.98)

    phase_colors = {'pre': '#64B4FF', 'during': '#FF6464', 'post': '#64FF64', 'caption': '#FF6464'}

    for i, img_info in enumerate(images):
        ax = fig.add_subplot(gs_imgs[0, i])
        try:
            img = mpimg.imread(img_info['path'])
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'err', ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

        phase = 'pre' if '_pre_' in img_info['path'] else ('post' if '_post_' in img_info['path'] else 'during')
        color = phase_colors.get(phase, '#FF6464')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_title(img_info['date'], fontsize=8, color=color, fontweight='bold')

    # Bottom: QA pairs
    ax_qa = fig.add_subplot(gs[1])
    ax_qa.axis('off')

    qa_text = ""
    for i, qa in enumerate(selected_qa):
        convos = qa.get('conversations', [])
        question = convos[0]['value'] if len(convos) > 0 else ''
        answer = convos[1]['value'] if len(convos) > 1 else ''
        task = qa.get('task', 'unknown')

        # Clean up question (remove <video> tokens)
        question = question.replace('<video>', '').replace('This is a sequence of satellite images:\n', '')
        question = question.replace('This is a sequence of sentinel-2 satellite images, centered at', 'Images centered at')
        question = question.strip()

        q_wrapped = textwrap.fill(f"Q{i+1} [{task}]: {question}", width=120)
        a_wrapped = textwrap.fill(f"A{i+1}: {answer}", width=120)

        qa_text += f"{q_wrapped}\n{a_wrapped}\n\n"

    ax_qa.text(0.02, 0.98, qa_text.strip(), transform=ax_qa.transAxes,
               fontsize=7, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#cccccc'))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--event', type=int, default=None)
    parser.add_argument('--type', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(RESULTS_FILE):
        print(f"No results at {RESULTS_FILE}")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    qa_by_event = load_qa_by_event()
    print(f"Loaded QA for {len(qa_by_event)} events")

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.event is not None:
        event_ids = [str(args.event)]
    else:
        candidates = []
        for eid, data in results.items():
            if 'error' in data:
                continue
            if args.type and data.get('type') != args.type:
                continue
            if eid not in qa_by_event or not qa_by_event[eid]:
                continue
            if args.task:
                has_task = any(q.get('task') == args.task for q in qa_by_event[eid])
                if not has_task:
                    continue
            if get_event_images(eid):
                candidates.append(eid)
        event_ids = random.sample(candidates, min(args.n, len(candidates)))

    print(f"Visualizing {len(event_ids)} events...")

    for eid in event_ids:
        if eid not in results or 'error' in results[eid]:
            print(f"  Event {eid}: skipped")
            continue
        data = results[eid]
        qa_items = qa_by_event.get(eid, [])

        if args.task:
            qa_items = [q for q in qa_items if q.get('task') == args.task]

        if not qa_items:
            print(f"  Event {eid}: no QA")
            continue

        fig = create_qa_viz(eid, data, qa_items)
        if fig:
            out_path = join(OUT_DIR, f'qa_{eid}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  Event {eid} ({data['event'][:30]}): {len(qa_items)} QA, saved")
        else:
            print(f"  Event {eid}: no images")

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
