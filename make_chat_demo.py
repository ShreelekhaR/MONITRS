"""
Build an HTML demo showing side-by-side model predictions on test samples.
Models: Qwen-base, Qwen-ft (ours), Gemini 3.1-flash-lite.

Usage:
    python make_chat_demo.py                       # 30 samples random
    python make_chat_demo.py --n 60 --seed 1
    python make_chat_demo.py --tasks event_type custom  # filter tasks
"""

import json
import os
import re
import argparse
import random
import base64
from os.path import join, isdir
from collections import defaultdict


RESULTS_FILE = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'
CKPT_DIR = 'benchmark_ckpts'
TEST_FILE = 'test_total.json'


MODEL_ORDER = ['qwen-base', 'qwen-ft', 'gemini']
MODEL_LABELS = {
    'qwen-base': 'Qwen2.5-VL (baseline)',
    'qwen-ft':   'Ours (LoRA on MONITRS)',
    'gemini':    'Gemini 3.1-flash-lite',
}
MODEL_COLORS = {
    'qwen-base': '#c0392b',
    'qwen-ft':   '#2ecc71',
    'gemini':    '#3498db',
}


def img_b64(path):
    try:
        with open(path, 'rb') as f:
            data = f.read()
        ext = 'png' if path.endswith('.png') else 'jpeg'
        return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return ''


def get_event_images(eid):
    for suffix in ['', '_firms', '_llm', '_fema']:
        d = join(IMAGES_DIR, f"{eid}{suffix}")
        if isdir(d):
            imgs = []
            for fname in sorted(os.listdir(d)):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    m = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
                    if m:
                        imgs.append({'path': join(d, fname), 'date': m.group(1)})
            return sorted(imgs, key=lambda x: x['date'])
    return []


def clean_question(text):
    q = re.sub(r'<image>|<video>', '', text).strip()
    q = re.sub(r'^This is a sequence of .*?:\s*', '', q).strip()
    return q


def extract_letter(text):
    m = re.search(r'\b([A-Da-d])\b', text)
    return m.group(1).lower() if m else None


def is_mcq(task):
    return task in ('event_type', 'temporal_grounding', 'location_identification', 'multiple_choice')


def build_index_of_predictions():
    """Load predictions from benchmark_ckpts/. Match samples by (question, gt)."""
    preds_by_model = {}
    for name in MODEL_ORDER:
        path = join(CKPT_DIR, f'{name}.json')
        if not os.path.exists(path):
            print(f"Missing {path}")
            continue
        raw = json.load(open(path))
        # raw is {task: [{pred, gt}, ...]} — matches order of samples in test set at that seed
        preds_by_model[name] = raw
    return preds_by_model


def sample_by_task_at_seed(test_data, seed=42):
    """Match the ordering used by benchmark.py sample_by_task."""
    by_task = defaultdict(list)
    for item in test_data:
        by_task[item.get('task', 'unknown')].append(item)
    random.seed(seed)
    for t in by_task:
        random.shuffle(by_task[t])
    return dict(by_task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=30)
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Filter to specific task types')
    parser.add_argument('--seed', type=int, default=42,
                        help='Must match benchmark seed to align predictions with samples')
    parser.add_argument('--out', default='Data/chat_demo.html')
    parser.add_argument('--sample-seed', type=int, default=7,
                        help='Random seed for picking which samples to show in demo')
    args = parser.parse_args()

    # Load
    test_data = json.load(open(TEST_FILE))
    preds_by_model = build_index_of_predictions()
    ordered = sample_by_task_at_seed(test_data, seed=args.seed)

    if not preds_by_model:
        print("No benchmark_ckpts/*.json found")
        return

    # For each task, align samples with model predictions (position-indexed)
    items = []
    for task, samples in ordered.items():
        if args.tasks and task not in args.tasks:
            continue
        for i, s in enumerate(samples):
            entry = {
                'task': task,
                'sample': s,
                'preds': {},
            }
            for name, model_preds in preds_by_model.items():
                if task not in model_preds:
                    continue
                if i < len(model_preds[task]):
                    entry['preds'][name] = model_preds[task][i]['pred']
            if entry['preds']:
                items.append(entry)

    # Pick a random selection, balanced across tasks
    by_task = defaultdict(list)
    for it in items:
        by_task[it['task']].append(it)

    random.seed(args.sample_seed)
    for t in by_task:
        random.shuffle(by_task[t])

    n_tasks = len(by_task)
    per_task = max(1, args.n // n_tasks)
    picks = []
    for t, xs in by_task.items():
        picks.extend(xs[:per_task])
    random.shuffle(picks)
    picks = picks[:args.n]

    print(f"Building demo with {len(picks)} samples from {n_tasks} tasks")

    # Render
    cards_html = []
    for entry in picks:
        s = entry['sample']
        task = entry['task']
        eid = str(s.get('folder_id', ''))
        images = get_event_images(eid)

        convos = s['conversations']
        q = clean_question(convos[0]['value'])
        gt = convos[1]['value'].strip()

        # Determine correctness per model
        gt_letter = extract_letter(gt) if is_mcq(task) else None

        imgs_html = ''
        for img in images[:8]:
            imgs_html += f'<div class="img"><img src="{img_b64(img["path"])}"><div class="date">{img["date"]}</div></div>'

        preds_html = ''
        for name in MODEL_ORDER:
            if name not in entry['preds']:
                continue
            pred = entry['preds'][name]
            correct_class = ''
            correct_badge = ''
            if is_mcq(task) and gt_letter:
                pl = extract_letter(pred)
                if pl == gt_letter:
                    correct_class = 'correct'
                    correct_badge = '<span class="badge-ok">✓</span>'
                elif pl:
                    correct_class = 'wrong'
                    correct_badge = '<span class="badge-no">✗</span>'
            label = MODEL_LABELS.get(name, name)
            color = MODEL_COLORS.get(name, '#666')
            preds_html += f'''
              <div class="pred {correct_class}">
                <div class="pred-header" style="border-left-color:{color}">
                  {label} {correct_badge}
                </div>
                <div class="pred-body">{pred[:400]}</div>
              </div>'''

        cards_html.append(f'''
          <div class="card" data-task="{task}">
            <div class="card-header">
              <span class="task-tag">{task}</span>
              <span class="event-id">event #{eid}</span>
            </div>
            <div class="imgs">{imgs_html}</div>
            <div class="question"><b>Q:</b> {q}</div>
            <div class="gt"><b>Ground truth:</b> {gt[:400]}</div>
            <div class="preds">{preds_html}</div>
          </div>''')

    # Task filter buttons
    task_buttons = ''.join(
        f'<button data-task="{t}" onclick="filter(\'{t}\')">{t} ({len(xs)})</button>'
        for t, xs in sorted(by_task.items()))

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MONITRS: Side-by-side model demo</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; background: #f7f8fa; color: #222; }}
header {{ background: #234; color: white; padding: 20px 32px; }}
header h1 {{ margin: 0 0 6px; font-size: 22px; }}
header p {{ margin: 0; color: #9bd; font-size: 14px; }}
.filters {{ background: white; padding: 12px 32px; border-bottom: 1px solid #ddd;
           position: sticky; top: 0; z-index: 10; display: flex; gap: 8px; flex-wrap: wrap; }}
.filters button {{ background: #f0f2f5; border: 1px solid #ccc; padding: 6px 14px;
                  border-radius: 16px; cursor: pointer; font-size: 12px; }}
.filters button.active {{ background: #234; color: white; border-color: #234; }}
.container {{ max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
.card {{ background: white; margin: 16px 0; padding: 20px; border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.card-header {{ display: flex; justify-content: space-between; margin-bottom: 12px; align-items: center; }}
.task-tag {{ background: #34495e; color: white; padding: 3px 10px; border-radius: 12px;
            font-size: 11px; font-weight: bold; }}
.event-id {{ color: #789; font-size: 12px; }}
.imgs {{ display: flex; gap: 6px; overflow-x: auto; padding: 4px 0; margin-bottom: 12px; }}
.img {{ text-align: center; }}
.img img {{ width: 120px; height: 120px; object-fit: cover; border-radius: 6px; border: 1px solid #ddd; }}
.date {{ font-size: 10px; color: #b44; margin-top: 3px; font-weight: bold; }}
.question {{ background: #eef; padding: 10px 14px; border-radius: 6px; margin: 8px 0;
            font-size: 14px; }}
.gt {{ background: #efe; padding: 10px 14px; border-radius: 6px; margin: 8px 0;
      font-size: 14px; color: #163; }}
.preds {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }}
.pred {{ background: #fafbfc; border-radius: 6px; overflow: hidden;
         border: 1px solid #e0e0e0; }}
.pred.correct {{ border-color: #4a4; background: #f0fbf0; }}
.pred.wrong {{ border-color: #c44; background: #fdf3f3; }}
.pred-header {{ padding: 6px 10px; font-size: 11px; font-weight: bold; color: #345;
                border-left: 4px solid #666; background: white; }}
.pred-body {{ padding: 8px 10px; font-size: 12px; line-height: 1.4; color: #234; min-height: 40px; }}
.badge-ok {{ background: #4a4; color: white; padding: 0 6px; border-radius: 8px; font-size: 10px; margin-left: 4px; }}
.badge-no {{ background: #c44; color: white; padding: 0 6px; border-radius: 8px; font-size: 10px; margin-left: 4px; }}
</style>
<script>
function filter(task) {{
  document.querySelectorAll('.card').forEach(c => {{
    c.style.display = (task === 'all' || c.dataset.task === task) ? 'block' : 'none';
  }});
  document.querySelectorAll('.filters button').forEach(b => {{
    b.classList.toggle('active', b.dataset.task === task);
  }});
}}
</script>
</head><body>
<header>
  <h1>MONITRS: Side-by-side model predictions</h1>
  <p>Same test samples · Qwen2.5-VL baseline vs our LoRA finetune vs Gemini 3.1-flash-lite</p>
</header>
<div class="filters">
  <button data-task="all" onclick="filter('all')" class="active">All ({len(picks)})</button>
  {task_buttons}
</div>
<div class="container">
{''.join(cards_html)}
</div>
</body></html>'''

    with open(args.out, 'w') as f:
        f.write(html)
    print(f"Demo saved: {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
