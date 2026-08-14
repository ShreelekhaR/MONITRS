"""
Generate HTML slides documenting MONITRS v2 pipeline improvements.

Creates slides showing:
  - Pipeline changes (v1 → v2)
  - Center strategy comparison (FIRMS vs LLM vs FEMA)
  - Caption/QA quality improvements
  - Sample events with images
  - Benchmark results

Usage:
    python make_slides.py                    # generates Data/slides.html
    python make_slides.py --n-samples 5      # more sample events
"""

import json
import os
import re
import argparse
import base64
import random
from os.path import join, isdir
from collections import defaultdict, Counter


RESULTS_FILE = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'
CKPT_DIR = 'benchmark_ckpts'


def img_b64(path):
    try:
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = 'png' if path.endswith('.png') else 'jpeg'
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return ''


def get_event_images(eid):
    for suffix in ['', '_firms', '_llm', '_fema']:
        d = join(IMAGES_DIR, f"{eid}{suffix}")
        if isdir(d):
            paths = []
            for fname in sorted(os.listdir(d)):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    m = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
                    if m:
                        paths.append({'path': join(d, fname), 'date': m.group(1)})
            return sorted(paths, key=lambda x: x['date'])
    return []


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
    qa = defaultdict(list)
    for f in ['train_total.json', 'test_total.json']:
        if os.path.exists(f):
            for item in json.load(open(f)):
                qa[str(item.get('folder_id', ''))].append(item)
    return dict(qa)


def load_bench_results():
    """Try to load raw ckpt predictions and score."""
    import sys
    sys.path.insert(0, 'Train')
    try:
        from benchmark import compute_mcq_accuracy, compute_open_metrics, TASK_OPEN
    except Exception:
        return None

    scores = {}
    for f in sorted(os.listdir(CKPT_DIR)) if os.path.isdir(CKPT_DIR) else []:
        if not f.endswith('.json'):
            continue
        name = f.replace('.json', '')
        try:
            raw = json.load(open(join(CKPT_DIR, f)))
        except Exception:
            continue
        s = {}
        for task, pairs in raw.items():
            preds = [p['pred'] if isinstance(p, dict) else p[0] for p in pairs]
            gts = [p['gt'] if isinstance(p, dict) else p[1] for p in pairs]
            if task == TASK_OPEN:
                try:
                    s[task] = compute_open_metrics(preds, gts)
                except Exception:
                    s[task] = {}
            else:
                s[task] = {'accuracy': compute_mcq_accuracy(preds, gts)}
        scores[name] = s
    return scores


PRETTY = {
    'qwen-base': 'Qwen2.5-VL-base',
    'qwen-ft': 'Ours (Qwen2.5-VL-ft)',
    'gemini': 'Gemini 3.1-flash-lite',
}


# ─── Slide builders ─────────────────────────────────────────────────────────

def slide_title():
    return f"""
    <section class="slide slide-title">
      <h1>MONITRS v2</h1>
      <p class="subtitle">Multi-image Satellite VQA for Natural Disasters</p>
      <p class="small">9,838 events · 50K Sentinel-2 images · 70,911 QA pairs</p>
    </section>
    """


def slide_stats(results):
    n_events = len([v for v in results.values() if 'error' not in v])
    type_counts = Counter(v.get('type', '?') for v in results.values() if 'error' not in v)
    strategy_counts = Counter(v.get('strategy', '?') for v in results.values() if 'error' not in v)

    types_html = ''.join(f'<li>{t}: {n}</li>' for t, n in type_counts.most_common(10))
    strat_html = ''.join(f'<li>{s}: {n}</li>' for s, n in strategy_counts.most_common())

    return f"""
    <section class="slide">
      <h2>Dataset composition</h2>
      <div class="two-col">
        <div>
          <h3>Event types</h3>
          <ul>{types_html}</ul>
        </div>
        <div>
          <h3>Center strategy</h3>
          <ul>{strat_html}</ul>
          <p class="small">FIRMS: NASA fire hotspots · LLM: Gemini-estimated · FEMA: fallback county centroid</p>
        </div>
      </div>
    </section>
    """


def slide_pipeline_diff():
    rows = [
        ("Article search",       "Google Search API",              "DuckDuckGo (ddgs)"),
        ("LLM SDK",              "google.generativeai",             "google-genai / Vertex AI"),
        ("LLM model",            "gemini-pro / 1.5",                "gemini-2.5-flash"),
        ("Fire geolocation",     "LLM guess",                       "NASA FIRMS hotspots"),
        ("Non-fire geolocation", "LLM guess (unbounded)",           "LLM + bbox validation + FEMA fallback"),
        ("Image download",       "Serial per-date, 30s/event",      "Single EE query per event, few seconds"),
        ("Caption grounding",    "Speculative ('satellite would…')", "Factual from article facts only"),
        ("QA input",             "Short event summaries",           "Full dated captions with facts"),
        ("QA prompts",           "Generic questions",               "Ban 'can/would/could'; ground in captions"),
        ("Location QA",          "Random pixel coords (broken)",    "Nominatim + bbox filter (real coords)"),
    ]
    rows_html = ''.join(
        f'<tr><td>{k}</td><td class="old">{v1}</td><td class="new">{v2}</td></tr>'
        for k, v1, v2 in rows)
    return f"""
    <section class="slide">
      <h2>Pipeline changes (v1 → v2)</h2>
      <table class="diff">
        <thead><tr><th>Area</th><th>v1 (2023-2024)</th><th>v2 (2026)</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </section>
    """


def slide_qa_evolution():
    """Show the QA prompt evolution with example."""
    return f"""
    <section class="slide">
      <h2>QA quality iterations</h2>
      <div class="qa-examples">
        <div class="qa-bad">
          <h3>Before (speculative)</h3>
          <p class="q">Q: Can satellite imagery detect physical damage from Hurricane Ida?</p>
          <p class="a">A: High-resolution satellite imagery <b>can detect</b> roof damage, collapsed structures, and debris accumulation...</p>
          <p class="why">Hedge words. Model can't verify from image. Not learnable.</p>
        </div>
        <div class="qa-mid">
          <h3>Interim (image-analysis attempt)</h3>
          <p class="q">Q: What visual evidence indicates the Road 702 Fire reached the Republican River?</p>
          <p class="a">A: A dark burn scar is <b>visibly abutting</b> the Republican River. This visual evidence confirms...</p>
          <p class="why">Better but still Gemini pretending to look at images it can't verify.</p>
        </div>
        <div class="qa-good">
          <h3>Final (facts from captions)</h3>
          <p class="q">Q: How did the burn scar evolve between July 23 and August 2, 2022?</p>
          <p class="a">A: The burn scar became more pronounced and larger, eventually covering approximately 6,735 acres north of FM205 by August 2.</p>
          <p class="why">Facts from article captions. Model learns caption ↔ image mapping.</p>
        </div>
      </div>
    </section>
    """


def slide_sample_event(eid, data, qa_items, caption_map):
    images = get_event_images(eid)
    if not images:
        return ''

    imgs_html = ''
    for img in images[:8]:
        cap = caption_map.get(img['date'], '')[:80]
        imgs_html += f'''
        <div class="img-card">
          <img src="{img_b64(img['path'])}">
          <div class="date">{img['date']}</div>
          <div class="cap">{cap}</div>
        </div>'''

    qa_html = ''
    for i, item in enumerate(qa_items[:3]):
        convos = item.get('conversations', [])
        q = convos[0]['value'][:220] if len(convos) > 0 else ''
        a = convos[1]['value'][:220] if len(convos) > 1 else ''
        q = re.sub(r'<image>|<video>', '', q).strip()
        q = re.sub(r'^This is a sequence of .*?:\s*', '', q).strip()
        task = item.get('task', '?')
        qa_html += f'<div class="qa-row"><span class="tag">{task}</span><div><b>Q:</b> {q}</div><div><b>A:</b> {a}</div></div>'

    return f"""
    <section class="slide">
      <h2>{data.get('event', '?')}</h2>
      <p class="meta">{data.get('type', '')} · {data.get('county', '')}, {data.get('state', '')} · {data.get('start_date', '')} to {data.get('end_date', '')} · center strategy: <b>{data.get('strategy', '?')}</b></p>
      <div class="imgs">{imgs_html}</div>
      <div class="qa">{qa_html}</div>
    </section>
    """


def slide_benchmark_table(scores):
    if not scores:
        return '<section class="slide"><h2>Benchmark</h2><p>No benchmark results yet.</p></section>'

    # Table 2: MCQ classification/grounding
    t2_rows = ''
    for name, s in scores.items():
        pretty = PRETTY.get(name, name)
        ec = s.get('event_type', {}).get('accuracy', 0) * 100
        tg = s.get('temporal_grounding', {}).get('accuracy', 0) * 100
        lg = s.get('location_identification', {}).get('accuracy', 0) * 100
        t2_rows += f'<tr><td>{pretty}</td><td>{ec:.2f}%</td><td>{tg:.2f}%</td><td>{lg:.2f}%</td></tr>'

    # Table 3: Generated VQA
    t3_rows = ''
    for name, s in scores.items():
        pretty = PRETTY.get(name, name)
        mcq = s.get('multiple_choice', {}).get('accuracy', 0) * 100
        op = s.get('custom', {})
        t3_rows += (f'<tr><td>{pretty}</td><td>{mcq:.2f}%</td>'
                    f'<td>{op.get("BLEU-1", 0):.4f}</td>'
                    f'<td>{op.get("BLEU-2", 0):.4f}</td>'
                    f'<td>{op.get("BLEU-3", 0):.4f}</td>'
                    f'<td>{op.get("BLEU-4", 0):.4f}</td>'
                    f'<td>{op.get("METEOR", 0):.4f}</td>'
                    f'<td>{op.get("ROUGE-L", 0):.4f}</td></tr>')

    return f"""
    <section class="slide">
      <h2>Benchmark results</h2>
      <h3>Table 2: Templated MCQ (event classification & grounding)</h3>
      <table class="bench">
        <thead><tr><th>Method</th><th>Event Classification</th><th>Temporal Grounding</th><th>Location Grounding</th></tr></thead>
        <tbody>{t2_rows}</tbody>
      </table>
      <h3>Table 3: Generated VQA</h3>
      <table class="bench">
        <thead><tr><th>Method</th><th>MCQ Acc</th><th>BLEU-1</th><th>BLEU-2</th><th>BLEU-3</th><th>BLEU-4</th><th>METEOR</th><th>ROUGE-L</th></tr></thead>
        <tbody>{t3_rows}</tbody>
      </table>
    </section>
    """


# ─── Main ───────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; background: #f0f0f0; color: #222; }
.slide { background: white; margin: 20px auto; padding: 32px 48px;
         max-width: 1200px; border-radius: 12px;
         box-shadow: 0 2px 10px rgba(0,0,0,0.08); page-break-after: always; }
h1 { font-size: 42px; margin-bottom: 8px; }
h2 { font-size: 28px; margin-bottom: 16px; color: #234; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; }
h3 { font-size: 18px; color: #345; margin-top: 20px; }
.slide-title { text-align: center; padding: 80px 48px; }
.subtitle { font-size: 20px; color: #567; }
.small { color: #789; font-size: 14px; }
.meta { color: #567; font-style: italic; margin-bottom: 16px; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }
th { background: #f5f7fa; }
table.diff td.old { color: #a44; background: #fdf5f5; }
table.diff td.new { color: #275; background: #f5fbf5; }
table.bench td:first-child { font-weight: bold; }
.qa-examples { display: flex; flex-direction: column; gap: 12px; }
.qa-bad { background: #fdf1f1; padding: 12px 16px; border-left: 4px solid #d44; border-radius: 4px; }
.qa-mid { background: #fff8ec; padding: 12px 16px; border-left: 4px solid #d90; border-radius: 4px; }
.qa-good { background: #f0f9ee; padding: 12px 16px; border-left: 4px solid #4a4; border-radius: 4px; }
.q { font-weight: 600; margin: 4px 0; }
.a { margin: 4px 0; }
.why { color: #789; font-size: 12px; font-style: italic; margin-top: 4px; }
.imgs { display: flex; gap: 6px; overflow-x: auto; padding: 8px 0; margin: 8px 0; }
.img-card { min-width: 140px; text-align: center; }
.img-card img { width: 140px; height: 140px; object-fit: cover; border-radius: 6px; border: 1px solid #ddd; }
.date { font-size: 11px; color: #b44; font-weight: bold; margin-top: 4px; }
.cap { font-size: 10px; color: #666; margin-top: 2px; max-width: 140px; line-height: 1.3; }
.qa { background: #f9fafc; border-radius: 6px; padding: 12px; margin-top: 12px; }
.qa-row { padding: 6px 0; border-bottom: 1px solid #eee; font-size: 13px; }
.qa-row:last-child { border-bottom: none; }
.tag { display: inline-block; background: #4488cc; color: white; padding: 2px 8px;
       border-radius: 10px; font-size: 10px; margin-bottom: 4px; }
li { margin: 4px 0; }
ul { padding-left: 22px; }
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='Data/slides.html')
    parser.add_argument('--n-samples', type=int, default=4,
                        help='Number of example events to include')
    parser.add_argument('--sample-events', type=int, nargs='+', default=None,
                        help='Specific event IDs to include (overrides random)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Loading events from {RESULTS_FILE}...")
    results = json.load(open(RESULTS_FILE))
    qa = load_qa()
    scores = load_bench_results()

    # Pick sample events (one per major type)
    if args.sample_events:
        picks = [str(e) for e in args.sample_events]
    else:
        random.seed(args.seed)
        by_type = defaultdict(list)
        for eid, d in results.items():
            if 'error' in d or not get_event_images(eid):
                continue
            by_type[d.get('type', '?')].append(eid)
        picks = []
        for t in ['Fire', 'Hurricane', 'Flood', 'Severe Storm', 'Tornado', 'Severe Ice Storm']:
            if t in by_type and len(picks) < args.n_samples:
                random.shuffle(by_type[t])
                picks.append(by_type[t][0])

    print(f"Building slides with {len(picks)} sample events...")

    slides = [slide_title(),
              slide_pipeline_diff(),
              slide_stats(results),
              slide_qa_evolution()]

    for eid in picks:
        if eid not in results:
            continue
        d = results[eid]
        cap_map = parse_captions(d.get('captions', ''))
        slides.append(slide_sample_event(eid, d, qa.get(eid, []), cap_map))

    slides.append(slide_benchmark_table(scores))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MONITRS v2 Updates</title>
<style>{CSS}</style></head><body>
{''.join(slides)}
</body></html>"""

    with open(args.out, 'w') as f:
        f.write(html)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Slides saved: {args.out} ({size_mb:.1f} MB)")
    print(f"Open in browser or download to view.")


if __name__ == '__main__':
    main()
