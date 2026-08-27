"""
Test 2 — blind baseline. Can the question be answered WITHOUT the imagery?

Feeds each test question to an LLM with no images attached. Any question a
blind model answers correctly is testing priors, format artifacts, or answer
distribution — not perception.

Diagnostics per task type:
  blind_acc     accuracy with no image
  chance        1/n_options for MCQ
  excess        blind_acc - chance   (how much leaks through text alone)
  majority_acc  accuracy of always picking the most common gold letter

A task where blind accuracy approaches sighted accuracy is broken and needs
redesign. This is the check that catches letter-distribution bias and
"pick the earliest date"-style shortcuts.

Usage:
    export GCP_PROJECT_ID=ai-sandbox-dev-f139
    python pipeline/test_blind_baseline.py --n-per-task 150
    python pipeline/test_blind_baseline.py --test-file test_total.json
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

TEST_FILE = 'test_total.json'
OUT_PATH = 'Data/blind_baseline.json'
MCQ_TASKS = {'event_type', 'temporal_grounding',
             'location_identification', 'multiple_choice', 'visual_mcq'}


def clean_question(text):
    q = re.sub(r'<image>|<video>', '', text).strip()
    q = re.sub(r'^This is a sequence of .*?:\s*', '', q).strip()
    return q


def extract_letter(text):
    if not isinstance(text, str):
        return None
    m = re.search(r'\b([A-Da-d])\b', text)
    return m.group(1).lower() if m else None


def n_options(question):
    letters = set(re.findall(r'^\s*([a-dA-D])[\.\)]\s', question, re.MULTILINE))
    return max(len(letters), 2) if letters else 4


def call_blind(client, model_id, question):
    from google.genai import types
    try:
        cfg = types.GenerateContentConfig(
            max_output_tokens=24, temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0))
    except Exception:
        cfg = types.GenerateContentConfig(max_output_tokens=24, temperature=0.0)
    prompt = (question +
              "\n\nYou have NOT been shown any images. Based only on the text, "
              "give your single best guess. Answer with one letter.")
    try:
        r = client.models.generate_content(model=model_id, contents=prompt, config=cfg)
        return (r.text or '').strip()
    except Exception as e:
        return f'[ERR: {e}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-file', default=TEST_FILE)
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--n-per-task', type=int, default=150)
    ap.add_argument('--model', default='gemini-2.5-flash-lite')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import random
    data = json.load(open(args.test_file))
    by_task = defaultdict(list)
    for x in data:
        by_task[x.get('task', '?')].append(x)

    random.seed(args.seed)
    sampled = {}
    for t, xs in by_task.items():
        if t not in MCQ_TASKS:
            continue                      # open-ended isn't letter-gradable
        random.shuffle(xs)
        sampled[t] = xs[:args.n_per_task]

    print('Blind baseline — no images shown')
    for t, xs in sampled.items():
        print(f'  {t}: {len(xs)} questions')
    print()

    from google import genai
    from google.genai.types import HttpOptions
    project = os.environ.get('GCP_PROJECT_ID', 'ai-sandbox-dev-f139')
    client = genai.Client(vertexai=True, project=project, location='us-central1',
                          http_options=HttpOptions(api_version='v1'))

    results = {}
    for task, xs in sampled.items():
        preds = [None] * len(xs)

        def work(i):
            q = clean_question(xs[i]['conversations'][0]['value'])
            return i, call_blind(client, args.model, q)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(work, i) for i in range(len(xs))]
            done = 0
            for fut in as_completed(futs):
                i, p = fut.result()
                preds[i] = p
                done += 1
                if done % 50 == 0:
                    print(f'  [{task}] {done}/{len(xs)}', flush=True)

        golds = [x['conversations'][1]['value'].strip() for x in xs]
        gold_letters = [extract_letter(g) for g in golds]
        pred_letters = [extract_letter(p) for p in preds]

        gradable = [(g, p) for g, p in zip(gold_letters, pred_letters) if g]
        correct = sum(1 for g, p in gradable if g == p)
        blind_acc = correct / max(1, len(gradable))

        dist = Counter(g for g in gold_letters if g)
        majority_acc = (dist.most_common(1)[0][1] / max(1, sum(dist.values()))
                        if dist else 0.0)
        avg_opts = sum(n_options(clean_question(x['conversations'][0]['value']))
                       for x in xs) / max(1, len(xs))
        chance = 1.0 / avg_opts

        results[task] = {
            'n': len(gradable),
            'blind_acc': round(blind_acc, 4),
            'chance': round(chance, 4),
            'excess_over_chance': round(blind_acc - chance, 4),
            'majority_class_acc': round(majority_acc, 4),
            'gold_letter_dist': {k: v for k, v in sorted(dist.items())},
            'samples': [
                {'q': clean_question(x['conversations'][0]['value'])[:220],
                 'gold': g, 'blind_pred': p}
                for x, g, p in list(zip(xs, gold_letters, pred_letters))[:5]
            ],
        }

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"task":<26} {"n":>5} {"blind":>7} {"chance":>7} {"excess":>8} {"majority":>9}  flag')
    print('-' * 80)
    for t, r in results.items():
        flag = ''
        if r['excess_over_chance'] > 0.25:
            flag = 'LEAKY — answerable from text'
        elif r['excess_over_chance'] > 0.12:
            flag = 'suspect'
        if r['majority_class_acc'] > 0.4:
            flag += ('; ' if flag else '') + 'skewed answer distribution'
        print(f'{t:<26} {r["n"]:>5} {r["blind_acc"]:>7.3f} {r["chance"]:>7.3f} '
              f'{r["excess_over_chance"]:>8.3f} {r["majority_class_acc"]:>9.3f}  {flag}')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
