"""
Combine benchmark result files and print comparison tables.

Usage:
    python Train/print_tables.py benchmark_full.json benchmark_gemini.json
    python Train/print_tables.py --ckpts                # use benchmark_ckpts/*.json
"""

import json
import os
import re
import argparse
import sys
from glob import glob

sys.path.insert(0, 'Train')
from benchmark import compute_mcq_accuracy, compute_open_metrics, TASK_OPEN


PRETTY_NAMES = {
    'qwen-base': 'Qwen2.5-VL-base',
    'qwen-ft': 'Ours (Qwen2.5-VL-ft)',
    'gemini': 'Gemini 3.1-flash-lite',
    'videollava': 'VideoLLaVA',
    'teochat': 'TEOChat',
}


def score_from_pairs(task_pairs):
    scored = {}
    for task, pairs in task_pairs.items():
        preds = [p['pred'] if isinstance(p, dict) else p[0] for p in pairs]
        gts = [p['gt'] if isinstance(p, dict) else p[1] for p in pairs]
        if task == TASK_OPEN:
            scored[task] = compute_open_metrics(preds, gts)
        else:
            scored[task] = {'accuracy': compute_mcq_accuracy(preds, gts)}
    return scored


def load_scores_from_result_file(path):
    """Load scores dict from a benchmark output JSON."""
    with open(path) as f:
        data = json.load(f)
    return data.get('scores', {})


def load_scores_from_ckpt_file(path):
    """Load raw predictions from checkpoint and score them."""
    with open(path) as f:
        raw = json.load(f)
    return score_from_pairs(raw)


def print_tables(all_scores):
    """Print Table 2 (MCQ classification/grounding) and Table 3 (Generated VQA)."""
    def fmt_pct(x):
        return f"{x*100:>7.2f}%"

    print(f"\n{'='*80}")
    print("Table 2: Multiple Choice Event Classification & Grounding")
    print(f"{'='*80}")
    header = f"{'Method':<28} {'Event Class':>13} {'Temporal':>13} {'Location':>13}"
    print(header)
    print('-' * len(header))
    for name, scores in all_scores.items():
        ec = scores.get('event_type', {}).get('accuracy', 0)
        tg = scores.get('temporal_grounding', {}).get('accuracy', 0)
        lg = scores.get('location_identification', {}).get('accuracy', 0)
        print(f"{name:<28} {fmt_pct(ec):>13} {fmt_pct(tg):>13} {fmt_pct(lg):>13}")

    print(f"\n{'='*100}")
    print("Table 3: Generated VQA")
    print(f"{'='*100}")
    header = f"{'Method':<28} {'MCQ Acc':>9} {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-3':>8} {'BLEU-4':>8} {'METEOR':>8} {'ROUGE-L':>8}"
    print(header)
    print('-' * len(header))
    for name, scores in all_scores.items():
        mcq = scores.get('multiple_choice', {}).get('accuracy', 0)
        op = scores.get(TASK_OPEN, {})
        print(f"{name:<28} {fmt_pct(mcq):>9} "
              f"{op.get('BLEU-1', 0):>8.4f} {op.get('BLEU-2', 0):>8.4f} "
              f"{op.get('BLEU-3', 0):>8.4f} {op.get('BLEU-4', 0):>8.4f} "
              f"{op.get('METEOR', 0):>8.4f} {op.get('ROUGE-L', 0):>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*', help='benchmark_*.json files to combine')
    parser.add_argument('--ckpts', action='store_true',
                        help='Score benchmark_ckpts/*.json directly instead of using --files')
    args = parser.parse_args()

    all_scores = {}

    if args.ckpts:
        for path in sorted(glob('benchmark_ckpts/*.json')):
            name = os.path.basename(path).replace('.json', '')
            pretty = PRETTY_NAMES.get(name, name)
            print(f"Scoring {pretty} from {path}...")
            all_scores[pretty] = load_scores_from_ckpt_file(path)
    else:
        if not args.files:
            print("Provide result files or use --ckpts")
            sys.exit(1)
        for path in args.files:
            with open(path) as f:
                data = json.load(f)
            scores = data.get('scores', {})
            for name, s in scores.items():
                all_scores[name] = s

    print_tables(all_scores)


if __name__ == '__main__':
    main()
