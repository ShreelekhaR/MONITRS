"""
Post-hoc cleanup on train_total.json / test_total.json:

1. Drop samples where captions/answers are clearly error strings
   ("text is not a caption describing...", empty, etc.)
2. Rebalance option letters in generated multiple_choice (fixes B/C bias)
3. Filter multiple_choice questions that are meta-questions about data quality
   ("Do the dates correspond to hurricane events?")

Usage:
    python MONITRS_QA/fix_qa.py                    # writes *_clean.json
    python MONITRS_QA/fix_qa.py --inplace          # overwrite in place
"""

import json
import os
import re
import argparse
import random
from collections import Counter, defaultdict


BAD_PHRASES = [
    'text is not a caption',
    'is not a caption describing',
    'cannot generate',
    "can't generate",
    'i am unable to',
    'i cannot generate',
    'without correct coordinates',
    'due to a lack of',
    'no valid caption',
    'insufficient information',
    'no data available',
]

BAD_META_PHRASES = [
    'according to the information provided',
    'do the dates provided',
    'correspond to major',
    'is preventing the accurate generation',
    'what is preventing',
    'is there sufficient data',
]


def is_bad_content(text, allow_short=False):
    """Detect failed-caption text.

    allow_short: MCQ answers are a single letter ("b"), so the minimum-length
    heuristic must not apply to them — it was silently dropping every
    multiple_choice item.
    """
    if not text:
        return True
    t = text.strip()
    if not allow_short and len(t) < 5:
        return True
    tl = t.lower()
    return any(p in tl for p in BAD_PHRASES)


def is_meta_question(question):
    tl = question.lower()
    return any(p in tl for p in BAD_META_PHRASES)


def parse_mcq_options(question):
    """Extract A/B/C/D options from MCQ question text.
    Returns (question_stem, [(letter, text), ...]) or None if not parseable.
    Supports 'a. foo', 'A) foo', 'A. foo', etc.
    """
    # Try uppercase then lowercase
    for pattern in [
        r'^(.*?)(A[\.\)]\s*.+?)(?:\n|$)(B[\.\)]\s*.+?)(?:\n|$)(C[\.\)]\s*.+?)(?:\n|$)(D[\.\)]\s*.+?)(?:\n|$)',
        r'^(.*?)(a[\.\)]\s*.+?)(?:\n|$)(b[\.\)]\s*.+?)(?:\n|$)(c[\.\)]\s*.+?)(?:\n|$)(d[\.\)]\s*.+?)(?:\n|$)',
    ]:
        m = re.match(pattern, question, re.DOTALL)
        if m:
            stem = m.group(1).strip()
            opts = []
            for i, letter in enumerate(['a', 'b', 'c', 'd']):
                raw = m.group(i + 2).strip()
                # Strip leading "A." or "A)" etc.
                text = re.sub(r'^[A-Da-d][\.\)]\s*', '', raw).strip()
                opts.append((letter, text))
            return stem, opts
    return None


def shuffle_mcq_letters(item, rng):
    """Randomize option letter positions to break dataset letter bias."""
    convos = item.get('conversations', [])
    if len(convos) < 2:
        return item
    q_full = convos[0]['value']
    gt = convos[1]['value'].strip()

    # Extract answer letter
    m = re.match(r'^([a-dA-D])\b', gt)
    if not m:
        return item
    orig_letter = m.group(1).lower()

    parsed = parse_mcq_options(q_full)
    if not parsed:
        return item

    stem, opts = parsed
    # Find the correct option text
    correct_text = None
    for l, t in opts:
        if l == orig_letter:
            correct_text = t
            break
    if correct_text is None:
        return item

    # Shuffle
    rng.shuffle(opts)
    # Reassign letters
    new_letters = ['a', 'b', 'c', 'd']
    new_opts = [(nl, t) for nl, (_, t) in zip(new_letters, opts)]
    new_correct = next(nl for nl, t in new_opts if t == correct_text)

    # Rebuild question — preserve original letter-case style by detecting
    was_upper = bool(re.search(r'\bA[\.\)]', q_full))
    def fmt(letter, text):
        L = letter.upper() if was_upper else letter
        return f"{L}. {text}"
    opts_text = '\n'.join(fmt(l, t) for l, t in new_opts)
    new_q = f"{stem}\n{opts_text}"

    # Rebuild answer (preserve trailing explanation after the letter, if any)
    rest = gt[1:].lstrip('.).: ').strip()
    new_gt = f"{new_correct}\n{rest}" if rest else new_correct

    item['conversations'] = [
        {**convos[0], 'value': new_q},
        {**convos[1], 'value': new_gt},
    ]
    return item


def clean_split(path, out_path, rng):
    data = json.load(open(path))
    print(f"\n=== {path} ({len(data)} samples) ===")

    kept = []
    stats = Counter()

    for item in data:
        task = item.get('task', '?')
        convos = item.get('conversations', [])
        if len(convos) < 2:
            stats['drop_no_convos'] += 1
            continue

        q = convos[0]['value']
        a = convos[1]['value']

        # 1. Drop degenerate content (failed captions)
        # MCQ gold answers are a single letter — exempt them from the
        # minimum-length check, which otherwise drops every one.
        short_ok = task in ('multiple_choice', 'event_type',
                            'temporal_grounding', 'location_identification')
        if is_bad_content(q) or is_bad_content(a, allow_short=short_ok):
            stats[f'drop_bad_content_{task}'] += 1
            continue

        # 2. Drop meta questions about data quality
        if task == 'multiple_choice' and is_meta_question(q):
            stats[f'drop_meta_{task}'] += 1
            continue

        # 3. Rebalance MCQ letters (except templated ones — those already shuffled)
        if task in ('multiple_choice',):
            item = shuffle_mcq_letters(item, rng)

        kept.append(item)
        stats[f'keep_{task}'] += 1

    with open(out_path, 'w') as f:
        json.dump(kept, f)
    print(f"  Kept: {len(kept)}  Dropped: {len(data) - len(kept)}")
    for k in sorted(stats):
        print(f"    {k}: {stats[k]}")

    # Verify balance after shuffling
    mcq = [x for x in kept if x.get('task') == 'multiple_choice']
    if mcq:
        letter_dist = Counter(x['conversations'][1]['value'].strip()[0].lower() for x in mcq)
        total = sum(letter_dist.values())
        print(f"  multiple_choice letter distribution after fix:")
        for l in 'abcd':
            print(f"    {l}: {letter_dist[l]} ({100*letter_dist[l]/total:.1f}%)")

    return len(kept), len(data) - len(kept)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inplace', action='store_true', help='Overwrite train_total.json/test_total.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    for split in ['train_total.json', 'test_total.json']:
        if not os.path.exists(split):
            print(f"Missing {split}")
            continue
        out = split if args.inplace else split.replace('.json', '_clean.json')
        clean_split(split, out, rng)


if __name__ == '__main__':
    main()
