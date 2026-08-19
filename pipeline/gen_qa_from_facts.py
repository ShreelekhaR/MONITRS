"""
Step 3: generate visually-verifiable QA from per-event fact timeseries.

Reads Data/event_facts.json + image dates per event (from Data/images/).
Writes Data/qa_v3.json — list of QA entries.

Each entry:
    {
      "event_id": 88,
      "axis": "WHAT|WHERE|WHEN|HOW",
      "task": "extent_bin|feature_locate|onset_range|...",
      "video": [image paths in date order],
      "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
      ]
    }

Usage:
    python pipeline/gen_qa_from_facts.py --limit 20
"""

import argparse
import json
import os
import re
import random
from bisect import bisect_right
from collections import defaultdict


EVENT_FACTS = 'Data/event_facts.json'
IMAGES_DIR = 'Data/images'
OUT_PATH = 'Data/qa_v3.json'


def get_image_dates(eid):
    for suffix in ['', '_firms', '_llm', '_fema']:
        d = os.path.join(IMAGES_DIR, f'{eid}{suffix}')
        if os.path.isdir(d):
            dates = []
            paths = {}
            for fn in sorted(os.listdir(d)):
                if fn.endswith('.png') or fn.endswith('.jpg'):
                    m = re.search(r'(\d{4}-\d{2}-\d{2})', fn)
                    if m:
                        dt = m.group(1)
                        dates.append(dt)
                        paths[dt] = os.path.join(d, fn)
            return sorted(set(dates)), paths
    return [], {}


def bound_extent_at(extent_ts, image_date, monotonic=True):
    """Return the lower bound of extent visible at image_date.

    For monotonic-increasing facts (fire), lower bound = latest reported value
    before or on the image_date.
    """
    dates = [e['date'] for e in extent_ts]
    idx = bisect_right(dates, image_date) - 1
    if idx < 0:
        return None
    return extent_ts[idx]


def acreage_bin(v):
    """Round a numeric extent into a coarse bin label."""
    if v is None:
        return None
    if v < 100:      return 'less than 100 acres'
    if v < 1000:     return 'a few hundred acres'
    if v < 10000:    return 'several thousand acres'
    if v < 100000:   return 'tens of thousands of acres'
    return 'hundreds of thousands of acres'


ACREAGE_BINS = [
    'less than 100 acres',
    'a few hundred acres',
    'several thousand acres',
    'tens of thousands of acres',
    'hundreds of thousands of acres',
]


def _shuffle_mcq(correct_option, distractors, rng):
    """Return (options_dict_letter->text, correct_letter)."""
    all_opts = [correct_option] + [d for d in distractors if d != correct_option][:3]
    while len(all_opts) < 4:
        all_opts.append(f'None of the above ({len(all_opts)})')
    rng.shuffle(all_opts)
    letters = ['a', 'b', 'c', 'd']
    correct_letter = letters[all_opts.index(correct_option)]
    return dict(zip(letters, all_opts)), correct_letter


def _mcq_qa(question, options, correct_letter):
    opt_text = '\n'.join(f'{l}. {t}' for l, t in options.items())
    return {
        'from_human': f'{question}\n{opt_text}\nAnswer with a single letter (a, b, c, or d).',
        'from_gpt': correct_letter,
    }


# ── Question generators ──

def gen_what_type(event, dates, paths, rng):
    """WHAT: event type MCQ with confusable distractors."""
    from MONITRS_QA.templated_mcq import MultipleChoiceQAGenerator
    # Reuse the confusable-distractor map inline
    CONFUSABLE = {
        'Hurricane':       ['Tropical Storm', 'Coastal Storm', 'Flood'],
        'Tropical Storm':  ['Hurricane', 'Coastal Storm', 'Flood'],
        'Coastal Storm':   ['Hurricane', 'Tropical Storm', 'Flood'],
        'Flood':           ['Hurricane', 'Tropical Storm', 'Severe Storm'],
        'Severe Storm':    ['Tornado', 'Coastal Storm', 'Hurricane'],
        'Tornado':         ['Severe Storm', 'Hurricane', 'Coastal Storm'],
        'Fire':            ['Landslide', 'Volcanic Eruption', 'Flood'],
    }
    etype = event['type']
    distractors = CONFUSABLE.get(etype, [])
    if not distractors:
        return None
    options, letter = _shuffle_mcq(etype, distractors, rng)
    qa = _mcq_qa('What type of natural disaster is shown in these satellite images?', options, letter)
    return {'axis': 'WHAT', 'task': 'event_type', **qa}


def gen_how_extent(event, dates, paths, rng):
    """HOW: extent bin from last article extent before latest image."""
    if not event['extent_timeseries'] or not dates:
        return None
    last_date = dates[-1]
    b = bound_extent_at(event['extent_timeseries'], last_date, event['monotonic_extent'])
    if not b or b['unit'] != 'acres':
        return None
    correct = acreage_bin(b['value'])
    if correct is None:
        return None
    distractors = [x for x in ACREAGE_BINS if x != correct]
    rng.shuffle(distractors)
    options, letter = _shuffle_mcq(correct, distractors[:3], rng)
    q = f'By {last_date}, roughly how much land had the {event["type"].lower()} affected?'
    qa = _mcq_qa(q, options, letter)
    return {'axis': 'HOW', 'task': 'extent_bin', **qa}


def gen_when_onset(event, dates, paths, rng):
    """WHEN: which image-date interval contains the reported start date."""
    start = event['notable_dates'].get('start')
    if not start or len(dates) < 3:
        return None
    # Find interval containing start
    correct_range = None
    for i in range(len(dates) - 1):
        if dates[i] <= start < dates[i + 1]:
            correct_range = f'Between {dates[i]} and {dates[i + 1]}'
            break
    if not correct_range:
        if start < dates[0]:
            correct_range = f'Before {dates[0]}'
        else:
            return None

    all_ranges = [f'Before {dates[0]}']
    for i in range(len(dates) - 1):
        all_ranges.append(f'Between {dates[i]} and {dates[i + 1]}')
    all_ranges.append(f'After {dates[-1]}')
    distractors = [r for r in all_ranges if r != correct_range]
    rng.shuffle(distractors)
    options, letter = _shuffle_mcq(correct_range, distractors[:3], rng)
    q = f'During which interval between the satellite images does the {event["type"].lower()} first become visible?'
    qa = _mcq_qa(q, options, letter)
    return {'axis': 'WHEN', 'task': 'onset_range', **qa}


def gen_where_feature(event, dates, paths, rng):
    """WHERE: locate an article-mentioned affected feature.

    For now, this returns None if we don't have per-feature pixel coords.
    Placeholder — needs concept POI lookup pipeline (Nominatim viewbox).
    """
    return None  # TODO milestone 3


def build_qa_for_event(eid, event, rng):
    dates, paths = get_image_dates(eid)
    if not dates:
        return []
    image_paths = [paths[d] for d in dates]

    entries = []
    for gen in [gen_what_type, gen_when_onset, gen_how_extent, gen_where_feature]:
        item = gen(event, dates, paths, rng)
        if not item:
            continue
        entries.append({
            'event_id': int(eid),
            'video': image_paths,
            'timestamp': dates,
            'axis': item['axis'],
            'task': item['task'],
            'conversations': [
                {'from': 'human', 'value': item['from_human']},
                {'from': 'gpt',   'value': item['from_gpt']},
            ],
        })
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--facts', default=EVENT_FACTS)
    parser.add_argument('--out', default=OUT_PATH)
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process first N events with facts')
    parser.add_argument('--event', nargs='+', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    facts = json.load(open(args.facts))
    rng = random.Random(args.seed)

    if args.event:
        eids = [str(e) for e in args.event if str(e) in facts]
    else:
        eids = sorted(facts.keys(), key=lambda x: int(x))
        if args.limit:
            eids = eids[:args.limit]

    all_qa = []
    per_axis = defaultdict(int)
    for eid in eids:
        entries = build_qa_for_event(eid, facts[eid], rng)
        for e in entries:
            per_axis[e['axis']] += 1
        all_qa.extend(entries)

    with open(args.out, 'w') as f:
        json.dump(all_qa, f, indent=2)

    print(f"\nGenerated {len(all_qa)} QA entries → {args.out}")
    for a in sorted(per_axis):
        print(f"  {a}: {per_axis[a]}")


if __name__ == '__main__':
    main()
