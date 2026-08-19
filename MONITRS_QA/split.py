"""
County-level holdout train/test split, stratified by event type.

For each event type, sample ~20% of unique (state, county) tuples for test.
No county appears in both splits. All 3 QA generators share this assignment
so train/test are consistent across question types.

Cached to Data/split_assignment.json so re-runs produce the same split.

Usage:
    from split import get_train_test_ids
    train_ids, test_ids = get_train_test_ids()
"""

import json
import os
import random
from collections import defaultdict


ASSIGN_PATH = 'Data/split_assignment.json'
RESULTS_FILE = 'Data/events_processed.json'


def _county_key(v):
    return f"{(v.get('state') or 'UNK').strip()}::{(v.get('county') or 'UNK').strip()}"


def compute_split(results_file=RESULTS_FILE, test_frac=0.2, seed=42):
    """Return (train_ids, test_ids) with county-level holdout, type-stratified."""
    results = json.load(open(results_file))

    # Group counties by event_type
    counties_by_type = defaultdict(set)
    events_by_county = defaultdict(list)
    for eid, v in results.items():
        if 'error' in v:
            continue
        etype = v.get('type', 'Unknown')
        ckey = _county_key(v)
        counties_by_type[etype].add(ckey)
        events_by_county[ckey].append(eid)

    rng = random.Random(seed)
    test_counties = set()
    for etype, counties in counties_by_type.items():
        counties = sorted(counties)
        rng.shuffle(counties)
        n_test = max(1, int(round(len(counties) * test_frac)))
        for c in counties[:n_test]:
            test_counties.add(c)

    train_ids, test_ids = set(), set()
    for ckey, eids in events_by_county.items():
        if ckey in test_counties:
            test_ids.update(eids)
        else:
            train_ids.update(eids)

    return train_ids, test_ids, test_counties


def get_train_test_ids(results_file=RESULTS_FILE):
    """Load cached assignment or compute + cache. Returns (train_ids: set, test_ids: set)."""
    if os.path.exists(ASSIGN_PATH):
        d = json.load(open(ASSIGN_PATH))
        return set(d['train_ids']), set(d['test_ids'])

    train_ids, test_ids, test_counties = compute_split(results_file)
    os.makedirs(os.path.dirname(ASSIGN_PATH), exist_ok=True)
    with open(ASSIGN_PATH, 'w') as f:
        json.dump({
            'train_ids': sorted(train_ids, key=int),
            'test_ids': sorted(test_ids, key=int),
            'test_counties': sorted(test_counties),
            'seed': 42,
            'test_frac': 0.2,
            'method': 'county-holdout-stratified-by-type',
        }, f, indent=2)
    return train_ids, test_ids


def print_split_stats():
    """Diagnostic: show per-type split breakdown."""
    results = json.load(open(RESULTS_FILE))
    train_ids, test_ids = get_train_test_ids()

    print(f"Total: {len(train_ids)} train, {len(test_ids)} test")
    print(f"       ({len(test_ids)*100/(len(train_ids)+len(test_ids)):.1f}% test)")

    per_type = defaultdict(lambda: {'train': 0, 'test': 0})
    counties_per_type = defaultdict(lambda: {'train': set(), 'test': set()})
    for eid, v in results.items():
        if 'error' in v:
            continue
        etype = v.get('type', 'Unknown')
        ckey = _county_key(v)
        split = 'train' if eid in train_ids else ('test' if eid in test_ids else None)
        if split:
            per_type[etype][split] += 1
            counties_per_type[etype][split].add(ckey)

    print(f"\n{'Event type':<25} {'train events':>13} {'test events':>13} {'train counties':>16} {'test counties':>15}")
    print('-' * 87)
    for etype in sorted(per_type):
        tr, te = per_type[etype]['train'], per_type[etype]['test']
        tc, tec = len(counties_per_type[etype]['train']), len(counties_per_type[etype]['test'])
        print(f"{etype:<25} {tr:>13} {te:>13} {tc:>16} {tec:>15}")

    # Sanity: county-disjointness
    all_train_counties = set().union(*counties_per_type[t]['train'] for t in counties_per_type)
    all_test_counties  = set().union(*counties_per_type[t]['test']  for t in counties_per_type)
    overlap = all_train_counties & all_test_counties
    print(f"\nCounty overlap between train and test: {len(overlap)} (should be 0)")


if __name__ == '__main__':
    if os.path.exists(ASSIGN_PATH):
        print(f"Assignment exists at {ASSIGN_PATH}. Delete to regenerate.")
    print_split_stats()
