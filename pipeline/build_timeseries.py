"""
Step 2: aggregate per-article facts into per-event timeseries.

Reads Data/article_facts.json (per-URL facts).
Writes Data/event_facts.json keyed by event_id:

    {
      "88": {
        "type": "Fire",
        "center": [lat, lon],
        "extent_timeseries": [
          {"date": "2022-07-19", "value": 500, "unit": "acres", "n_articles": 2},
          {"date": "2022-07-22", "value": 4000, "unit": "acres", "n_articles": 3},
          {"date": "2022-07-28", "value": 6735, "unit": "acres", "n_articles": 1}
        ],
        "containment_timeseries": [...],
        "affected_features": [
          {"name": "FM 205", "n_mentions": 3},
          {"name": "Glen Rose ISD", "n_mentions": 1}
        ],
        "notable_dates": {"start": "2022-07-18", "peak": "2022-07-28", "contained": "..."},
        "monotonic_extent": true,
        "n_articles": 5
      }
    }

Usage:
    python pipeline/build_timeseries.py
"""

import argparse
import json
import os
from collections import defaultdict, Counter


FACTS_PATH = 'Data/article_facts.json'
EVENTS_PATH = 'Data/events_processed.json'
OUT_PATH = 'Data/event_facts.json'


# Event types where extent is monotonic non-decreasing during the event
MONOTONIC_TYPES = {'Fire'}
# Event types where extent peaks then recedes (flood, storm surge)
PEAK_TYPES = {'Flood', 'Hurricane', 'Tropical Storm', 'Coastal Storm'}


def _canonical_extent(v, u):
    """Normalize to canonical (value, unit) or None."""
    if v is None or u is None:
        return None
    u = str(u).lower().strip()
    unit_map = {
        'acres': 'acres', 'acre': 'acres',
        'sq_miles': 'sq_miles', 'square miles': 'sq_miles', 'sq mile': 'sq_miles',
        'structures': 'structures', 'homes': 'structures', 'buildings': 'structures',
    }
    if u not in unit_map:
        return None
    return (float(v), unit_map[u])


def _in_window(date_str, start, end, buffer_days=60):
    """True if date_str falls within [start - buffer, end + buffer]."""
    from datetime import datetime, timedelta
    if not date_str:
        return False
    try:
        d = datetime.strptime(date_str, '%Y-%m-%d')
    except Exception:
        return False
    lo = hi = None
    if start:
        try:
            lo = datetime.strptime(start, '%Y-%m-%d') - timedelta(days=buffer_days)
        except Exception:
            pass
    if end:
        try:
            hi = datetime.strptime(end, '%Y-%m-%d') + timedelta(days=buffer_days)
        except Exception:
            pass
    if lo and d < lo:
        return False
    if hi and d > hi:
        return False
    return True


def aggregate_event(event_id, article_facts_for_event, event_meta):
    """Build one event's fact record from its articles."""
    etype = event_meta.get('type', 'Unknown') if event_meta else 'Unknown'
    fema_start = event_meta.get('start_date') if event_meta else None
    fema_end = event_meta.get('end_date') if event_meta else None

    # Collect (date, value, unit) from each article
    extent_by_date = defaultdict(list)  # date -> list of (value, unit, url)
    contain_by_date = defaultdict(list)
    features = Counter()
    notable = defaultdict(list)  # key -> list of dates
    n_rejected = 0

    for f in article_facts_for_event:
        if not isinstance(f, dict) or f.get('error'):
            continue
        # Article-level relevance gate from the extraction step
        if f.get('is_about_target_event') is False:
            n_rejected += 1
            continue

        # extent
        v = _canonical_extent(f.get('extent_number'), f.get('extent_unit'))
        d = f.get('extent_as_of_date') or f.get('pub_date')
        # Reject facts dated far outside the FEMA event window — stale pages,
        # or the LLM grabbed a different incident off a re-used URL.
        if d and not _in_window(d, fema_start, fema_end):
            n_rejected += 1
            continue
        if v and d:
            extent_by_date[d].append((v[0], v[1], f.get('url')))

        # containment
        cp = f.get('contained_pct')
        if cp is not None and d:
            try:
                cp = float(cp)
                if 0 <= cp <= 100:
                    contain_by_date[d].append((cp, f.get('url')))
            except Exception:
                pass

        # affected features
        for feat in f.get('affected_features') or []:
            if isinstance(feat, str) and feat.strip():
                features[feat.strip().lower()] += 1

        # notable dates
        nd = f.get('notable_dates') or {}
        for k, v in nd.items():
            if v:
                notable[k].append(v)

    # Consolidate extent timeseries (median value per date, use dominant unit)
    extent_ts = []
    for date in sorted(extent_by_date):
        vals = extent_by_date[date]
        units = Counter(u for _, u, _ in vals)
        dominant_unit = units.most_common(1)[0][0]
        values = sorted(v for v, u, _ in vals if u == dominant_unit)
        median = values[len(values) // 2] if values else None
        if median is not None:
            extent_ts.append({
                'date': date,
                'value': median,
                'unit': dominant_unit,
                'n_articles': len(vals),
            })

    # Enforce monotonicity for fires — running max
    monotonic = etype in MONOTONIC_TYPES
    if monotonic and extent_ts:
        running_max = 0
        for e in extent_ts:
            if e['value'] > running_max:
                running_max = e['value']
            else:
                e['value'] = running_max

    # Containment timeseries
    contain_ts = []
    for date in sorted(contain_by_date):
        vals = [v for v, _ in contain_by_date[date]]
        contain_ts.append({
            'date': date,
            'value': max(vals),  # containment only goes up
            'n_articles': len(vals),
        })

    # Consolidate notable dates: pick most-frequent, or earliest
    notable_final = {}
    for k in ['start', 'peak', 'contained', 'recovery']:
        candidates = notable.get(k) or []
        if candidates:
            counter = Counter(candidates)
            notable_final[k] = counter.most_common(1)[0][0]
        else:
            notable_final[k] = None

    return {
        'type': etype,
        'center': event_meta.get('center') if event_meta else None,
        'halfwidth': event_meta.get('halfwidth', 0.05) if event_meta else 0.05,
        'fema_start': fema_start,
        'fema_end': fema_end,
        'extent_timeseries': extent_ts,
        'containment_timeseries': contain_ts,
        'affected_features': [
            {'name': name, 'n_mentions': cnt}
            for name, cnt in features.most_common(20)
        ],
        'notable_dates': notable_final,
        'monotonic_extent': monotonic,
        'n_articles': len(article_facts_for_event),
        'n_facts_rejected_out_of_window': n_rejected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--facts', default=FACTS_PATH)
    parser.add_argument('--events', default=EVENTS_PATH)
    parser.add_argument('--out', default=OUT_PATH)
    args = parser.parse_args()

    facts = json.load(open(args.facts))
    events = json.load(open(args.events)) if os.path.exists(args.events) else {}

    # Group facts by event_id
    by_event = defaultdict(list)
    for url, f in facts.items():
        eid = f.get('event_id')
        if eid is not None:
            by_event[str(eid)].append(f)

    out = {}
    for eid, arr in by_event.items():
        meta = events.get(eid, {})
        if 'error' in meta:
            continue
        out[eid] = aggregate_event(eid, arr, meta)

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    # Stats
    n_events = len(out)
    n_with_extent = sum(1 for e in out.values() if e['extent_timeseries'])
    n_with_features = sum(1 for e in out.values() if e['affected_features'])
    print(f"\nBuilt {n_events} event fact records → {args.out}")
    print(f"  With extent timeseries: {n_with_extent} ({100*n_with_extent/max(1,n_events):.0f}%)")
    print(f"  With affected features: {n_with_features} ({100*n_with_features/max(1,n_events):.0f}%)")


if __name__ == '__main__':
    main()
