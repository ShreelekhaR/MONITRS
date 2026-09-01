"""
Per-event agentic harvest loop.

For ONE event, iterate until we have enough verified facts or run out of query
strategies:

    1. build queries (named storm > county+state+date > authoritative domains)
    2. search    -> candidate urls
    3. scrape    -> page text + publication date
    4. verify    -> LLM decides "is this article about THIS event?" + extracts facts
    5. score coverage: extent? dates? affected features? enough relevant articles?
    6. if gaps remain and query budget left -> new query angle, goto 2

Writes per-event records to Data/harvest/<event_id>.json so runs are resumable
and inspectable one event at a time.

Usage:
    export GCP_PROJECT_ID=ai-sandbox-dev-f139
    python pipeline/harvest_event.py --event 5341
    python pipeline/harvest_event.py --event 5341 453 1346 --workers 3
    python pipeline/harvest_event.py --limit 20
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from search_articles import (build_queries, search_ddg, domain_of,
                             SKIP_DOMAINS, month_year, year_of,
                             consecutive_empty_searches)
from scrape_articles import scrape_one
from extract_facts_from_scrape import PROMPT, call
from validate_features import validate_features


EVENTS_PATH = 'Data/events_processed.json'
HARVEST_DIR = 'Data/harvest'


# ── coverage scoring ────────────────────────────────────────────────────────

def coverage(facts):
    """What fraction of the target fact set do we have?

    Extent must be LOCAL — statewide aggregates are the wrong order of
    magnitude for a single-county chip. But dates and features are credited
    from regional articles too: a hurricane's landfall date is the same across
    every county it hit, and features are spatially filtered downstream.
    """
    rel = [f for f in facts if f.get('is_about_target_event')]
    local = [f for f in rel if f.get('extent_scope') in (None, 'local')]
    has_extent = any(f.get('extent_number') for f in local)
    has_start = any((f.get('notable_dates') or {}).get('start') for f in rel)
    has_feats = any(f.get('validated_features') for f in rel)
    multi_date = len({f.get('extent_as_of_date') or f.get('pub_date')
                      for f in local if f.get('extent_number')}) >= 2
    n_county = sum(1 for f in rel if f.get('geographic_scope') == 'county')
    return {
        'n_relevant': len(rel),
        'n_county_scoped': n_county,
        'n_local_extent': sum(1 for f in local if f.get('extent_number')),
        'has_extent': has_extent,
        'has_start_date': has_start,
        'has_features': has_feats,
        'has_extent_timeseries': multi_date,
        'score': sum([has_extent, has_start, has_feats, multi_date, len(rel) >= 3]) / 5.0,
    }


def gap_queries(event, cov, already_used):
    """Extra query angles targeting whatever's still missing."""
    from search_articles import place_phrase, clean_county, STATE_NAMES
    etype = (event.get('type') or '').lower()
    state_abbr = (event.get('state') or '').strip()
    state = STATE_NAMES.get(state_abbr.upper(), state_abbr)
    county_bare, _ = clean_county(event.get('county'))
    place = place_phrase(event.get('county'), state_abbr)
    title = (event.get('event') or '').title()
    yr = year_of(event.get('start_date') or '')
    my = month_year(event.get('start_date') or '')

    out = []
    if not cov['has_extent']:
        if etype == 'fire':
            out += [f'{place} wildfire acres burned {yr}',
                    f'{county_bare} {state} fire perimeter acres {yr}']
        elif etype in ('flood', 'severe storm', 'hurricane', 'tropical storm',
                       'coastal storm'):
            out += [f'{place} flooding homes damaged {yr}',
                    f'{place} {etype} damage assessment {yr}']
        elif etype in ('snowstorm', 'severe ice storm', 'winter storm'):
            out += [f'{place} snowfall inches {my}',
                    f'{place} winter storm power outages {yr}']
        else:
            out += [f'{place} {etype} damage assessment {yr}']
    if not cov['has_extent_timeseries']:
        out += [f'{place} {etype} update {my}',
                f'{county_bare} {state} storm report {my}']
    if not cov['has_features']:
        out += [f'{place} {etype} highway closed {yr}',
                f'{place} {etype} river bridge damage {yr}']
    if cov['n_relevant'] < 3:
        out += [f'site:weather.gov {county_bare} {state} {yr}',
                f'{place} FEMA disaster declaration {yr}',
                f'{title} {state} news {yr}']

    return [q for q in out if q and q not in already_used]


# ── single-article verify+extract ───────────────────────────────────────────

def verify_and_extract(url, page, event_meta, client, model_id):
    content = page.get('content', '')
    if len(content) < 250:
        return {'url': url, 'error': 'too_short'}
    p = (PROMPT
         .replace('{event_name}', str(event_meta.get('event', 'unknown')))
         .replace('{event_type}', str(event_meta.get('type', 'unknown')))
         .replace('{county}', str(event_meta.get('county', 'unknown')))
         .replace('{state}', str(event_meta.get('state', 'unknown')))
         .replace('{fema_start}', str(event_meta.get('start_date', 'unknown')))
         .replace('{fema_end}', str(event_meta.get('end_date', 'unknown')))
         .replace('{pub_date}', page.get('pub_date') or 'unknown')
         .replace('{title}', page.get('title', ''))
         .replace('{content}', content[:18000]))
    resp = call(client, p, model_id)
    if resp.startswith('[ERR'):
        return {'url': url, 'error': resp[:160]}
    try:
        f = json.loads(resp)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if not m:
            return {'url': url, 'error': 'no_json'}
        try:
            f = json.loads(m.group(0))
        except Exception:
            return {'url': url, 'error': 'bad_json'}
    f['url'] = url
    f['domain'] = page.get('domain')
    f['pub_date'] = page.get('pub_date')
    f['title'] = page.get('title')

    # Drop non-local extent figures — statewide/regional totals are the wrong
    # order of magnitude for a single-county image chip.
    scope = f.get('extent_scope')
    if scope and scope != 'local' and f.get('extent_number'):
        f['rejected_extent'] = {
            'value': f.get('extent_number'),
            'unit': f.get('extent_unit'),
            'scope': scope,
        }
        f['extent_number'] = None
        f['extent_unit'] = None

    # Spatially validate named features against the event bbox
    names = f.get('affected_features') or []
    if names and f.get('is_about_target_event'):
        kept, dropped = validate_features(
            names,
            event_meta.get('center'),
            event_meta.get('halfwidth', 0.05),
            state=event_meta.get('state', ''),
            county=(event_meta.get('county') or '').split('(')[0].strip(),
        )
        f['validated_features'] = kept
        f['dropped_features'] = dropped
    else:
        f['validated_features'] = []
        f['dropped_features'] = []

    return f


# ── the loop ────────────────────────────────────────────────────────────────

def harvest(eid, event, client, model_id='gemini-2.5-flash-lite',
            max_rounds=3, per_query=8, target_score=0.8, verbose=True,
            scrape_workers=6, extract_workers=4):
    used_queries, seen_urls = set(), set()
    facts, pages = [], {}

    def log(msg):
        if verbose:
            print(f'  [ev{eid}] {msg}', flush=True)

    log(f'{event.get("type")} | {event.get("county")} County, {event.get("state")} '
        f'| {event.get("start_date")} .. {event.get("end_date")}')

    queries = build_queries(event)
    for rnd in range(1, max_rounds + 1):
        queries = [q for q in queries if q not in used_queries][:5]
        if not queries:
            log('no new queries left')
            break
        log(f'round {rnd}: {len(queries)} queries')

        # 1) search
        candidates = []
        for q in queries:
            used_queries.add(q)
            for u in search_ddg(q, per_query):
                if u in seen_urls or domain_of(u) in SKIP_DOMAINS:
                    continue
                seen_urls.add(u)
                candidates.append(u)
            time.sleep(0.5)
        log(f'  {len(candidates)} new candidate urls')
        if not candidates:
            cov = coverage(facts)
            queries = gap_queries(event, cov, used_queries)
            continue

        # 2) scrape
        with ThreadPoolExecutor(max_workers=scrape_workers) as pool:
            futs = [pool.submit(scrape_one, eid, u) for u in candidates]
            for fut in as_completed(futs):
                u, page = fut.result()
                if page.get('content'):
                    pages[u] = page
        fresh = [u for u in candidates if u in pages]
        log(f'  {len(fresh)} scraped with content')

        # 3) verify + extract
        with ThreadPoolExecutor(max_workers=extract_workers) as pool:
            futs = [pool.submit(verify_and_extract, u, pages[u], event, client, model_id)
                    for u in fresh]
            for fut in as_completed(futs):
                facts.append(fut.result())

        cov = coverage(facts)
        n_scope_rej = sum(1 for f in facts if f.get('rejected_extent'))
        n_geo_rej = sum(len(f.get('dropped_features') or []) for f in facts)
        log(f'  coverage {cov["score"]:.2f} '
            f'(relevant={cov["n_relevant"]} extent={cov["has_extent"]} '
            f'start={cov["has_start_date"]} feats={cov["has_features"]} '
            f'ts={cov["has_extent_timeseries"]}) '
            f'| filtered: {n_scope_rej} non-local extents, {n_geo_rej} off-bbox features')

        if cov['score'] >= target_score:
            log('target coverage reached')
            break
        queries = gap_queries(event, cov, used_queries)

    cov = coverage(facts)
    return {
        'event_id': int(eid),
        'event': event.get('event'),
        'type': event.get('type'),
        'state': event.get('state'),
        'county': event.get('county'),
        'fema_start': event.get('start_date'),
        'fema_end': event.get('end_date'),
        'center': event.get('center'),
        'halfwidth': event.get('halfwidth', 0.05),
        'queries_used': sorted(used_queries),
        'n_urls_seen': len(seen_urls),
        'n_scraped': len(pages),
        'facts': facts,
        'coverage': cov,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--out-dir', default=HARVEST_DIR)
    ap.add_argument('--event', nargs='+', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--stratified', action='store_true',
                    help='With --limit, sample proportionally across event '
                         'types instead of taking the first N ids. The FEMA '
                         'file is type-ordered, so the first 200 ids are '
                         '199 fires — the easiest case and not representative.')
    ap.add_argument('--model', default='gemini-2.5-flash-lite')
    ap.add_argument('--max-rounds', type=int, default=3)
    ap.add_argument('--per-query', type=int, default=8)
    ap.add_argument('--target-score', type=float, default=0.8)
    ap.add_argument('--workers', type=int, default=1,
                    help='Parallel events (each already parallelizes internally)')
    ap.add_argument('--force', action='store_true', help='Re-harvest existing')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    events = json.load(open(args.events))

    targets = []
    for eid, v in events.items():
        if 'error' in v:
            continue
        if args.event and int(eid) not in args.event:
            continue
        if not args.force and os.path.exists(os.path.join(args.out_dir, f'{eid}.json')):
            continue
        targets.append((eid, v))
    targets.sort(key=lambda x: int(x[0]))
    if args.limit:
        if args.stratified:
            from collections import defaultdict
            import random as _r
            by_type = defaultdict(list)
            for eid, v in targets:
                by_type[v.get('type', 'Unknown')].append((eid, v))
            rng = _r.Random(42)
            total = sum(len(v) for v in by_type.values())
            picked = []
            for t, xs in by_type.items():
                rng.shuffle(xs)
                share = max(1, round(args.limit * len(xs) / total))
                picked.extend(xs[:share])
            rng.shuffle(picked)
            targets = picked[:args.limit]
            shown = defaultdict(int)
            for _, v in targets:
                shown[v.get('type', 'Unknown')] += 1
            print('Stratified sample:')
            for t, c in sorted(shown.items(), key=lambda x: -x[1]):
                print(f'    {t}: {c}')
            print()
        else:
            targets = targets[:args.limit]

    if not targets:
        print('Nothing to harvest'); return
    print(f'Harvesting {len(targets)} events\n')

    from google import genai
    from google.genai.types import HttpOptions
    project = os.environ.get('GCP_PROJECT_ID', 'ai-sandbox-dev-f139')
    client = genai.Client(vertexai=True, project=project, location='us-central1',
                          http_options=HttpOptions(api_version='v1'))

    def run(eid, ev):
        try:
            rec = harvest(eid, ev, client, args.model, args.max_rounds,
                          args.per_query, args.target_score)
        except Exception as e:
            # One bad event must not discard hours of completed work
            print(f'  [ev{eid}] FAILED: {type(e).__name__}: {str(e)[:120]}',
                  flush=True)
            return None
        # Don't persist a zero-article result: it is far more likely that
        # search was throttled than that the event has no coverage, and a
        # cached zero would be skipped on resume and never retried.
        if rec['coverage']['n_relevant'] == 0 and rec['n_scraped'] == 0:
            return rec
        with open(os.path.join(args.out_dir, f'{eid}.json'), 'w') as f:
            json.dump(rec, f, indent=2)
        return rec

    results = []
    n_zero_streak = 0
    ABORT_AFTER_ZERO = 25

    def record(r):
        nonlocal n_zero_streak
        if not r:
            return
        results.append(r)
        if r['coverage']['n_relevant'] == 0:
            n_zero_streak += 1
        else:
            n_zero_streak = 0

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run, eid, ev): eid for eid, ev in targets}
            for fut in as_completed(futs):
                record(fut.result())
                if n_zero_streak >= ABORT_AFTER_ZERO:
                    print(f'\nABORTING: {n_zero_streak} consecutive events with '
                          f'zero relevant articles '
                          f'({consecutive_empty_searches()} empty searches in a row).\n'
                          f'Search is almost certainly rate-limited. Wait ~30 min, '
                          f'then retry with fewer workers:\n'
                          f'    DDG_MIN_INTERVAL=5 python pipeline/harvest_event.py '
                          f'--limit {args.limit or 200} --stratified --workers 2\n'
                          f'Completed events are saved and will be skipped on resume.\n',
                          flush=True)
                    for f in futs:
                        f.cancel()
                    break
    else:
        for eid, ev in targets:
            record(run(eid, ev))
            if n_zero_streak >= ABORT_AFTER_ZERO:
                print(f'\nABORTING: {n_zero_streak} consecutive zero-article events; '
                      f'search is rate-limited. Wait and resume.\n', flush=True)
                break

    if not results:
        print('\nNo events harvested successfully.')
        return

    print(f'\n{"="*78}')
    print(f'{"event":>7} {"type":<18} {"rel":>4} {"score":>6}  extent / start / feats / ts')
    print('-' * 78)
    for r in sorted(results, key=lambda x: x['event_id']):
        c = r['coverage']
        flags = ''.join('Y' if c[k] else '.' for k in
                        ['has_extent', 'has_start_date', 'has_features', 'has_extent_timeseries'])
        print(f'{r["event_id"]:>7} {str(r["type"])[:18]:<18} {c["n_relevant"]:>4} '
              f'{c["score"]:>6.2f}  {flags}')
    avg = sum(r['coverage']['score'] for r in results) / max(1, len(results))
    print(f'\nmean coverage {avg:.2f} over {len(results)} events')


if __name__ == '__main__':
    main()
