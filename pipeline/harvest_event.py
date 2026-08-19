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
                             SKIP_DOMAINS, month_year, year_of)
from scrape_articles import scrape_one
from extract_facts_from_scrape import PROMPT, call


EVENTS_PATH = 'Data/events_processed.json'
HARVEST_DIR = 'Data/harvest'


# ── coverage scoring ────────────────────────────────────────────────────────

def coverage(facts):
    """What fraction of the target fact set do we have?"""
    rel = [f for f in facts if f.get('is_about_target_event')]
    has_extent = any(f.get('extent_number') for f in rel)
    has_start = any((f.get('notable_dates') or {}).get('start') for f in rel)
    has_feats = any(f.get('affected_features') for f in rel)
    multi_date = len({f.get('extent_as_of_date') or f.get('pub_date')
                      for f in rel if f.get('extent_number')}) >= 2
    return {
        'n_relevant': len(rel),
        'has_extent': has_extent,
        'has_start_date': has_start,
        'has_features': has_feats,
        'has_extent_timeseries': multi_date,
        'score': sum([has_extent, has_start, has_feats, multi_date, len(rel) >= 3]) / 5.0,
    }


def gap_queries(event, cov, already_used):
    """Extra query angles targeting whatever's still missing."""
    etype = (event.get('type') or '').lower()
    state = (event.get('state') or '').strip()
    county = (event.get('county') or '').strip()
    title = (event.get('event') or '').title()
    yr = year_of(event.get('start_date') or '')
    my = month_year(event.get('start_date') or '')
    place = f'"{county} County" {state}' if county else state

    out = []
    if not cov['has_extent']:
        if etype == 'fire':
            out += [f'{place} fire acres burned {yr}',
                    f'{title} {state} acres {yr}']
        elif etype in ('flood', 'severe storm', 'hurricane', 'tropical storm'):
            out += [f'{place} flooding homes damaged {yr}',
                    f'{place} {etype} damage assessment {yr}']
        else:
            out += [f'{place} {etype} damage assessment {yr}']
    if not cov['has_extent_timeseries']:
        out += [f'{place} {etype} update {my}',
                f'{place} {etype} latest {my}']
    if not cov['has_features']:
        out += [f'{place} {etype} road closed {yr}',
                f'{place} {etype} river bridge damage {yr}']
    if cov['n_relevant'] < 3:
        out += [f'site:weather.gov {county} {state} {yr}',
                f'{place} disaster declaration {yr}']

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
        log(f'  coverage {cov["score"]:.2f} '
            f'(relevant={cov["n_relevant"]} extent={cov["has_extent"]} '
            f'start={cov["has_start_date"]} feats={cov["has_features"]} '
            f'ts={cov["has_extent_timeseries"]})')

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
        rec = harvest(eid, ev, client, args.model, args.max_rounds,
                      args.per_query, args.target_score)
        with open(os.path.join(args.out_dir, f'{eid}.json'), 'w') as f:
            json.dump(rec, f, indent=2)
        return rec

    results = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run, eid, ev): eid for eid, ev in targets}
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for eid, ev in targets:
            results.append(run(eid, ev))

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
