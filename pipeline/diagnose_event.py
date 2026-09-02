"""
Diagnose one event's harvest, stage by stage.

Prints what happens at each step so a zero-coverage result can be attributed to
a specific cause rather than guessed at:

    queries      -> what we searched for
    urls         -> what search returned
    scrape       -> whether pages have real article text or boilerplate
    verify       -> accept/reject per article, with the model's reason
    facts        -> what got extracted from the accepted ones

Usage:
    export GCP_PROJECT_ID=ai-sandbox-dev-f139
    python pipeline/diagnose_event.py --event 6971
    python pipeline/diagnose_event.py --event 6971 --full-text
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search_articles import build_queries, search_ddg, domain_of, SKIP_DOMAINS
from scrape_articles import scrape_one
from harvest_event import verify_and_extract, gap_queries, coverage

EVENTS_PATH = 'Data/events_processed.json'


def rule(title=''):
    print('\n' + '=' * 78)
    if title:
        print(title)
        print('=' * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', type=int, required=True)
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--model', default='gemini-2.5-flash-lite')
    ap.add_argument('--per-query', type=int, default=8)
    ap.add_argument('--max-queries', type=int, default=5)
    ap.add_argument('--full-text', action='store_true',
                    help='Print more of each scraped page')
    args = ap.parse_args()

    events = json.load(open(args.events))
    eid = str(args.event)
    if eid not in events:
        print(f'event {eid} not found'); return
    ev = events[eid]

    rule(f'EVENT {eid}')
    print(f'  name:   {ev.get("event")}')
    print(f'  type:   {ev.get("type")}')
    print(f'  place:  {ev.get("county")} County, {ev.get("state")}')
    print(f'  window: {ev.get("start_date")} .. {ev.get("end_date")}')
    print(f'  center: {ev.get("center")}  halfwidth: {ev.get("halfwidth", 0.05)}')

    # 1. queries
    queries = build_queries(ev)[:args.max_queries]
    rule('1. QUERIES')
    for q in queries:
        print(f'  {q}')

    # 2. search
    rule('2. SEARCH RESULTS')
    seen, candidates = set(), []
    for q in queries:
        urls = search_ddg(q, args.per_query)
        kept = [u for u in urls
                if u not in seen and domain_of(u) not in SKIP_DOMAINS]
        for u in kept:
            seen.add(u)
        candidates.extend(kept)
        print(f'  [{len(urls):>2} raw, {len(kept):>2} new] {q[:66]}')
    print(f'\n  total unique candidates: {len(candidates)}')
    if not candidates:
        print('\n  >> DIAGNOSIS: search returned nothing. Rate limiting or '
              'queries too narrow.')
        return

    # 3. scrape
    rule('3. SCRAPE')
    pages = {}
    for u in candidates:
        _, page = scrape_one(eid, u)
        status = ('ok' if page.get('content')
                  else page.get('skipped') or page.get('error') or 'empty')
        n = len(page.get('content', ''))
        print(f'  [{status:<14}] {n:>6} chars  {domain_of(u)[:34]:<34} '
              f'{str(page.get("pub_date")):>10}')
        if page.get('content'):
            pages[u] = page
    print(f'\n  pages with content: {len(pages)}/{len(candidates)}')
    if not pages:
        print('\n  >> DIAGNOSIS: nothing scraped. Paywalls, blocks, or dead links.')
        return

    # Show a sample so boilerplate-only pages are visible
    rule('3b. SAMPLE SCRAPED TEXT  (is this a real article?)')
    for u, p in list(pages.items())[:3]:
        n = 700 if args.full_text else 260
        print(f'  [{p.get("domain")}] {str(p.get("title"))[:70]}')
        print(f'      {p["content"][:n]}')
        print()

    # 4. verify + extract
    rule('4. RELEVANCE VERIFY')
    from google import genai
    from google.genai.types import HttpOptions
    client = genai.Client(vertexai=True,
                          project=os.environ.get('GCP_PROJECT_ID',
                                                 'ai-sandbox-dev-f139'),
                          location='us-central1',
                          http_options=HttpOptions(api_version='v1'))

    facts = []
    for u, p in pages.items():
        f = verify_and_extract(u, p, ev, client, args.model)
        facts.append(f)
        if f.get('error'):
            print(f'  [ERROR ] {f["error"][:90]}')
            continue
        tag = 'KEEP  ' if f.get('is_about_target_event') else 'REJECT'
        print(f'  [{tag}] {f.get("domain","")[:28]:<28} '
              f'scope={str(f.get("geographic_scope")):<9} '
              f'{str(p.get("title"))[:40]}')
        print(f'           {str(f.get("relevance_reason"))[:120]}')

    rel = [f for f in facts if f.get('is_about_target_event')]
    rule('5. EXTRACTED FACTS  (from accepted articles)')
    for f in rel:
        bits = []
        if f.get('extent_number'):
            bits.append(f'{f["extent_number"]:,} {f.get("extent_unit")} '
                        f'({f.get("extent_scope")}) as-of {f.get("extent_as_of_date")}')
        if f.get('contained_pct') is not None:
            bits.append(f'{f["contained_pct"]}% contained')
        nd = {k: v for k, v in (f.get('notable_dates') or {}).items() if v}
        if nd:
            bits.append(str(nd))
        if f.get('validated_features'):
            bits.append('in-chip: ' +
                        ', '.join(x['name'] for x in f['validated_features']))
        dropped = [x for x in (f.get('dropped_features') or [])
                   if x.get('name') != '__geocoder__']
        if dropped:
            bits.append(f'{len(dropped)} features off-chip')
        print(f'  [{f.get("domain","")[:26]:<26}] ' +
              ('; '.join(bits) if bits else '(no facts extracted)'))

    cov = coverage(facts)
    rule('6. COVERAGE')
    for k, v in cov.items():
        print(f'  {k}: {v}')

    rule('DIAGNOSIS')
    n_rej = len(facts) - len(rel)
    if not rel:
        print(f'  All {len(facts)} articles rejected. Read the reasons in step 4:')
        print('    - "different storm/year/region" -> search is pulling the '
              'wrong event; fix queries')
        print('    - "does not mention <county>"    -> gate still applying a '
              'county filter; fix prompt')
        print('    - boilerplate in step 3b         -> scraping cookie/paywall '
              'shells, not articles')
    else:
        print(f'  {len(rel)} accepted, {n_rej} rejected.')
        if not cov['has_extent']:
            print('  No local extent figure — common outside fires; the event '
                  'still contributes dates and features.')
        if not cov['has_features']:
            print('  No in-chip features. Check NOMINATIM_URL is reachable, '
                  'else everything is dropped as ungeocodable.')


if __name__ == '__main__':
    main()
