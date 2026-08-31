"""
Step -1: search for articles with targeted, event-anchored queries.

The original articles.csv was built from generic FEMA declaration titles like
"SEVERE STORMS AND FLOODING", which match any storm anywhere. Measured
relevance on a diverse 20-event sample was 35%.

This builds queries anchored on county + state + date + type, plus the named
storm if the FEMA title contains one. Writes Data/articles_v2.csv:
    event_id, url, query, rank

Usage:
    python pipeline/search_articles.py --event 5341 453 1346
    python pipeline/search_articles.py --limit 50 --per-query 8
    python pipeline/search_articles.py --all --workers 4
"""

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urlparse


EVENTS_PATH = 'Data/events_processed.json'
OUT_CSV = 'Data/articles_v2.csv'

SKIP_DOMAINS = {
    'facebook.com', 'm.facebook.com', 'youtube.com', 'youtu.be',
    'twitter.com', 'x.com', 'instagram.com', 'tiktok.com', 'pinterest.com',
    'irs.gov', 'guycarp.com', 'augurisk.com', 'ladrc.org', 'linkedin.com',
    'amazon.com', 'ebay.com', 'zillow.com', 'realtor.com', 'tripadvisor.com',
    # generic place/demographic pages that match county names but carry no
    # event information — these dominated results for weakly-named events
    'maps.apple.com', 'mapsof.net', 'city-data.com', 'rentometer.com',
    'nursinghomes.com', 'niche.com', 'bestplaces.net', 'areavibes.com',
    'homes.com', 'redfin.com', 'trulia.com', 'apartments.com',
    'census.gov', 'usa.com', 'zip-codes.com', 'countyoffice.org',
    'yelp.com', 'mapquest.com', 'google.com', 'bing.com',
}

# Named-storm patterns that appear in FEMA declaration titles
NAMED_STORM_RE = re.compile(
    r'\b(HURRICANE|TROPICAL STORM|TYPHOON|TROPICAL DEPRESSION)\s+([A-Z][A-Z\-]+)\b')
NAMED_FIRE_RE = re.compile(r'\b([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?)\s+FIRE\b')


STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands',
    'GU': 'Guam', 'AS': 'American Samoa', 'MP': 'Northern Mariana Islands',
    'DC': 'District of Columbia',
}

# FEMA county strings carry a parenthetical designator, e.g.
#   "Klamath (County)", "Petersburg (Borough)", "St. Croix (Island)"
# Strip it and remember the designator so we can phrase the query naturally.
COUNTY_DESIGNATOR_RE = re.compile(r'\s*\(([^)]+)\)\s*$')


def clean_county(raw):
    """('Klamath (County)') -> ('Klamath', 'County').

    Returns (bare_name, designator). Designator defaults to 'County'.
    """
    if not raw:
        return '', 'County'
    raw = raw.strip()
    m = COUNTY_DESIGNATOR_RE.search(raw)
    if m:
        return COUNTY_DESIGNATOR_RE.sub('', raw).strip(), m.group(1).strip()
    return raw, 'County'


def place_phrase(raw_county, state_abbr):
    """Natural search phrase: '"Klamath County" Oregon'."""
    bare, desig = clean_county(raw_county)
    state = STATE_NAMES.get((state_abbr or '').strip().upper(), state_abbr or '')
    if bare:
        return f'"{bare} {desig}" {state}'.strip()
    return state


def month_year(date_str):
    try:
        d = datetime.strptime(date_str, '%Y-%m-%d')
        return d.strftime('%B %Y')
    except Exception:
        return ''


def year_of(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y')
    except Exception:
        return ''


def build_queries(event):
    """Targeted query variants for one event. Most specific first."""
    title = (event.get('event') or '').strip()
    etype = (event.get('type') or '').strip()
    state_abbr = (event.get('state') or '').strip()
    state = STATE_NAMES.get(state_abbr.upper(), state_abbr)
    county_bare, desig = clean_county(event.get('county'))
    place = place_phrase(event.get('county'), state_abbr)
    start = event.get('start_date') or ''
    my = month_year(start)
    yr = year_of(start)

    queries = []

    # 1. Named storm / named fire — strongest anchor when present
    m = NAMED_STORM_RE.search(title.upper())
    if m:
        storm = f'{m.group(1).title()} {m.group(2).title()}'
        if county_bare:
            queries.append(f'"{storm}" {place} damage')
        queries.append(f'"{storm}" {state} damage {yr}')
    mf = NAMED_FIRE_RE.search(title.upper())
    if mf and etype == 'Fire':
        fire = f'{mf.group(1).title()} Fire'
        queries.append(f'"{fire}" acres contained {yr}')
        if county_bare:
            queries.append(f'"{fire}" {place}')

    # 2. Place + type + month/year — the core geographic anchor
    if place:
        queries.append(f'{place} {etype.lower()} {my}')
        queries.append(f'{place} {etype.lower()} damage {yr}')

    # 3. Authoritative-domain hints
    if county_bare:
        queries.append(f'site:weather.gov {county_bare} {desig} {state} {etype.lower()} {yr}')
    if etype == 'Fire' and county_bare:
        queries.append(f'site:inciweb.wildfire.gov {county_bare} {state} fire {yr}')

    # 4. Fallback: raw title + place + year
    if state:
        queries.append(f'{title.title()} {state} {yr}')

    # Dedupe, keep order
    seen, out = set(), []
    for q in queries:
        q = re.sub(r'\s+', ' ', q).strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


_DDGS_CLS = None
_DDGS_CHECKED = False


def _get_ddgs():
    """Resolve the DDGS class once. Returns None if the package is missing."""
    global _DDGS_CLS, _DDGS_CHECKED
    if _DDGS_CHECKED:
        return _DDGS_CLS
    _DDGS_CHECKED = True
    for mod in ('ddgs', 'duckduckgo_search'):
        try:
            _DDGS_CLS = __import__(mod, fromlist=['DDGS']).DDGS
            return _DDGS_CLS
        except ImportError:
            continue
    import sys
    print('\nERROR: no search backend. Install with:\n'
          f'    {sys.executable} -m pip install ddgs\n', flush=True)
    return None


def search_ddg(query, max_results=8):
    """DuckDuckGo search. Returns list of URLs, empty on any failure.

    Never raises: a harvest run spans hours across thousands of events, and a
    transient search error or a missing dependency should not discard
    everything already collected.
    """
    DDGS = _get_ddgs()
    if DDGS is None:
        return []
    urls = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                u = r.get('href') or r.get('url') or ''
                if u.startswith('http'):
                    urls.append(u)
    except Exception:
        pass
    return urls


def domain_of(u):
    try:
        return urlparse(u).netloc.lower().replace('www.', '').replace('m.', '')
    except Exception:
        return ''


def search_event(eid, event, per_query=8, max_queries=5, sleep=0.6):
    queries = build_queries(event)[:max_queries]
    rows, seen = [], set()
    for q in queries:
        for rank, u in enumerate(search_ddg(q, per_query)):
            d = domain_of(u)
            if d in SKIP_DOMAINS or u in seen:
                continue
            seen.add(u)
            rows.append({'event_id': eid, 'url': u, 'query': q, 'rank': rank})
        time.sleep(sleep)
    return rows


def load_existing(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                done.add(row['event_id'])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--out', default=OUT_CSV)
    ap.add_argument('--event', nargs='+', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--per-query', type=int, default=8)
    ap.add_argument('--max-queries', type=int, default=5)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()

    events = json.load(open(args.events))
    done = load_existing(args.out)
    if done:
        print(f'Resuming: {len(done)} events already searched')

    targets = []
    for eid, v in events.items():
        if 'error' in v or eid in done:
            continue
        if args.event and int(eid) not in args.event:
            continue
        targets.append((eid, v))
    targets.sort(key=lambda x: int(x[0]))
    if args.limit:
        targets = targets[:args.limit]
    if not args.all and not args.event and not args.limit:
        print('Pass --event, --limit, or --all'); return

    print(f'Searching {len(targets)} events '
          f'(<= {args.max_queries} queries x {args.per_query} results each)')

    # Show sample queries so the user can sanity-check anchoring
    if targets:
        eid, v = targets[0]
        print(f'\nExample queries for event {eid} ({v.get("type")}, '
              f'{v.get("county")} County, {v.get("state")}):')
        for q in build_queries(v)[:args.max_queries]:
            print(f'   {q}')
        print()

    write_header = not os.path.exists(args.out)
    fh = open(args.out, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(fh, fieldnames=['event_id', 'url', 'query', 'rank'])
    if write_header:
        w.writeheader()

    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(search_event, eid, v, args.per_query, args.max_queries): eid
                for eid, v in targets}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows = fut.result()
            except Exception as e:
                print(f'  error: {e}'); continue
            for r in rows:
                w.writerow(r)
            total += len(rows)
            fh.flush()
            if i % 10 == 0:
                print(f'  {i}/{len(targets)} events, {total} urls', flush=True)

    fh.close()
    print(f'\nWrote {total} urls for {len(targets)} events -> {args.out}')


if __name__ == '__main__':
    main()
