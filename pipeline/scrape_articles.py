"""
Step 0: re-scrape article content + publication dates for URLs in Data/articles.csv.

articles.csv is only (event_id, url) — content was never saved. This fetches
each page, extracts main text and publication date, and caches to
Data/article_content.json keyed by URL.

Prioritizes authoritative domains (weather.gov, nhc.noaa.gov, inciweb, fema.gov)
and skips junk (facebook, youtube, insurance marketing).

Usage:
    python pipeline/scrape_articles.py --event 0 1 2      # specific events
    python pipeline/scrape_articles.py --limit 200        # first N urls
    python pipeline/scrape_articles.py --workers 8
"""

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

csv.field_size_limit(sys.maxsize)

ARTICLES_CSV = 'Data/articles.csv'
OUT_PATH = 'Data/article_content.json'

# Domains with structured, factual disaster reporting — scrape these first
PRIORITY_DOMAINS = {
    'weather.gov', 'nhc.noaa.gov', 'ncei.noaa.gov', 'spc.noaa.gov',
    'inciweb.wildfire.gov', 'inciweb.nwcg.gov', 'fema.gov',
    'usgs.gov', 'water.noaa.gov', 'noaa.gov', 'nasa.gov', 'blogs.nasa.gov',
    'cnn.com', 'nytimes.com', 'washingtonpost.com', 'npr.org',
    'cbsnews.com', 'weather.com', 'accuweather.com',
}

# Domains with no useful factual content
SKIP_DOMAINS = {
    'facebook.com', 'm.facebook.com', 'youtube.com', 'youtu.be',
    'twitter.com', 'x.com', 'instagram.com', 'tiktok.com',
    'irs.gov', 'guycarp.com', 'augurisk.com', 'ladrc.org',
    'linkedin.com', 'pinterest.com', 'reddit.com',
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; MONITRS-research/2.0; academic dataset construction)'
}

DATE_META_PATTERNS = [
    r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)',
    r'<meta[^>]+name=["\']publish[-_]?date["\'][^>]+content=["\']([^"\']+)',
    r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([^"\']+)',
    r'<meta[^>]+itemprop=["\']datePublished["\'][^>]+content=["\']([^"\']+)',
    r'"datePublished"\s*:\s*"([^"]+)"',
    r'<time[^>]+datetime=["\']([^"\']+)',
]


def domain_of(url):
    try:
        d = urlparse(url).netloc.lower()
        return d.replace('www.', '').replace('m.', '')
    except Exception:
        return ''


def extract_pub_date(html, url):
    """Try meta tags, JSON-LD, then URL path for a YYYY-MM-DD."""
    for pat in DATE_META_PATTERNS:
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            raw = m.group(1)
            dm = re.search(r'(\d{4})-(\d{2})-(\d{2})', raw)
            if dm:
                return dm.group(0)
    # URL path like /2022/07/20/
    um = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
    if um:
        return f'{um.group(1)}-{um.group(2)}-{um.group(3)}'
    # Body text "July 20, 2022"
    months = ('January|February|March|April|May|June|July|August|September|'
              'October|November|December')
    bm = re.search(rf'\b({months})\s+(\d{{1,2}}),\s+(\d{{4}})', html)
    if bm:
        mon = ['january','february','march','april','may','june','july',
               'august','september','october','november','december'
               ].index(bm.group(1).lower()) + 1
        return f'{bm.group(3)}-{mon:02d}-{int(bm.group(2)):02d}'
    return None


def html_to_text(html):
    """Crude but robust main-text extraction: strip scripts/styles/tags."""
    html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<header[^>]*>.*?</header>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    # Unescape a few common entities
    for a, b in [('&nbsp;', ' '), ('&amp;', '&'), ('&quot;', '"'),
                 ('&#39;', "'"), ('&lt;', '<'), ('&gt;', '>')]:
        text = text.replace(a, b)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_title(html):
    m = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
    if m:
        return re.sub(r'\s+', ' ', m.group(1)).strip()[:300]
    return ''


def scrape_one(event_id, url, timeout=15):
    d = domain_of(url)
    if d in SKIP_DOMAINS:
        return url, {'event_id': event_id, 'skipped': 'junk_domain', 'domain': d}
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return url, {'event_id': event_id, 'error': f'HTTP {r.status_code}', 'domain': d}
        html = r.text
    except Exception as e:
        return url, {'event_id': event_id, 'error': str(e)[:120], 'domain': d}

    text = html_to_text(html)
    if len(text) < 250:
        return url, {'event_id': event_id, 'error': 'too_short', 'domain': d}

    return url, {
        'event_id': event_id,
        'domain': d,
        'title': extract_title(html),
        'pub_date': extract_pub_date(html, url),
        'content': text[:30000],
        'priority': d in PRIORITY_DOMAINS,
    }


def load_url_list(csv_path):
    rows = []
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            eid, url = row[0].strip(), row[1].strip()
            if url.startswith('http'):
                rows.append((eid, url))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=ARTICLES_CSV)
    parser.add_argument('--out', default=OUT_PATH)
    parser.add_argument('--event', nargs='+', type=int, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--priority-only', action='store_true',
                        help='Only scrape authoritative domains')
    args = parser.parse_args()

    cache = {}
    if os.path.exists(args.out):
        cache = json.load(open(args.out))
        print(f"Resumed {len(cache)} cached pages")

    all_rows = load_url_list(args.csv)
    targets = []
    for eid, url in all_rows:
        if url in cache:
            continue
        if args.event and int(eid) not in args.event:
            continue
        d = domain_of(url)
        if d in SKIP_DOMAINS:
            continue
        if args.priority_only and d not in PRIORITY_DOMAINS:
            continue
        targets.append((eid, url))
        if args.limit and len(targets) >= args.limit:
            break

    print(f"To scrape: {len(targets)} URLs")
    if not targets:
        return

    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(scrape_one, eid, url): (eid, url) for eid, url in targets}
        for i, fut in enumerate(as_completed(futs), 1):
            url, entry = fut.result()
            cache[url] = entry
            if entry.get('content'):
                n_ok += 1
            else:
                n_err += 1
            if i % 25 == 0:
                json.dump(cache, open(args.out, 'w'))
                print(f"  {i}/{len(targets)}  ok={n_ok} err={n_err}", flush=True)

    json.dump(cache, open(args.out, 'w'))

    # Stats
    with_content = [v for v in cache.values() if v.get('content')]
    with_date = [v for v in with_content if v.get('pub_date')]
    print(f"\nCached {len(cache)} pages")
    print(f"  With content:  {len(with_content)}")
    print(f"  With pub_date: {len(with_date)} ({100*len(with_date)/max(1,len(with_content)):.0f}%)")


if __name__ == '__main__':
    main()
