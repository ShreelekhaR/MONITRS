"""
MONITRS v2 — Caption generation from article texts
One caption per Sentinel-2 image date (the aligned prompt: the image dates are
given, with no hints about what each image shows).

For each location in the locations GeoJSON from scrape_articles.py:
  1. Take the articles its location-events came from (article_urls)
  2. Scrape them again and find each one's publish date with htmldate; cached in
     Data/article_texts.json
  3. One LLM call per location, with every article wrapped in its publish date so
     relative times ("yesterday", "last week") can be resolved per article
  4. Save the captions as {images_dir}/{image_dir_id}/captions.npz (dates[i] <->
     captions[i], plus the articles used)
At the end (and on Ctrl+C), every location with a captions.npz is written to the
GeoParquet dataset index --output.

Locations that already have a captions.npz are skipped, so rerunning continues
where it left off (use --overwrite to redo them). --input also accepts an older
scrape_articles.py events JSON, converted on the fly.

Usage:
    export GEMINI_API_KEY=your-api-key
    python MONITRS/generate_article_captions.py
    python MONITRS/generate_article_captions.py --limit 2
    python MONITRS/generate_article_captions.py --input Data/random_events_processed.geojson \
        --output Data/random_dataset.geoparquet --images-dir Data/random_images
    python MONITRS/generate_article_captions.py --limit 1 --dry-run
    python MONITRS/generate_article_captions.py --index-only    # just rebuild the GeoParquet
"""

import os
import re
import sys
import json
import datetime
import argparse

import requests
from bs4 import BeautifulSoup
from htmldate import find_date

# scrape_articles creates a Gemini client on import; --dry-run makes no LLM calls
if '--dry-run' in sys.argv or '--index-only' in sys.argv:
    os.environ.setdefault('GEMINI_API_KEY', 'unused')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrape_articles import llm_call, log, save_results, HEADERS, MAX_ARTICLE_CHARS
from geo_outputs import read_locations, captions_path, save_captions, write_geoparquet

INPUT_FILE = 'Data/generated_events_processed.geojson'
OUTPUT_FILE = 'Data/generated_dataset.geoparquet'
ODIR = 'Data/images'
ARTICLE_CACHE = 'Data/article_texts.json'   # {url: {'title', 'text', 'published'}}


def fetch_article(url):
    """Title, paragraph text and publish date (YYYY-MM-DD or None) of url, or None."""
    try:
        response = requests.get(url, timeout=15, headers=HEADERS)
    except Exception:
        return None
    if response.status_code != 200:
        return None
    # Pass bytes so BeautifulSoup detects the page encoding (non-English sites)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string if soup.title else ''
    content = ' '.join(p.get_text() for p in soup.find_all('p'))
    if len(content) <= 100:
        return None
    try:
        published = find_date(response.text, original_date=True, outputformat='%Y-%m-%d')
    except Exception:
        published = None
    # htmldate falls back to the download date for pages it can't date
    if published and published >= datetime.date.today().isoformat():
        published = None
    return {'title': (title or '').strip(), 'text': content[:MAX_ARTICLE_CHARS], 'published': published}


def fetch_articles(urls, cache):
    """Articles for urls, scraping (and caching) any not cached or cached without a publish date."""
    articles = []
    for url in urls:
        if 'published' not in cache.get(url, {}):
            fetched = fetch_article(url)
            if fetched:
                cache[url] = fetched
            elif url in cache:
                cache[url]['published'] = None   # keep the cached text; don't retry the date
            else:
                log(f"Could not scrape {url[:70]}")
                continue
        articles.append({'url': url, **cache[url]})
    return articles


def format_articles(articles):
    """Article texts, each wrapped in its publish date."""
    blocks = []
    for i, a in enumerate(articles):
        published = a.get('published') or 'unknown'
        blocks.append(f"=== Article {i + 1} | Published: {published} | {a['url']}\n"
                      f"Title: {a['title']}\n{a['text']}\n"
                      f"=== End of article {i + 1} (published {published})")
    return '\n\n'.join(blocks)


def parse_dated_lines(raw):
    """{date: caption} from 'YYYY-MM-DD: caption' lines."""
    captions = {}
    for line in (raw or '').splitlines():
        m = re.match(r'^\s*[*\-]*\s*\**(\d{4}-\d{2}-\d{2})\**\s*:\s*(.+)$', line)
        if m:
            captions[m.group(1)] = m.group(2).strip()
    return captions


def build_prompt(loc, articles):
    disaster_type = loc.get('disaster_type') or 'disaster'
    place = '; '.join((loc.get('places') or [])[:3]) or loc.get('country', '')
    dates_text = '\n    '.join(sorted(loc['image_dates']))
    articles_text = format_articles(articles)

    return f"""
    Task: Write a factual caption for each satellite image of a {disaster_type} event in {place}.
    Images centered at ({float(loc['lat']):.4f}, {float(loc['lon']):.4f}).
    The damage is estimated to have developed from {loc.get('start_date')} to {loc.get('end_date')}.

    These are optical (true color RGB) satellite images from Sentinel-2, taken on these dates:
    {dates_text}

    Base the captions on the news articles below. Each caption describes what the landscape most likely 
    looks like on one specific image date. They may be in any language; write in English.
    Use only what the articles report about this location; ignore other places they mention.
    For each date, use only what the articles say had happened by that date.
    Each article is wrapped with its publish date. Articles were published on different days, so read
    relative times ("yesterday", "last week", "on Sunday") against that article's own publish date.
    If an article's publish date is unknown, rely on the dates stated in its text.

    Based on each date:
    - Before the event: describe the normal landscape (river channels, land cover, vegetation, built-up areas, roads).
    - During the event: describe the active physical effects reported for that period.
    - Between reported events: reason about the likely state of the ground. For example, floodwater partly recedes and leaves sediment, or a burn scar stays the same until a new event. Do not invent new damage.
    - After the event: describe lasting effects (receding water, sediment and debris deposits, scoured channels, burn scars, early recovery).

    CAPTION RULES:
    - 1–2 sentences, present tense.
    - Describe only PHYSICAL CHANGES visible in visible light: standing or flowing water, sediment-colored water, mud and debris deposits, washed-out or covered roads, damaged structures, vegetation loss, burn scars (darkened ground), smoke, cloud cover.
    - Use specific figures from the article (flood extent, area burned, structures destroyed, %) only if the article gives them. Never estimate or invent numbers.
    - You may use place names and nearby rivers or landmarks from the article to say where changes are.
    - Do not mention anything you cannot see in an image: deaths, injuries, evacuations, shelter numbers, housing programs, quotes, road-closure orders (show physical road damage instead), or thermal "hotspots".
    - Do not mention that the images are taken from satellites.
    - Subtle changes may not show at 512 px, but still describe the known state of the ground.

    Return format — one line per date, every date listed above, nothing else:
    YYYY-MM-DD: caption

    EXAMPLE ARTICLE:
    "Colorado flood evacuees forced out again as new storm hits before recovery begins
    The Town of Pagosa Springs urged residents Sunday afternoon to shelter in place and avoid all nonessential travel.
    Author: Marc Sallinger (9 News)
    Published: 10:44 PM MST November 1, 2026
    Updated: 10:44 PM MST November 1, 2026

    PAGOSA SPRINGS, Colo — People in southwest Colorado who were forced from their homes by historic flooding just over a week ago faced a second round of floodwater Sunday. More evacuations were ordered and roads were closed across the region before rebuilding from the first storm had even begun.

    Karen Evans was evacuated from her home near Pagosa Springs when flooding hit the area last week. She has been staying in an Airbnb through a temporary housing program run by 211 Colorado, Mile High United Way and Airbnb.org. On Sunday, she started getting alerts that more flooding was on the way.

    “We haven’t even recovered from this flooding, and I don’t know what it’s going to look like over the next couple of days,” she said.

    Water is still flowing through her property, Evans said, and she is scared to stay inside her house for very long. She worries the new storm could damage it beyond repair, and she said all she can do now is wait.

    “I don’t know if I’ll have anything left to take,” she said.

    The Town of Pagosa Springs urged residents Sunday afternoon to shelter in place and avoid all nonessential travel. The town said continued rainfall, flooding and road damage had created hazardous conditions throughout Pagosa Springs and Archuleta County, and more rain could bring more flooding and road damage. Mandatory evacuation orders remain in effect where they have been issued, according to the town.

    In La Plata County, Sheriff Sean Smith closed all county roads at 2 p.m. Sunday because of dangerous flood conditions and debris flows on numerous roads, according to the county. He asked residents to stay home until further notice.

    By Sunday evening, La Plata County had ordered evacuations of several areas in Durango, Ignacio and Bayfield.

    Since the 211 Colorado Airbnb.org program launched Oct. 23, it has placed 74 evacuated households in Airbnb stays, according to 211 Colorado. Most came from Archuleta County, including 53 households from Pagosa Springs. The program is also housing 27 critical workers and first responders in the flood zone. As of Saturday, 211 Colorado expanded eligibility to households affected in Archuleta, Conejos, Rio Grande and La Plata counties."

    EXAMPLE OUTPUT:
    2026-10-15: The San Juan River winds through Pagosa Springs in a narrow, well-defined channel, with fall-dormant pastures, conifer-covered slopes and an intact road network across Archuleta and La Plata counties.
    2026-10-24: Floodwater spreads beyond the San Juan River channel through Pagosa Springs and low-lying parts of Archuleta County, with brown, sediment-laden standing water covering floodplain fields and roads near homes.
    2026-10-29: Water levels drop across Archuleta County, but water still flows across some properties near Pagosa Springs, and mud, sediment deposits and debris remain on the floodplain, with damaged road segments.
    2026-11-01: A second storm sends new floodwater through Archuleta County, where roads are already damaged, and debris flows cover many roads across La Plata County near Durango, Ignacio and Bayfield, with swollen, muddy rivers spreading onto the floodplains again.
    2026-11-08: Floodwater recedes again in Archuleta and La Plata counties, leaving scoured river channels, wide sediment and debris deposits, and washed-out or debris-covered roads across both counties.

    Articles:
    {articles_text}
    """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=INPUT_FILE,
                        help='Locations GeoJSON (or events JSON) from scrape_articles.py')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE, help='GeoParquet dataset index')
    parser.add_argument('--images-dir', type=str, default=ODIR, help='Folder holding one image folder per location')
    parser.add_argument('--limit', type=int, default=None, help='Caption at most this many locations')
    parser.add_argument('--overwrite', action='store_true', help='Redo locations that already have captions')
    parser.add_argument('--dry-run', action='store_true', help='Print the first prompt and exit (no LLM call)')
    parser.add_argument('--index-only', action='store_true', help='Only rebuild the GeoParquet index')
    parser.add_argument('--article-cache', type=str, default=ARTICLE_CACHE,
                        help='JSON cache of scraped article texts and publish dates')
    args = parser.parse_args()

    locations = read_locations(args.input)

    def write_index():
        n = write_geoparquet(locations, args.images_dir, args.output)
        print(f"{n} captioned locations -> {args.output}")

    if args.index_only:
        write_index()
        return

    cache = {}
    if os.path.exists(args.article_cache):
        with open(args.article_cache) as f:
            cache = json.load(f)

    todo = []
    for loc in locations:
        if not loc.get('image_dates'):
            continue
        if not os.path.isdir(os.path.join(args.images_dir, loc['image_dir_id'])):
            continue
        if args.overwrite or not os.path.exists(captions_path(args.images_dir, loc['image_dir_id'])):
            todo.append(loc)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} location(s) to caption")

    if args.dry_run:
        if todo:
            loc = todo[0]
            articles = fetch_articles(loc.get('article_urls') or [], cache)
            save_results(cache, args.article_cache)
            prompt = build_prompt(loc, articles)
            print(f"\n{'=' * 30} prompt for {loc['image_dir_id']} {'=' * 30}")
            print(prompt[:4000])
            if len(prompt) > 4000:
                print(f"    ... ({len(prompt)} chars total, {len(articles)} article(s))")
        return

    n_ok = 0
    try:
        for i, loc in enumerate(todo):
            dates = set(loc['image_dates'])
            print(f"\n[{i+1}/{len(todo)}] {loc['image_dir_id']} — {loc.get('disaster_type')}, {len(dates)} images")
            articles = fetch_articles(loc.get('article_urls') or [], cache)
            save_results(cache, args.article_cache)
            if not articles:
                log("No articles could be scraped, skipping")
                continue
            log(f"{len(articles)} article(s), published: "
                f"{', '.join(a.get('published') or 'unknown' for a in articles)}")

            # Keep only captions for the given image dates
            captions = parse_dated_lines(llm_call(build_prompt(loc, articles)))
            captions = {d: c for d, c in captions.items() if d in dates}
            missing = sorted(dates - captions.keys())
            if missing:
                log(f"missing captions for {len(missing)} date(s): {missing[:5]}")
            if not captions:
                log("no captions parsed")
                continue
            save_captions(captions_path(args.images_dir, loc['image_dir_id']), captions, articles)
            n_ok += 1
            log(f"{len(captions)} captions")
    except KeyboardInterrupt:
        print("\n\nInterrupted. Captions so far are saved.")

    print(f"\n{n_ok}/{len(todo)} locations captioned this run")
    write_index()


if __name__ == '__main__':
    main()
