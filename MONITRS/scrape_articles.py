"""
MONITRS v2 — Language + image pipeline for generated_articles.csv
Event information comes only from the articles (no FEMA / FIRMS data).

For each query_id (event):
  1. Scrape every article URL
  2. One LLM call per article to extract location-event pairs
  3. Combine the pairs for the query_id and ask the LLM for damage centers
     (one per distinct location if the articles describe separate events)
  4. Download Sentinel-2 images for each (lat, lon, date range): an RGB PNG
     and a GeoTIFF of all bands (same name, .tif) per date; --rgb-only skips
     the GeoTIFFs

Processes --batch-size events per run (default 5), skipping events already in
the output file, so rerunning continues with the next batch.

Usage:
    export GEMINI_API_KEY=your-api-key
    export EE_PROJECT_ID=your-earth-engine-project
    python MONITRS/scrape_articles.py
    python MONITRS/scrape_articles.py --batch-size 5 --no-images
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import sys
import datetime
import json
import re
import argparse
import numpy as np
import urllib.request
from os.path import join, isfile
from os import makedirs
from dateutil.relativedelta import relativedelta
import ee
from google import genai
from PIL import Image
from time import sleep

# --- Config ---
PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-project-id')

EE_PROJECT_ID = os.environ.get('EE_PROJECT_ID', PROJECT_ID)

# Gemini API key client — reads GEMINI_API_KEY from the environment
client = genai.Client()
MODEL = "gemini-3.5-flash-lite"

try:
    ee.Initialize(project=EE_PROJECT_ID)
except Exception:
    print("Warning: Earth Engine not initialized (images won't download)")

ODIR = 'Data/images'
ARTICLES_FILE = 'Data/generated_articles.csv'
OUTPUT_FILE = 'Data/generated_events_processed.json'

BLACK_LIST = ['google', 'wikipedia', 'youtube', 'twitter', 'facebook', 'instagram',
              'linkedin', 'pinterest', 'reddit', 'quora', 'tiktok', 'tumblr']

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'}

MAX_ARTICLE_CHARS = 200000

# Imagery days before start_date / after end_date when the LLM gives no estimate
DEFAULT_PRE_DAYS = 30
DEFAULT_POST_DAYS = 30
MAX_BUFFER_DAYS = 180   # cap on each LLM buffer, including after widening
MAX_DAMAGE_DAYS = 180   # cap on the LLM's start_date..end_date damage period
MIN_IMAGES = 6          # keep widening the buffers until at least this many clear images
MAX_IMAGES = 30         # most images downloaded per location; sampled evenly over the window

# Every Sentinel-2 L2A spectral band (B10 is not in the surface-reflectance
# product) plus SCL, the scene classification layer (cloud, shadow, water,
# snow, ...). Saved as a uint16 GeoTIFF next to each RGB PNG, on the same
# 512x512 grid: spectral bands are reflectance x 10000, SCL is the class code.
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']
IMG_DIMENSIONS = '512x512'

# Errors meaning the articles had nothing usable; these events are not retried
EMPTY_ERRORS = {'no_content', 'no_location_events'}


def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"  [{ts}] {msg}")
    sys.stdout.flush()


# --- Article scraping ---

def get_article_content(url):
    try:
        response = requests.get(url, timeout=15, headers=HEADERS)
        if response.status_code == 200:
            # Pass bytes so BeautifulSoup detects the page encoding (non-English sites)
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.title.string if soup.title else ''
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            return title, content
    except Exception:
        pass
    return None, None


def scrape_articles(articles):
    """Scrape each article. Returns a list of dicts, one per URL, with text or a skip reason."""
    scraped = []
    for a in articles:
        url = a['url']
        entry = {'url': url, 'country': a['country'], 'lang': a['lang']}
        if any(b in url for b in BLACK_LIST):
            entry['skipped'] = 'blacklisted'
        else:
            title, content = get_article_content(url)
            if content and len(content) > 100:
                entry['title'] = (title or '').strip()
                entry['text'] = content[:MAX_ARTICLE_CHARS]
            else:
                entry['skipped'] = 'no_content'
        scraped.append(entry)
    return scraped


# --- LLM calls ---

def llm_call(prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text.strip()
        except Exception as e:
            if attempt < retries:
                sleep(5 * (attempt + 1))
            else:
                log(f"LLM failed after {retries + 1} attempts: {e}")
                return None


def parse_json(raw):
    if not raw:
        return None
    try:
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def extract_location_events(text, title, country):
    prompt = f"""
    Task: Extract locations where a natural disaster caused VISIBLE PHYSICAL CHANGES
    detectable from satellite imagery, AND contextual impacts on people/infrastructure.

    The article below was collected for a natural disaster in {country}.
    It may be written in any language; answer in English.

    For each location, extract:
    - location: the specific place name (city, district, road, river, landmark), followed by
      its region and country so it can be found on a map, e.g. "Glen Rose, Texas, USA"
    - date: YYYY-MM-DD when this happened at this location (null if the article gives no date)
    - event: what happened (1 short sentence)
    - type: "visual" (physically visible from satellite) or "contextual" (impacts on people)
    - disaster_type: e.g. flood, wildfire, earthquake, storm, landslide, extreme temperature
    - duration_days: estimated number of days the damage at this location takes to develop,
      from onset until it stops changing (e.g. a flash flood rising and receding, a fire
      spreading, a drought deepening). Use the article if it says; otherwise estimate.

    Only include the disaster the article is about. Skip places that are only mentioned in passing.
    No political statements or general commentary.

    Return as JSON array:
    [{{"location": "Glen Rose, Texas, USA", "date": "2022-07-18", "event": "Fire burned 6,700 acres", "type": "visual", "disaster_type": "wildfire", "duration_days": 10}},
     {{"location": "Highway 89, Arizona, USA", "date": "2022-04-19", "event": "Highway closed due to fire", "type": "contextual", "disaster_type": "wildfire", "duration_days": 3}}]

    If no events found, return: []

    Article title: {title}
    Article content: {text}
    """
    parsed = parse_json(llm_call(prompt))
    return parsed if isinstance(parsed, list) else []


def estimate_damage_center(loc_events):
    prompt = f"""
    Task: The location-event pairs below were extracted from several news articles about
    one natural disaster. Estimate the geographic center and extent of the damage area so
    that satellite images can be taken of it.

    If, based on location, the pairs seem to describe two or more different events (for
    example places too far apart to fit in one ~30 km satellite image, or separate disasters),
    return a SEPARATE entry with its own lat and lon for each distinct location.
    Otherwise return a single entry.

    For each entry:
    - lat, lon: decimal degrees (WGS84) of the CENTER of the actual damage/impact area
    - radius_km: approximate radius of the affected area in kilometers
    - start_date: YYYY-MM-DD, when the damage described in the pairs began (the onset, not the
      report date). For long-running conditions such as a multi-year drought, start near the
      reported impacts, NOT at the original onset years earlier.
    - end_date: YYYY-MM-DD, estimated date the visible damage stops developing. Do NOT just copy
      the last reported date: use the pairs' duration_days and how this kind of damage evolves
      (e.g. floodwater receding, a fire spreading until contained, a drought deepening, lava
      flows advancing). Sudden events may end within days; slow ones can take months.
      start_date to end_date must be at most {MAX_DAMAGE_DAYS} days.
    - pre_event_days: days of satellite imagery needed before start_date for a clear baseline
      (longer for slow-onset events where conditions degraded gradually; at most {MAX_BUFFER_DAYS})
    - post_event_days: days of imagery needed after end_date to show the aftermath and any
      lasting change or recovery (e.g. burn scars, standing water, debris, regrowth; at most {MAX_BUFFER_DAYS})
    - disaster_type: e.g. flood, wildfire, earthquake, storm
    - event_ids: the "id" values of the pairs below that belong to this entry
    - reasoning: key landmarks at the center of damage and why you chose this point
    - duration_reasoning: why you chose this damage period and these pre/post buffers

    Return as JSON array:
    [{{"lat": 40.12, "lon": -100.45, "radius_km": 15, "start_date": "2022-07-15", "end_date": "2022-07-25",
       "pre_event_days": 30, "post_event_days": 90, "disaster_type": "wildfire", "event_ids": [0, 2, 3],
       "reasoning": "Fire centered along Republican River",
       "duration_reasoning": "Fire spread for ~10 days before containment; burn scar visible for months"}}]

    Location-event pairs:
    {json.dumps(loc_events, ensure_ascii=False, indent=1)}
    """
    parsed = parse_json(llm_call(prompt))
    if isinstance(parsed, dict):
        parsed = [parsed]
    return parsed if isinstance(parsed, list) else []


def valid_center(c, reported_dates=()):
    """Check an LLM center has usable coordinates and dates; fills a missing end_date and
    trims a damage period longer than MAX_DAMAGE_DAYS to the one around reported_dates."""
    try:
        lat, lon = float(c['lat']), float(c['lon'])
        start = datetime.datetime.strptime(c['start_date'], '%Y-%m-%d')
        end = datetime.datetime.strptime(c.get('end_date') or c['start_date'], '%Y-%m-%d')
    except (KeyError, TypeError, ValueError):
        return False
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return False
    if end < start:
        start, end = end, start
    if (end - start).days > MAX_DAMAGE_DAYS:
        reported = sorted(d for d in reported_dates if start <= d <= end)
        if reported:
            start = max(start, reported[0] - relativedelta(days=MAX_DAMAGE_DAYS // 2))
        end = min(end, start + relativedelta(days=MAX_DAMAGE_DAYS))
        log(f"Damage period {c['start_date']} to {c.get('end_date')} too long; "
            f"trimmed to {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    c['start_date'], c['end_date'] = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
    c['pre_event_days'] = buffer_days(c.get('pre_event_days'), DEFAULT_PRE_DAYS)
    c['post_event_days'] = buffer_days(c.get('post_event_days'), DEFAULT_POST_DAYS)
    return True


def buffer_days(value, default):
    """Clamp an LLM-estimated imagery buffer to [1, MAX_BUFFER_DAYS], or use the default."""
    try:
        return min(max(1, int(float(value))), MAX_BUFFER_DAYS)
    except (TypeError, ValueError):
        return default


def align_events_to_images(loc_events, image_dates, event_start):
    if not image_dates:
        return {}
    aligned = {d: [] for d in image_dates}
    for le in loc_events:
        event_date = le.get('date') or ''
        if not event_date:
            continue
        # Find the first image date on or after this event
        assigned = None
        for img_date in image_dates:
            if img_date >= event_date:
                assigned = img_date
                break
        # If no image after the event, assign to the last image
        if assigned is None:
            assigned = image_dates[-1]
        aligned[assigned].append(le)

    # Mark pre-event images
    for img_date in image_dates:
        if img_date < event_start:
            if not aligned[img_date]:
                aligned[img_date].append({
                    'type': 'pre_event',
                    'event': 'Pre-event baseline — no disaster activity yet'
                })
    return aligned


# --- Image download ---

def image_region(center, halfwidth):
    return ee.Geometry.Rectangle([
        [center[1] - halfwidth, center[0] - halfwidth],
        [center[1] + halfwidth, center[0] + halfwidth],
    ])


def date_mosaic(region, img_date, max_cloud_pct=50):
    """The mosaic of every scene over region on img_date (UTC), as used for the PNGs."""
    day_end = (datetime.datetime.strptime(img_date, '%Y-%m-%d') + relativedelta(days=1)).strftime('%Y-%m-%d')
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(img_date, day_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
            .sort('system:time_start')
            .mosaic())


def download_full_bands(mosaic, region, output_file):
    """Save all S2_BANDS of a mosaic as a GeoTIFF on the same grid as its RGB PNG.
    Returns True on success; never raises, so a failure here doesn't lose the PNG."""
    tmp = output_file + '.part'
    try:
        url = mosaic.select(S2_BANDS).toUint16().getDownloadURL({
            'region': region, 'dimensions': IMG_DIMENSIONS, 'format': 'GEO_TIFF',
        })
        r = requests.get(url, timeout=300)
        # A TIFF starts with II* (little-endian) or MM (big-endian); anything else
        # is an error page from Earth Engine
        if r.status_code != 200 or r.content[:2] not in (b'II', b'MM'):
            log(f"Full-band download failed for {os.path.basename(output_file)}: "
                f"HTTP {r.status_code} {r.content[:200]!r}")
            return False
        with open(tmp, 'wb') as f:
            f.write(r.content)
        os.replace(tmp, output_file)
        return True
    except Exception as e:
        log(f"Full-band download failed for {os.path.basename(output_file)}: {e}")
        if isfile(tmp):
            os.remove(tmp)
        return False


def download_images(center, start_date, end_date, event_idx, halfwidth=0.05,
                    pre_days=DEFAULT_PRE_DAYS, post_days=DEFAULT_POST_DAYS, disaster_type=None,
                    min_images=MIN_IMAGES, max_cloud_pct=50, full_bands=True):
    """Download clear Sentinel-2 images from pre_days before start_date to post_days after
    end_date (the LLM's estimated damage period), at most MAX_IMAGES sampled evenly. If fewer
    than min_images clear images, or none before the event, the buffers double (each up to
    MAX_BUFFER_DAYS). With full_bands, each kept RGB PNG also gets a GeoTIFF of all bands
    (same name, .tif)."""
    region = image_region(center, halfwidth)
    event_start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_clean = end_date[:10] if len(end_date) > 10 else end_date
    event_end = datetime.datetime.strptime(end_clean, '%Y-%m-%d')

    outdir = join(ODIR, str(event_idx))
    makedirs(outdir, exist_ok=True)
    # e.g. "extreme temperature" -> "_extreme_temperature", appended to image filenames
    type_slug = re.sub(r'[^a-z0-9]+', '_', (disaster_type or '').lower()).strip('_')
    suffix = f'_{type_slug}' if type_slug else ''

    downloaded, tried = [], set()
    while True:
        query_start = (event_start - relativedelta(days=pre_days)).strftime('%Y-%m-%d')
        query_end = min(event_end + relativedelta(days=post_days),
                        datetime.datetime.now()).strftime('%Y-%m-%d')
        downloaded += _download_window(region, query_start, query_end, outdir, event_idx,
                                       suffix, tried, max_cloud_pct, MAX_IMAGES - len(downloaded),
                                       full_bands)
        downloaded = sorted(set(downloaded))
        has_pre = any(d < start_date for d in downloaded)
        wider = (min(pre_days * 2, MAX_BUFFER_DAYS), min(post_days * 2, MAX_BUFFER_DAYS))
        if ((len(downloaded) >= min_images and has_pre) or len(downloaded) >= MAX_IMAGES
                or wider == (pre_days, post_days)):
            break
        pre_days, post_days = wider
        log(f"Only {len(downloaded)} images (pre-event: {has_pre}); widening to -{pre_days}/+{post_days} days")

    log(f"Downloaded {len(downloaded)} images total (window -{pre_days}/+{post_days} days)")
    return downloaded


def _download_window(region, query_start, query_end, outdir, event_idx, suffix, tried,
                     max_cloud_pct, max_new, full_bands=True):
    """Download not-yet-tried scene dates in [query_start, query_end), at most max_new
    of them, evenly spaced over the window."""
    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filterDate(query_start, query_end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct)) \
        .sort('system:time_start')

    timestamps = col.aggregate_array('system:time_start').getInfo()
    log(f"Found {len(timestamps)} scenes ({query_start} to {query_end})")

    unique_dates = []
    for ts in timestamps:
        d = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc).strftime('%Y-%m-%d')
        if d not in tried and d not in unique_dates:
            unique_dates.append(d)
    if len(unique_dates) > max_new > 0:
        step = len(unique_dates) / max_new
        unique_dates = [unique_dates[int(i * step)] for i in range(max_new)]
    elif max_new <= 0:
        unique_dates = []
    tried.update(unique_dates)

    downloaded = []
    for img_date in unique_dates:
        output_file = join(outdir, f'{event_idx}_{img_date}{suffix}.png')
        tif_file = output_file[:-len('.png')] + '.tif'
        day_end = (datetime.datetime.strptime(img_date, '%Y-%m-%d') + relativedelta(days=1)).strftime('%Y-%m-%d')
        mosaic = col.filterDate(img_date, day_end).mosaic()
        if isfile(output_file) and os.path.getsize(output_file) > 1000:
            downloaded.append(img_date)
            if full_bands and not isfile(tif_file):
                download_full_bands(mosaic, region, tif_file)
            continue
        try:
            url = mosaic.getThumbURL({
                'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000,
                'gamma': 1, 'dimensions': '512x512', 'region': region,
            })
            urllib.request.urlretrieve(url, output_file)
            img = Image.open(output_file)
            if img.format != 'PNG':
                img.save(output_file, 'PNG')
            img_array = np.array(img)
            mean_val = np.mean(img_array)
            black_pct = np.count_nonzero(img_array.sum(axis=2) == 0) / (img_array.shape[0] * img_array.shape[1])
            if mean_val < 25 or black_pct > 0.05:
                os.remove(output_file)
                continue
            downloaded.append(img_date)
            if full_bands:
                download_full_bands(mosaic, region, tif_file)
        except Exception:
            if isfile(output_file):
                os.remove(output_file)
            continue

    log(f"Downloaded {len(downloaded)}/{len(unique_dates)} new images")
    return downloaded


# --- Process one event ---

def process_event(query_id, articles, args=None):
    country = articles[0]['country']

    print(f"\n{'='*60}")
    print(f"EVENT {query_id}: {country}, {len(articles)} articles")

    # 1. Scrape articles
    log("Scraping articles")
    scraped = scrape_articles(articles)
    good = [a for a in scraped if 'text' in a]
    log(f"Scraped {len(good)}/{len(articles)} articles")

    if not good:
        return {'error': 'no_content', 'country': country, 'articles': scraped}

    # 2. Extract location-events, one LLM call per article
    loc_events = []
    for a in good:
        log(f"Extracting location-event pairs: {a['url'][:70]}")
        events = extract_location_events(a['text'], a['title'], a['country'])
        a['num_location_events'] = len(events)
        for le in events:
            if isinstance(le, dict):
                le['source_url'] = a['url']
                loc_events.append(le)
    for i, le in enumerate(loc_events):
        le['id'] = i
    visual = [le for le in loc_events if le.get('type') == 'visual']
    log(f"{len(visual)} visual + {len(loc_events) - len(visual)} contextual (combined)")

    if not loc_events:
        return {'error': 'no_location_events', 'country': country,
                'articles': [{k: v for k, v in a.items() if k != 'text'} for a in scraped]}

    # 3. Determine centers from the combined location-events
    log("LLM coordinate estimation")
    llm_centers = estimate_damage_center(
        [{k: le.get(k) for k in ('id', 'location', 'date', 'event', 'type', 'disaster_type', 'duration_days')}
         for le in loc_events])

    locations = []
    for c in llm_centers:
        ids = {str(x) for x in c.get('event_ids') or []} if isinstance(c, dict) else set()
        reported = []
        for le in loc_events:
            try:
                if not ids or str(le['id']) in ids:
                    reported.append(datetime.datetime.strptime(le.get('date') or '', '%Y-%m-%d'))
            except ValueError:
                pass
        if not isinstance(c, dict) or not valid_center(c, reported):
            log(f"Skipping invalid center: {c}")
            continue
        radius = c.get('radius_km') or 10
        c['halfwidth'] = min(max(0.05, radius / 111 / 2), 0.15)
        locations.append(c)
        log(f"Center {len(locations) - 1}: ({float(c['lat']):.4f}, {float(c['lon']):.4f}), "
            f"radius ~{radius}km, {c['start_date']} to {c['end_date']} "
            f"(imagery -{c['pre_event_days']}/+{c['post_event_days']} days)")

    if not locations:
        return {'error': 'no_center', 'country': country, 'location_events': loc_events,
                'llm_centers': llm_centers,
                'articles': [{k: v for k, v in a.items() if k != 'text'} for a in scraped]}

    # 4. Download images for each (lat, lon, date range)
    for k, c in enumerate(locations):
        c['image_dir_id'] = f'{query_id}_{k}'
        c['image_dates'] = []
        if not getattr(args, 'no_images', False):
            log(f"Downloading images for center {k}")
            try:
                c['image_dates'] = download_images(
                    [float(c['lat']), float(c['lon'])], c['start_date'], c['end_date'],
                    c['image_dir_id'], halfwidth=c['halfwidth'],
                    pre_days=c['pre_event_days'], post_days=c['post_event_days'],
                    disaster_type=c.get('disaster_type'),
                    full_bands=not getattr(args, 'rgb_only', False))
            except Exception as e:
                log(f"Download failed: {e}")
        else:
            log("Skipping image download (--no-images)")

        ids = {str(x) for x in c.get('event_ids') or []}
        c_events = [le for le in loc_events if str(le['id']) in ids] or loc_events
        c['event_image_alignment'] = align_events_to_images(c_events, c['image_dates'], c['start_date'])

    log(f"Done — {len(locations)} location(s)")

    return {
        'country': country,
        'locations': locations,
        'location_events': loc_events,
        'num_articles_scraped': len(good),
        'articles': [{k: v for k, v in a.items() if k != 'text'} for a in scraped],
    }


def save_results(results, path):
    # Write to a temp file then rename, so an interrupted write never empties the output
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=5, help='Number of events to process this run')
    parser.add_argument('--start', type=int, default=0, help='Only consider query_ids >= this')
    parser.add_argument('--no-images', action='store_true', help='Skip image download (text only)')
    parser.add_argument('--rgb-only', action='store_true',
                        help='Download only the RGB PNGs, not the all-band GeoTIFFs')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE, help='Output JSON file')
    parser.add_argument('--retry-empty', action='store_true',
                        help='Also retry events that previously had no content / no location events')
    args = parser.parse_args()

    df = pd.read_csv(ARTICLES_FILE)
    events = {int(qid): g[['url', 'country', 'lang']].to_dict('records')
              for qid, g in df.groupby('query_id')}

    # Load existing results for resume; events that errored are retried, except
    # events whose articles had nothing to extract (unless --retry-empty)
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} events in {args.output}")
    else:
        results = {}

    def needs_run(q):
        r = results.get(str(q))
        if r is None:
            return True
        if 'error' not in r:
            return False
        return args.retry_empty or r['error'] not in EMPTY_ERRORS

    todo = [q for q in sorted(events) if q >= args.start and needs_run(q)]
    batch = todo[:args.batch_size]
    print(f"Processing {len(batch)} events: {batch} ({len(todo) - len(batch)} left after this batch)")

    for i, query_id in enumerate(batch):
        print(f"\n[{i+1}/{len(batch)}]", end='')
        try:
            results[str(query_id)] = process_event(query_id, events[query_id], args)
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            break
        except Exception as e:
            log(f"FAILED: {e}")
            results[str(query_id)] = {'error': str(e)}
        finally:
            save_results(results, args.output)

    # Summary
    done = [results[str(q)] for q in batch if str(q) in results]
    n_ok = sum(1 for v in done if 'error' not in v)
    n_locs = sum(len(v.get('locations', [])) for v in done)
    n_imgs = sum(len(c.get('image_dates', [])) for v in done for c in v.get('locations', []))
    print(f"\n\n{'='*60}")
    print(f"DONE: {n_ok}/{len(batch)} events processed, {n_locs} locations, {n_imgs} images")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
