"""
MONITRS v2 — Language pipeline (runs on laptop)
Processes all FEMA events: scrapes articles, extracts locations,
determines image centers, generates captions. NO image download.

Outputs Data/events_processed.json — feed this to the Workbench
for image download + QA generation.

Usage:
    export GCP_PROJECT_ID=your-project-id
    export FIRMS_MAP_KEY=your-firms-key
    python MONITRS/run_language_pipeline.py

    # Process specific batch:
    python MONITRS/run_language_pipeline.py --start 0 --end 1000
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import sys
import datetime
import json
import argparse
import numpy as np
import urllib.request
from os.path import join, isfile
from os import makedirs
from dateutil.relativedelta import relativedelta
from ddgs import DDGS
import ee
from google import genai
from PIL import Image
from time import sleep

# --- Config ---
PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-project-id')
LOCATION = os.environ.get('GCP_LOCATION', 'us-central1')
FIRMS_MAP_KEY = os.environ.get('FIRMS_MAP_KEY', '')

EE_PROJECT_ID = os.environ.get('EE_PROJECT_ID', PROJECT_ID)

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
MODEL = "gemini-2.5-flash"

try:
    ee.Initialize(project=EE_PROJECT_ID)
except Exception:
    print("Warning: Earth Engine not initialized (images won't download)")

ODIR = 'Data/images'

BLACK_LIST = ['google', 'wikipedia', 'youtube', 'twitter', 'facebook', 'instagram',
              'linkedin', 'pinterest', 'reddit', 'quora', 'tiktok', 'tumblr']

OUTPUT_FILE = 'Data/events_processed.json'


def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"  [{ts}] {msg}")
    sys.stdout.flush()


# --- Article scraping ---

def get_article_content(url):
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else ''
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            return title, content
    except Exception:
        pass
    return None, None


def scrape_articles(links, start_date=None, event_name=None):
    year = start_date[:4] if start_date else None
    content = ''
    scraped = 0
    for link in links:
        if any(b in link for b in BLACK_LIST):
            continue
        title, article_content = get_article_content(link)
        if article_content and len(article_content) > 100:
            text_to_check = (article_content + ' ' + (title or '') + ' ' + link).lower()
            if year and year not in text_to_check:
                if event_name and event_name.lower().split()[0] not in text_to_check:
                    continue
            content += article_content + '\n'
            scraped += 1
    return content, scraped


def search_ddg(event_name, event_type, county, state, start_date, max_results=5):
    year = start_date[:4]
    queries = [
        f"{event_name} {event_type} {county} {state} {year}",
        f"{event_name} {state} {year}",
        f"\"{event_name}\" {year}",
    ]
    all_links = []
    seen = set()
    for query in queries:
        try:
            results = DDGS().text(query, max_results=max_results)
            for r in results:
                href = r.get('href', '')
                if href and href not in seen:
                    seen.add(href)
                    all_links.append(href)
            if all_links:
                break
            sleep(1)
        except Exception:
            sleep(2)
    return all_links


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


def extract_location_events(text, start_date, end_date, state, county):
    prompt = f"""
    Task: Extract locations where a natural disaster caused VISIBLE PHYSICAL CHANGES
    detectable from satellite imagery, AND contextual impacts on people/infrastructure.

    Event: Natural disaster in {county}, {state}, from {start_date} to {end_date}.

    For each location, extract:
    - location: the place name (city, road, river, landmark)
    - date: YYYY-MM-DD
    - event: what happened (1 short sentence)
    - type: "visual" (physically visible from satellite) or "contextual" (impacts on people)

    Only locations within or near {county}, {state}. No state/country names.
    No political statements or general commentary.

    Return as JSON array:
    [{{"location": "Glen Rose", "date": "2022-07-18", "event": "Fire burned 6,700 acres", "type": "visual"}},
     {{"location": "Highway 89", "date": "2022-04-19", "event": "Highway closed due to fire", "type": "contextual"}}]

    If no events found, return: []

    Article Content: {text}
    """
    raw = llm_call(prompt)
    if not raw:
        return []
    try:
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def estimate_damage_center(text, event_type, state, county, fema_center):
    prompt = f"""
    Task: Estimate the geographic center and extent of the damage area for this {event_type} event.

    News articles about a {event_type} in {county}, {state}.
    FEMA-reported center: ({fema_center[0]:.4f}, {fema_center[1]:.4f}).

    Estimate:
    1. Latitude and longitude of the CENTER of the actual damage/impact area
    2. Approximate RADIUS in kilometers of the affected area
    3. Key landmarks at the center of damage

    Return as JSON:
    {{"lat": 40.12, "lon": -100.45, "radius_km": 15, "reasoning": "Fire centered along Republican River"}}

    Article Content: {text}
    """
    raw = llm_call(prompt)
    if not raw:
        return None
    try:
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def estimate_damage_center_with_firms(text, event_type, state, county, fema_center, firms_center, firms_count):
    prompt = f"""
    Task: Given NASA FIRMS fire detection data AND news articles, estimate the precise
    center and extent of the fire damage area.

    Event: {event_type} in {county}, {state}.
    FEMA center: ({fema_center[0]:.4f}, {fema_center[1]:.4f})
    NASA FIRMS detected {firms_count} fire hotspots centered at ({firms_center[0]:.4f}, {firms_center[1]:.4f}).

    Using BOTH the FIRMS hotspot location AND the article details (specific roads,
    landmarks, towns affected), estimate:
    1. The best center point for a satellite image that captures the main damage area
    2. The radius in km
    3. Your reasoning — how do the article details refine the FIRMS center?

    The FIRMS center is the average of all hotspot detections. But the articles may
    indicate that the worst damage, most structures destroyed, or most significant
    visual changes were concentrated in a specific part of the burn area.

    Return as JSON:
    {{"lat": 40.12, "lon": -100.45, "radius_km": 15, "reasoning": "FIRMS shows broad burn but articles indicate worst damage near Glen Rose"}}

    Article Content: {text}
    """
    raw = llm_call(prompt)
    if not raw:
        return None
    try:
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def align_events_to_images(loc_events, image_dates, event_start):
    if not image_dates:
        return {}
    aligned = {d: [] for d in image_dates}
    for le in loc_events:
        event_date = le.get('date', '')
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


def generate_aligned_captions(text, aligned, event_type, county, state, center, strategy):
    center_info = f"Images centered at ({center[0]:.4f}, {center[1]:.4f})."
    if strategy == 'firms':
        center_info += " Location from NASA FIRMS fire hotspot detections."

    alignment_text = ""
    for img_date in sorted(aligned.keys()):
        events = aligned[img_date]
        if not events:
            alignment_text += f"\n{img_date}: (no specific events reported by this date)"
        else:
            event_strs = [le.get('event', '') for le in events]
            alignment_text += f"\n{img_date}: {'; '.join(e for e in event_strs if e)}"

    prompt = f"""
    Task: Write a factual caption for each satellite image of a {event_type} event in {county}, {state}.
    {center_info}

    These are optical (true color RGB) satellite images from Sentinel-2.

    Below are image dates with the article-reported events that occurred by that date:
    {alignment_text}

    For each date, write a 1-2 sentence caption that:
    1. Describes what PHYSICAL CHANGES may be present in the landscape by this date
    2. Cites specific facts from articles (acres burned, structures destroyed, flood extent, %)
    3. For pre-event dates: describe the baseline landscape
    4. Uses present tense

    IMPORTANT — these are optical/visible-light images, NOT thermal or infrared:
    - Do NOT mention "hotspots" — those are only visible in thermal/infrared sensors
    - DO mention: burn scars (darkened terrain), smoke plumes, flooding (standing water),
      vegetation loss, debris fields, structural damage
    - Subtle changes may not be obvious in a 512px image but describe the known state

    Return format — one line per date:
    YYYY-MM-DD: caption
    """
    return llm_call(prompt) or ''


def generate_captions(text, dates, event_type, county, state, center, strategy):
    center_info = f"The satellite images are centered at ({center[0]:.4f}, {center[1]:.4f})."
    if strategy == 'firms':
        center_info += " This location was determined from NASA FIRMS active fire hotspot detections."

    prompt = f"""
    Task: Write a factual caption for each satellite image date of a {event_type} event in {county}, {state}.

    {center_info}
    Only describe what would be visible at THIS specific location.

    Each caption should:
    1. State specific facts from the articles — cite numbers (acres, homes, %) when available
    2. Describe the VISUAL change from the previous image
    3. Build a progressive narrative: pre-event → onset → peak → aftermath
    4. Be 1-2 sentences, factual and concrete

    RULES:
    - Use present tense ("burn scar covers", not "would show")
    - Never say "satellite imagery would depict"
    - Pull specific details from the articles
    - If a date is before the event started, describe the baseline landscape

    Image dates (in order): {dates}
    Article content: {text}

    Return format — one line per date:
    YYYY-MM-DD: caption
    """
    return llm_call(prompt) or ''


# --- FIRMS ---

def get_firms_hotspots(fema_center, start_date, end_date):
    if not FIRMS_MAP_KEY:
        return None
    lat, lon = fema_center
    end_clean = end_date[:10] if len(end_date) > 10 else end_date
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_clean, '%Y-%m-%d')

    west, south = lon - 2, lat - 2
    east, north = lon + 2, lat + 2
    area = f"{west},{south},{east},{north}"

    all_hotspots = []
    for source in ['VIIRS_SNPP_SP', 'MODIS_SP']:
        chunk_start = start_dt
        while chunk_start < end_dt:
            chunk_days = min(5, (end_dt - chunk_start).days + 1)
            date_str = chunk_start.strftime('%Y-%m-%d')
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                   f"{FIRMS_MAP_KEY}/{source}/{area}/{chunk_days}/{date_str}")
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200 and len(resp.text.strip().split('\n')) > 1:
                    lines = resp.text.strip().split('\n')
                    header = lines[0].split(',')
                    lat_idx = header.index('latitude')
                    lon_idx = header.index('longitude')
                    date_idx = header.index('acq_date')
                    for line in lines[1:]:
                        parts = line.split(',')
                        h_lat, h_lon = float(parts[lat_idx]), float(parts[lon_idx])
                        dist = ((h_lat - lat)**2 + (h_lon - lon)**2)**0.5 * 111
                        if dist <= 100:
                            all_hotspots.append({
                                'lat': h_lat, 'lon': h_lon,
                                'date': parts[date_idx], 'source': source,
                            })
            except Exception:
                pass
            chunk_start += relativedelta(days=chunk_days)
        if all_hotspots:
            break

    if not all_hotspots:
        return None

    seen = set()
    unique = []
    for h in all_hotspots:
        key = (round(h['lat'], 3), round(h['lon'], 3), h['date'])
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique


# --- Image download ---

def download_images(center, start_date, end_date, event_idx,
                    halfwidth=0.05, pre_days=14, post_days=14, max_cloud_pct=50):
    region = ee.Geometry.Rectangle([
        [center[1] - halfwidth, center[0] - halfwidth],
        [center[1] + halfwidth, center[0] + halfwidth],
    ])
    event_start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_clean = end_date[:10] if len(end_date) > 10 else end_date
    event_end = datetime.datetime.strptime(end_clean, '%Y-%m-%d')
    query_start = (event_start - relativedelta(days=pre_days)).strftime('%Y-%m-%d')
    query_end = (event_end + relativedelta(days=post_days)).strftime('%Y-%m-%d')

    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filterDate(query_start, query_end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct)) \
        .sort('system:time_start')

    num = col.size().getInfo()
    log(f"Found {num} scenes ({query_start} to {query_end})")

    if num == 0:
        return []

    outdir = join(ODIR, str(event_idx))
    makedirs(outdir, exist_ok=True)

    img_list = col.toList(num)
    unique_dates = []
    seen = set()
    for i in range(num):
        try:
            d = ee.Image(img_list.get(i)).date().format('YYYY-MM-dd').getInfo()
            if d not in seen:
                seen.add(d)
                unique_dates.append(d)
        except Exception:
            continue

    downloaded = []
    for img_date in unique_dates:
        output_file = join(outdir, f'{event_idx}_{img_date}.png')
        if isfile(output_file) and os.path.getsize(output_file) > 1000:
            downloaded.append(img_date)
            continue
        day_end = (datetime.datetime.strptime(img_date, '%Y-%m-%d') + relativedelta(days=1)).strftime('%Y-%m-%d')
        mosaic = col.filterDate(img_date, day_end).mosaic()
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
        except Exception:
            if isfile(output_file):
                os.remove(output_file)
            continue

    log(f"Downloaded {len(downloaded)}/{len(unique_dates)} images")
    return downloaded


# --- Process one event ---

def process_event(event_idx, links, df, args=None):
    row = df[df['index'] == event_idx].iloc[0]
    fema_lat, fema_lon = row['lat'], row['lon']
    fema_center = (fema_lat, fema_lon)
    state = row['state']
    county = row['designatedArea']
    event_type = row['incidentType']

    start_date = row['incidentBeginDate']
    end_date = str(row['incidentEndDate'])
    if len(end_date) > 10:
        end_date = end_date[:10]

    event_name = row['declarationTitle']

    print(f"\n{'='*60}")
    print(f"EVENT {event_idx}: {event_name}")
    print(f"  {event_type} | {state} | {county} | {start_date} to {end_date}")

    # 1. Scrape articles, fallback to DuckDuckGo
    log("Scraping articles")
    content, scraped = scrape_articles(links, start_date, event_name)
    log(f"Scraped {scraped}/{len(links)} articles, {len(content)} chars")

    if len(content) < 200:
        log("Insufficient content — searching DuckDuckGo")
        ddg_links = search_ddg(event_name, event_type, county, state, start_date)
        log(f"DDG found {len(ddg_links)} links")
        for link in ddg_links:
            if any(b in link for b in BLACK_LIST):
                continue
            title, article_content = get_article_content(link)
            year = start_date[:4]
            if article_content and len(article_content) > 100:
                text_to_check = (article_content + ' ' + (title or '') + ' ' + link).lower()
                if year not in text_to_check:
                    if event_name and event_name.lower().split()[0] not in text_to_check:
                        log(f"  [skip] {link[:60]}... (wrong year/event)")
                        continue
                content += article_content + '\n'
                scraped += 1
        log(f"After DDG: {scraped} articles, {len(content)} chars")

    if len(content) < 100:
        return {'error': 'no_content', 'event': event_name}

    # Truncate to avoid token limits (~4 chars per token, 500K token limit)
    if len(content) > 500000:
        log(f"Truncating content from {len(content)} to 500000 chars")
        content = content[:500000]

    # 2. Extract location-events
    log("Extracting location-event pairs")
    loc_events = extract_location_events(content, start_date, end_date, state, county)
    visual = [le for le in loc_events if le.get('type') == 'visual']
    contextual = [le for le in loc_events if le.get('type') != 'visual']
    log(f"{len(visual)} visual + {len(contextual)} contextual")

    # 3. Determine center
    center = list(fema_center)
    strategy = 'fema'
    halfwidth = 0.05
    llm_estimate = None
    llm_center = None
    firms_data = None

    # Always run LLM estimation
    log("LLM coordinate estimation")
    llm_estimate = estimate_damage_center(content, event_type, state, county, fema_center)
    llm_center = None
    if llm_estimate and llm_estimate.get('lat') and llm_estimate.get('lon'):
        llm_center = [llm_estimate['lat'], llm_estimate['lon']]
        radius = llm_estimate.get('radius_km', 10)
        halfwidth = min(max(0.05, radius / 111 / 2), 0.15)
        center = llm_center
        strategy = 'llm'
        log(f"LLM center: ({center[0]:.4f}, {center[1]:.4f}), radius ~{radius}km")

    # For fires, also get FIRMS — and give FIRMS data to LLM for refined estimate
    if event_type == 'Fire' and FIRMS_MAP_KEY:
        log("Querying FIRMS")
        hotspots = get_firms_hotspots(fema_center, start_date, end_date)
        if hotspots:
            h_lats = [h['lat'] for h in hotspots]
            h_lons = [h['lon'] for h in hotspots]
            firms_c = [float(np.mean(h_lats)), float(np.mean(h_lons))]
            firms_data = {'count': len(hotspots), 'center': firms_c}
            log(f"FIRMS center: ({firms_c[0]:.4f}, {firms_c[1]:.4f}), {len(hotspots)} hotspots")

            # Use FIRMS as primary center
            center = firms_c
            strategy = 'firms'
            halfwidth = 0.05

    if strategy == 'fema':
        log("All estimation failed, using FEMA center")

    # 4. Download images at the chosen center (skip with --no-images)
    image_dates = []
    if not getattr(args, 'no_images', False):
        log(f"Downloading images at {strategy} center ({center[0]:.4f}, {center[1]:.4f})")
        try:
            image_dates = download_images(center, start_date, end_date,
                                          event_idx, halfwidth=halfwidth)
        except Exception as e:
            log(f"Download failed: {e}")
    else:
        log("Skipping image download (--no-images)")

    # 5. Generate captions using FEMA date range
    log("Generating captions")
    caption_dates = pd.date_range(start_date, end_date, freq='5D').strftime('%Y-%m-%d').tolist()
    if end_date not in caption_dates:
        caption_dates.append(end_date)

    aligned = align_events_to_images(loc_events, caption_dates, start_date)

    captions = generate_aligned_captions(
        content, aligned, event_type, county, state, center, strategy)

    log(f"Done — {strategy} center")

    return {
        'event': row['declarationTitle'],
        'type': event_type,
        'state': state,
        'county': county,
        'start_date': start_date,
        'end_date': end_date,
        'fema_center': list(fema_center),
        'center': center,
        'strategy': strategy,
        'halfwidth': halfwidth,
        'llm_center': llm_center,
        'llm_estimate': llm_estimate,
        'firms': firms_data,
        'location_events': loc_events,
        'captions': captions,
        'event_image_alignment': {k: v for k, v in aligned.items()},
        'num_articles_scraped': scraped,
        'article_links': links,
        'image_dates': image_dates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start event index')
    parser.add_argument('--end', type=int, default=None, help='End event index (exclusive)')
    parser.add_argument('--no-images', action='store_true', help='Skip image download (text only)')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: Data/events_processed_{start}_{end}.json)')
    args = parser.parse_args()

    df = pd.read_csv('Data/FEMA_filtered.csv', header=0)
    articles_lines = open('Data/articles.csv', 'r').readlines()

    events = {}
    for line in articles_lines:
        parts = line.strip().split(',')
        idx = int(parts[0])
        url = parts[1]
        if idx not in events:
            events[idx] = []
        events[idx].append(url)

    # Set output file
    global OUTPUT_FILE
    if args.output:
        OUTPUT_FILE = args.output
    else:
        end_str = args.end if args.end else 'end'
        OUTPUT_FILE = f'Data/events_processed_{args.start}_{end_str}.json'

    # Filter to requested range
    event_ids = sorted(events.keys())
    if args.end:
        event_ids = [e for e in event_ids if args.start <= e < args.end]
    else:
        event_ids = [e for e in event_ids if e >= args.start]

    # Load existing results for resume
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} events already done")
    else:
        results = {}

    total = len(event_ids)
    done = 0

    for i, event_idx in enumerate(event_ids):
        if str(event_idx) in results:
            done += 1
            continue

        print(f"\n[{i+1}/{total}]", end='')

        try:
            result = process_event(event_idx, events[event_idx], df, args)
            results[str(event_idx)] = result
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            sys.exit(0)
        except Exception as e:
            log(f"FAILED: {e}")
            results[str(event_idx)] = {'error': str(e)}

        # Save every 10 events
        done += 1
        if done % 10 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            log(f"Checkpoint: {done} events saved")

    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    n_ok = sum(1 for v in results.values() if 'error' not in v)
    n_err = sum(1 for v in results.values() if 'error' in v)
    strategies = {}
    for v in results.values():
        s = v.get('strategy', 'error')
        strategies[s] = strategies.get(s, 0) + 1

    print(f"\n\n{'='*60}")
    print(f"DONE: {n_ok} processed, {n_err} errors out of {len(results)}")
    print(f"Strategies: {strategies}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
