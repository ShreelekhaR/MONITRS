"""
MONITRS v2 — Production pipeline
Processes FEMA disaster events: scrapes articles, extracts locations and captions,
determines image center (FIRMS for fires, LLM for everything else),
and outputs data for image download + QA generation.

Strategy (validated on 30 events):
  - Fire events: FIRMS hotspot center if available, else LLM estimate
  - All other events: LLM coordinate estimate, FEMA fallback

Usage:
    export GCP_PROJECT_ID=your-project-id
    export EE_PROJECT_ID=your-ee-project-id  # optional, defaults to GCP_PROJECT_ID
    export FIRMS_MAP_KEY=your-firms-key
    python MONITRS/get_article_aggregate_locations.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import os
import sys
import datetime
import json
from os.path import join, isfile, isdir
from os import mkdir, makedirs
import numpy as np
import urllib.request
from dateutil.relativedelta import relativedelta
import ee
from google import genai
from PIL import Image
from time import sleep, time as now

# --- Config ---
PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-project-id')
EE_PROJECT_ID = os.environ.get('EE_PROJECT_ID', PROJECT_ID)
LOCATION = os.environ.get('GCP_LOCATION', 'us-central1')
FIRMS_MAP_KEY = os.environ.get('FIRMS_MAP_KEY', '')

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
MODEL = "gemini-2.5-flash-lite"

ee.Initialize(project=EE_PROJECT_ID)

BLACK_LIST = ['google', 'wikipedia', 'youtube', 'twitter', 'facebook', 'instagram',
              'linkedin', 'pinterest', 'reddit', 'quora', 'tiktok', 'tumblr']

ODIR = 'Data/images'
RESULTS_FILE = 'Data/pipeline_results.json'


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


def scrape_articles(links):
    content = ''
    scraped = 0
    for link in links:
        if any(b in link for b in BLACK_LIST):
            continue
        title, article_content = get_article_content(link)
        if article_content:
            content += article_content + '\n'
            scraped += 1
    return content, scraped


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
                log(f"LLM call failed after {retries + 1} attempts: {e}")
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

    You are given news articles about a {event_type} in {county}, {state}.
    The FEMA-reported center is approximately ({fema_center[0]:.4f}, {fema_center[1]:.4f}).

    Based on the articles, estimate:
    1. The latitude and longitude of the CENTER of the actual damage/impact area
    2. The approximate RADIUS in kilometers of the affected area
    3. Key landmarks or features at the center of damage

    Use your geographic knowledge to pinpoint where the damage was concentrated.

    Return as JSON:
    {{"lat": 40.12, "lon": -100.45, "radius_km": 15, "reasoning": "Fire centered along Republican River south of Cambridge"}}

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


def generate_captions(text, dates, event_type, county, state):
    prompt = f"""
    Task: Write a factual caption for each satellite image date of a {event_type} event in {county}, {state}.

    Each caption should:
    1. State specific facts from the articles — cite numbers (acres, homes, %) when available
    2. Describe the VISUAL change from the previous image
    3. Build a progressive narrative: pre-event baseline → onset → peak → aftermath
    4. Be 1-2 sentences, factual and concrete

    RULES:
    - Use present tense ("burn scar covers", not "would show")
    - Never say "satellite imagery would depict" or "would likely show"
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

    log(f"FIRMS: {len(unique)} hotspots")
    return unique


def firms_center(hotspots):
    h_lats = [h['lat'] for h in hotspots]
    h_lons = [h['lon'] for h in hotspots]
    return (np.mean(h_lats), np.mean(h_lons))


# --- Image download ---

def download_images(center, start_date, end_date, event_idx,
                    halfwidth=0.05, pre_days=14, post_days=14, max_cloud_pct=30):
    region = ee.Geometry.Rectangle([
        [center[1] - halfwidth, center[0] - halfwidth],
        [center[1] + halfwidth, center[0] + halfwidth],
    ])

    event_start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_clean = end_date[:10] if len(end_date) > 10 else end_date
    event_end = datetime.datetime.strptime(end_clean, '%Y-%m-%d')

    pre_start = (event_start - relativedelta(days=pre_days)).strftime('%Y-%m-%d')
    post_end = (event_end + relativedelta(days=post_days)).strftime('%Y-%m-%d')

    base_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region)

    phases = {
        'pre': (pre_start, start_date),
        'during': (start_date, end_clean),
        'post': (end_clean, post_end),
    }

    makedirs(join(ODIR, str(event_idx)), exist_ok=True)
    outdir = join(ODIR, str(event_idx))
    all_dates = {}

    for phase, (d_start, d_end) in phases.items():
        col = base_col.filterDate(d_start, d_end).filter(
            ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
        num = col.size().getInfo()

        if num == 0:
            all_dates[phase] = []
            continue

        img_list = col.sort('system:time_start').toList(num)
        seen_dates = set()
        phase_dates = []

        for i in range(num):
            img_date = ee.Image(img_list.get(i)).date().format('YYYY-MM-dd').getInfo()
            if img_date in seen_dates:
                continue
            seen_dates.add(img_date)

            output_file = join(outdir, f'{event_idx}_{phase}_{img_date}.png')
            if isfile(output_file):
                phase_dates.append(img_date)
                continue

            day_end = (datetime.datetime.strptime(img_date, '%Y-%m-%d') + relativedelta(days=1)).strftime('%Y-%m-%d')
            mosaic = col.filterDate(img_date, day_end).mosaic()

            try:
                url = mosaic.getThumbURL({
                    'bands': ['B4', 'B3', 'B2'],
                    'min': 0, 'max': 3000, 'gamma': 1,
                    'dimensions': '512x512',
                    'region': region,
                })
                urllib.request.urlretrieve(url, output_file)
                img_array = np.array(Image.open(output_file))
                mean_val = np.mean(img_array)
                black_pct = np.count_nonzero(img_array.sum(axis=2) == 0) / (img_array.shape[0] * img_array.shape[1])
                white_pct = np.count_nonzero(img_array.min(axis=2) > 240) / (img_array.shape[0] * img_array.shape[1])
                if mean_val < 25 or mean_val > 240 or black_pct > 0.05 or white_pct > 0.5:
                    os.remove(output_file)
                    continue
                phase_dates.append(img_date)
            except Exception:
                continue

        all_dates[phase] = phase_dates
        log(f"{phase}: {len(phase_dates)} images")

    return all_dates


# --- Main pipeline ---

def process_event(event_idx, links, df, results, results_file):
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

    print(f"\n{'='*70}")
    print(f"EVENT {event_idx}: {row['declarationTitle']}")
    print(f"  Type: {event_type} | {state} | {county} | {start_date} to {end_date}")

    # 1. Scrape articles
    log("Scraping articles")
    content, scraped = scrape_articles(links)
    log(f"Scraped {scraped}/{len(links)} articles, {len(content)} chars")

    if not content:
        results[str(event_idx)] = {'error': 'no_content', 'event': row['declarationTitle']}
        return

    # 2. Extract location-event pairs
    log("Extracting location-event pairs")
    loc_events = extract_location_events(content, start_date, end_date, state, county)
    visual_events = [le for le in loc_events if le.get('type') == 'visual']
    contextual_events = [le for le in loc_events if le.get('type') != 'visual']
    log(f"{len(visual_events)} visual + {len(contextual_events)} contextual events")

    # 3. Determine image center
    log("Determining image center")

    # Strategy: FIRMS for fires, LLM for everything else, FEMA fallback
    center = fema_center
    strategy = 'fema'
    halfwidth = 0.05
    hotspots = None
    llm_estimate = None
    fc = None

    if event_type == 'Fire' and FIRMS_MAP_KEY:
        log("Querying FIRMS")
        hotspots = get_firms_hotspots(fema_center, start_date, end_date)
        if hotspots:
            fc = firms_center(hotspots)
            center = fc
            strategy = 'firms'
            log(f"FIRMS center: ({center[0]:.4f}, {center[1]:.4f}), {len(hotspots)} hotspots")

    if strategy != 'firms':
        log("LLM coordinate estimation")
        llm_estimate = estimate_damage_center(content, event_type, state, county, fema_center)
        if llm_estimate and llm_estimate.get('lat') and llm_estimate.get('lon'):
            center = (llm_estimate['lat'], llm_estimate['lon'])
            radius = llm_estimate.get('radius_km', 10)
            halfwidth = min(max(0.05, radius / 111 / 2), 0.15)
            strategy = 'llm'
            log(f"LLM center: ({center[0]:.4f}, {center[1]:.4f}), radius ~{radius}km")
        else:
            log(f"LLM failed, using FEMA center: ({center[0]:.4f}, {center[1]:.4f})")

    # 4. Generate captions
    log("Generating captions")
    all_dates = pd.date_range(start_date, end_date, freq='5D').strftime('%Y-%m-%d').tolist()
    if end_date not in all_dates:
        all_dates.append(end_date)
    captions = generate_captions(content, all_dates, event_type, county, state)

    # 5. Download images
    log(f"Downloading images ({strategy} center)")
    image_dates = download_images(center, start_date, end_date, event_idx, halfwidth=halfwidth)

    # 6. Save results
    results[str(event_idx)] = {
        'event': row['declarationTitle'],
        'type': event_type,
        'state': state,
        'county': county,
        'start_date': start_date,
        'end_date': end_date,
        'fema_center': list(fema_center),
        'center': list(center),
        'strategy': strategy,
        'halfwidth': halfwidth,
        'llm_estimate': llm_estimate,
        'firms_center': list(fc) if fc else None,
        'firms_hotspots_count': len(hotspots) if hotspots else 0,
        'location_events': loc_events,
        'captions': captions,
        'image_dates': image_dates,
        'num_articles_scraped': scraped,
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Event {event_idx} done — {strategy} center, {sum(len(v) for v in image_dates.values())} images")


def main():
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

    # Load existing results for resume
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} events already done")
    else:
        results = {}

    total = len(events)
    for i, (event_idx, links) in enumerate(sorted(events.items())):
        if str(event_idx) in results:
            continue

        print(f"\n[{i+1}/{total}]", end='')

        try:
            process_event(event_idx, links, df, results, RESULTS_FILE)
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            sys.exit(0)
        except Exception as e:
            log(f"FAILED: {e}")
            results[str(event_idx)] = {'error': str(e)}
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    print(f"\n\nDone. {len(results)} events processed. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
