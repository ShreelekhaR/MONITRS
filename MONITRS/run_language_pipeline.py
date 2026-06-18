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
from dateutil.relativedelta import relativedelta
from google import genai
from time import sleep

# --- Config ---
PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-project-id')
LOCATION = os.environ.get('GCP_LOCATION', 'us-central1')
FIRMS_MAP_KEY = os.environ.get('FIRMS_MAP_KEY', '')

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
MODEL = "gemini-2.5-flash-lite"

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


def generate_captions(text, dates, event_type, county, state):
    prompt = f"""
    Task: Write a factual caption for each satellite image date of a {event_type} event in {county}, {state}.

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


# --- Process one event ---

def process_event(event_idx, links, df):
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

    print(f"\n{'='*60}")
    print(f"EVENT {event_idx}: {row['declarationTitle']}")
    print(f"  {event_type} | {state} | {county} | {start_date} to {end_date}")

    # 1. Scrape
    log("Scraping articles")
    content, scraped = scrape_articles(links)
    log(f"Scraped {scraped}/{len(links)} articles, {len(content)} chars")

    if not content:
        return {'error': 'no_content', 'event': row['declarationTitle']}

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
    firms_data = None

    if event_type == 'Fire' and FIRMS_MAP_KEY:
        log("Querying FIRMS")
        hotspots = get_firms_hotspots(fema_center, start_date, end_date)
        if hotspots:
            h_lats = [h['lat'] for h in hotspots]
            h_lons = [h['lon'] for h in hotspots]
            center = [float(np.mean(h_lats)), float(np.mean(h_lons))]
            strategy = 'firms'
            firms_data = {'count': len(hotspots), 'center': center}
            log(f"FIRMS center: ({center[0]:.4f}, {center[1]:.4f}), {len(hotspots)} hotspots")

    if strategy != 'firms':
        log("LLM coordinate estimation")
        llm_estimate = estimate_damage_center(content, event_type, state, county, fema_center)
        if llm_estimate and llm_estimate.get('lat') and llm_estimate.get('lon'):
            center = [llm_estimate['lat'], llm_estimate['lon']]
            radius = llm_estimate.get('radius_km', 10)
            halfwidth = min(max(0.05, radius / 111 / 2), 0.15)
            strategy = 'llm'
            log(f"LLM center: ({center[0]:.4f}, {center[1]:.4f}), radius ~{radius}km")
        else:
            log("LLM failed, using FEMA center")

    # 4. Generate captions
    log("Generating captions")
    all_dates = pd.date_range(start_date, end_date, freq='5D').strftime('%Y-%m-%d').tolist()
    if end_date not in all_dates:
        all_dates.append(end_date)
    captions = generate_captions(content, all_dates, event_type, county, state)

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
        'llm_estimate': llm_estimate,
        'firms': firms_data,
        'location_events': loc_events,
        'captions': captions,
        'num_articles_scraped': scraped,
        'article_links': links,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start event index')
    parser.add_argument('--end', type=int, default=None, help='End event index (exclusive)')
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
            result = process_event(event_idx, events[event_idx], df)
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
