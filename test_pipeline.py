"""
Test the MONITRS v2 pipeline on events 0-4.
Scrapes articles, extracts locations + timelines via Vertex AI, geocodes them,
and downloads Sentinel-2 imagery for visual verification.

Usage:
    export GCP_PROJECT_ID=your-project-id
    export GEOCODE_API_KEY=your-geocode-key
    python test_pipeline.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import datetime
from os.path import join, isfile, isdir
from os import mkdir
import numpy as np
import urllib.request
from dateutil.relativedelta import relativedelta
import ee
from google import genai
from PIL import Image
from tqdm import tqdm
from time import sleep, time as now
import json
import sys


def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"  [{ts}] {msg}")
    sys.stdout.flush()

PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-project-id')
EE_PROJECT_ID = os.environ.get('EE_PROJECT_ID', PROJECT_ID)
LOCATION = os.environ.get('GCP_LOCATION', 'us-central1')
GEOCODE_API_KEY = os.environ.get('GEOCODE_API_KEY', 'your-key-here')

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)
MODEL = "gemini-2.5-flash-lite"

ee.Initialize(project=EE_PROJECT_ID)

ODIR = 'Data/images'

BLACK_LIST = ['google', 'wikipedia', 'youtube', 'twitter', 'facebook', 'instagram',
              'linkedin', 'pinterest', 'reddit', 'quora', 'tiktok', 'tumblr']

TEST_EVENT_IDS = [
    # 10 Fires
    6, 41, 50, 54, 130, 226, 280, 328, 3487, 7952,
    # 5 Hurricanes
    1661, 2899, 3362, 5223, 7447,
    # 5 Severe Storms
    8497, 8510, 9257, 9487, 10095,
    # 5 Floods
    2359, 2488, 2656, 2751, 2785,
    # 5 Tornados
    528, 1898, 4070, 4102, 4501,
]


def get_article_content(url):
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else 'No title found'
            paragraphs = soup.find_all('p')
            content = ' '.join([para.get_text() for para in paragraphs])
            return title, content
    except Exception as e:
        print(f"  [!] Failed to fetch {url}: {e}")
    return None, None


def extract_locations_and_events(text, start_date, end_date, state, county):
    prompt = f"""
    Task: Extract locations where a natural disaster caused VISIBLE PHYSICAL CHANGES detectable from satellite imagery.

    Event: Natural disaster in {county}, {state}, from {start_date} to {end_date}.

    For each location, extract:
    - location: the place name (city, road, river, landmark)
    - date: YYYY-MM-DD
    - event: what PHYSICALLY CHANGED at this location (1 short sentence)

    For each event, also classify it as:
    - "visual": physically visible from satellite (fire, burn scar, flooding, debris, structural damage)
    - "contextual": impacts on people/infrastructure not directly visible (evacuations, road closures, shelters, casualties, power outages)

    Both types are valuable. Include ALL disaster-related events at specific locations.

    Only locations within or near {county}, {state}. No state/country names.
    No political statements, press conferences, or general commentary.

    Return as JSON array:
    [{{"location": "Glen Rose", "date": "2022-07-18", "event": "Fire burned 6,700 acres of brush and timber", "type": "visual"}},
     {{"location": "Highway 89", "date": "2022-04-19", "event": "Highway 89 closed in both directions due to fire", "type": "contextual"}},
     {{"location": "Sinagua Middle School", "date": "2022-04-19", "event": "Evacuation shelter set up for 600 displaced residents", "type": "contextual"}}]

    If no events found, return: []

    Article Content: {text}
    """
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        raw = response.text.strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception as e:
        print(f"  [!] Gemini location-event extraction failed: {e}")
        sleep(5)
        return []


def generate_captions(text, dates, event_type, county, state, locations_inside):
    locs_str = ', '.join(locations_inside) if locations_inside else county
    prompt = f"""
    Task: Write a factual caption for each satellite image date of a {event_type} event in {county}, {state}.

    You are captioning a sequence of satellite images. Each caption should:
    1. State specific facts from the articles — cite numbers (acres burned, homes destroyed, flood extent) when available
    2. Describe the VISUAL change from the previous image — what grew, spread, receded, or appeared
    3. Build a progressive narrative — pre-event baseline → onset → peak → aftermath
    4. Be 1-2 sentences, factual and concrete

    RULES:
    - Use present tense ("burn scar covers", not "would show")
    - Never say "satellite imagery would depict" or "would likely show"
    - Pull specific details from the articles: acreage, road names, percentages, structure counts
    - If a date is before the event started, describe the baseline landscape

    Good examples:
    - "Pre-event: dry grassland and agricultural fields across the county. No fire activity."
    - "Wildfire has spread across tens of thousands of acres. Active fire fronts visible along Highway 6."
    - "Burn scar covering approximately 54,000 acres. 16 homes destroyed near Glen Rose."
    - "Fire 90% contained. Burn scar clearly delineated, scattered hotspots remain in timber areas."
    - "Floodwaters cover low-lying areas along the Republican River. Standing water visible in agricultural fields."

    Known locations in the image area: {locs_str}
    Image dates (in order): {dates}

    Article content: {text}

    Return format — one line per date, with the caption grounded in article facts:
    YYYY-MM-DD: caption
    """
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"  [!] Gemini caption generation failed: {e}")
        sleep(5)
        return ''


FIRMS_MAP_KEY = os.environ.get('FIRMS_MAP_KEY', '')


def get_firms_hotspots(fema_center, start_date, end_date, radius_km=100):
    if not FIRMS_MAP_KEY:
        print("  [!] No FIRMS_MAP_KEY set, skipping FIRMS verification")
        return None
    lat, lon = fema_center
    end_clean = end_date[:10] if len(end_date) > 10 else end_date
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_clean, '%Y-%m-%d')
    num_days = (end_dt - start_dt).days + 1

    # FIRMS area format: west,south,east,north (~2 degrees = ~220km box)
    west, south = lon - 2, lat - 2
    east, north = lon + 2, lat + 2
    area = f"{west},{south},{east},{north}"

    # Query ALL sources and merge for best coverage
    sources = ['VIIRS_SNPP_SP', 'MODIS_SP', 'VIIRS_NOAA20_NRT']
    all_hotspots = []

    for source in sources:
        source_count = 0
        chunk_start = start_dt
        chunk_num = 0
        while chunk_start < end_dt:
            chunk_days = min(5, (end_dt - chunk_start).days + 1)
            date_str = chunk_start.strftime('%Y-%m-%d')
            chunk_num += 1
            log(f"FIRMS {source} chunk {chunk_num}: {date_str} +{chunk_days}d")
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                   f"{FIRMS_MAP_KEY}/{source}/{area}/{chunk_days}/{date_str}")
            try:
                t0 = now()
                resp = requests.get(url, timeout=15)
                log(f"  -> {resp.status_code} in {now()-t0:.1f}s, {len(resp.text)} bytes")
                if resp.status_code == 200 and len(resp.text.strip().split('\n')) > 1:
                    lines = resp.text.strip().split('\n')
                    header = lines[0].split(',')
                    lat_idx = header.index('latitude')
                    lon_idx = header.index('longitude')
                    date_idx = header.index('acq_date')
                    conf_idx = header.index('confidence') if 'confidence' in header else None

                    for line in lines[1:]:
                        parts = line.split(',')
                        h_lat, h_lon = float(parts[lat_idx]), float(parts[lon_idx])
                        dist = ((h_lat - lat)**2 + (h_lon - lon)**2)**0.5 * 111
                        if dist <= radius_km:
                            source_count += 1
                            all_hotspots.append({
                                'lat': h_lat, 'lon': h_lon,
                                'date': parts[date_idx],
                                'confidence': parts[conf_idx] if conf_idx else 'n/a',
                                'source': source,
                            })
            except Exception:
                pass
            chunk_start += relativedelta(days=chunk_days)
        print(f"    {source}: {source_count} hotspots")

    if not all_hotspots:
        print("  [!] No FIRMS hotspots found across any source")
        return None

    # Deduplicate by lat/lon/date
    seen = set()
    unique = []
    for h in all_hotspots:
        key = (round(h['lat'], 3), round(h['lon'], 3), h['date'])
        if key not in seen:
            seen.add(key)
            unique.append(h)

    print(f"  FIRMS: {len(unique)} unique hotspots within {radius_km}km ({all_hotspots[0]['source']})")
    return unique


def evaluate_centers_vs_firms(hotspots, fema_center, geocoded_center, llm_center, halfwidth=0.05):
    if not hotspots:
        return None

    h_lats = [h['lat'] for h in hotspots]
    h_lons = [h['lon'] for h in hotspots]
    firms_center = (np.mean(h_lats), np.mean(h_lons))

    results = {}
    for name, center in [('fema', fema_center), ('geocoded', geocoded_center), ('llm', llm_center)]:
        dist_to_firms = ((center[0] - firms_center[0])**2 + (center[1] - firms_center[1])**2)**0.5 * 111
        hotspots_in_bbox = sum(1 for h in hotspots
                               if abs(h['lat'] - center[0]) <= halfwidth
                               and abs(h['lon'] - center[1]) <= halfwidth)
        results[name] = {
            'dist_to_firms_center_km': round(dist_to_firms, 1),
            'hotspots_in_bbox': hotspots_in_bbox,
            'hotspots_total': len(hotspots),
            'pct_captured': round(100 * hotspots_in_bbox / len(hotspots), 1),
        }

    print(f"\n  FIRMS ground truth center: ({firms_center[0]:.4f}, {firms_center[1]:.4f})")
    print(f"  {'Strategy':<12} {'Dist to FIRMS':>14} {'Hotspots captured':>18}")
    print(f"  {'-'*46}")
    for name, r in results.items():
        print(f"  {name:<12} {r['dist_to_firms_center_km']:>11.1f} km   "
              f"{r['hotspots_in_bbox']:>4}/{r['hotspots_total']} ({r['pct_captured']}%)")

    return {'firms_center': firms_center, 'scores': results}


def estimate_damage_center(text, event_type, state, county, fema_center):
    prompt = f"""
    Task: Estimate the geographic center and extent of the damage area for this {event_type} event.

    You are given news articles about a {event_type} in {county}, {state}.
    The FEMA-reported center is approximately ({fema_center[0]:.4f}, {fema_center[1]:.4f}).

    Based on the articles, estimate:
    1. The latitude and longitude of the CENTER of the actual damage/impact area
    2. The approximate RADIUS in kilometers of the affected area
    3. Key landmarks or features at the center of damage

    Use your geographic knowledge of roads, rivers, towns mentioned in the articles
    to pinpoint where the damage was concentrated — NOT just the county centroid.

    Return as JSON:
    {{"lat": 40.12, "lon": -100.45, "radius_km": 15, "reasoning": "Fire centered along Republican River south of Cambridge"}}

    Article Content: {text}
    """
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        raw = response.text.strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        print(f"  [!] LLM coordinate estimation failed: {e}")
        sleep(5)
        return None


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
    'AS': 'American Samoa', 'GU': 'Guam', 'MP': 'Northern Mariana Islands',
}


def geocode_location(loc_name, state_hint='', county_hint=''):
    query = loc_name
    if state_hint:
        state_full = STATE_NAMES.get(state_hint, state_hint)
        query = f"{loc_name}, {state_full}"

    try:
        link = f'https://geocode.maps.co/search?q={query}&api_key={GEOCODE_API_KEY}'
        response = requests.get(link, timeout=10)
        data = response.json()
        if not data:
            return None, None

        state_full = STATE_NAMES.get(state_hint, state_hint)

        # Check each result for one that's actually in the right state
        for result in data:
            display = result.get('display_name', '')
            if state_full.lower() in display.lower() or state_hint.lower() in display.lower():
                return float(result['lat']), float(result['lon'])

        # If no state match, check if the top result is at least in the US
        display = data[0].get('display_name', '')
        if 'United States' in display or 'Puerto Rico' in display:
            return float(data[0]['lat']), float(data[0]['lon'])

        # Top result is in wrong country — reject
        print(f"    [geocode] Rejected '{loc_name}': {display}")
        return None, None
    except Exception:
        pass
    return None, None


def find_optimal_center(locs_with_coords, fema_center, halfwidth=0.05):
    lats = [c[0] for c in locs_with_coords.values()] + [fema_center[0]]
    lons = [c[1] for c in locs_with_coords.values()] + [fema_center[1]]

    best_count = 0
    best_center = fema_center

    for lat in lats:
        for lon in lons:
            count = sum(1 for la, lo in zip(lats, lons)
                        if abs(la - lat) <= halfwidth and abs(lo - lon) <= halfwidth)
            fema_in = (abs(lat - fema_center[0]) <= halfwidth and
                       abs(lon - fema_center[1]) <= halfwidth)
            if count > best_count and fema_in:
                best_count = count
                best_center = (lat, lon)

    return best_center


def compute_change_indices(center, start_date, end_date, halfwidth=0.05, pre_days=30):
    region = ee.Geometry.Rectangle([
        [center[1] - halfwidth, center[0] - halfwidth],
        [center[1] + halfwidth, center[0] + halfwidth],
    ])

    event_start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_clean = end_date[:10] if len(end_date) > 10 else end_date
    event_end = datetime.datetime.strptime(end_clean, '%Y-%m-%d')
    pre_start = (event_start - relativedelta(days=pre_days)).strftime('%Y-%m-%d')

    base = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region)

    pre_col = base.filterDate(pre_start, start_date).filter(
        ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    post_col = base.filterDate(start_date, end_clean).filter(
        ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))

    pre_count = pre_col.size().getInfo()
    post_count = post_col.size().getInfo()
    log(f"Change detection: {pre_count} pre, {post_count} post scenes")

    if pre_count == 0 or post_count == 0:
        log("Not enough scenes for change detection")
        return None

    pre = pre_col.median()
    post = post_col.median()

    def calc_indices(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        nbr = img.normalizedDifference(['B8', 'B12']).rename('NBR')
        ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
        ndmi = img.normalizedDifference(['B8A', 'B11']).rename('NDMI')
        return img.addBands([ndvi, nbr, ndwi, ndmi])

    pre_idx = calc_indices(pre)
    post_idx = calc_indices(post)

    results = {}
    for idx_name in ['NDVI', 'NBR', 'NDWI', 'NDMI']:
        diff = post_idx.select(idx_name).subtract(pre_idx.select(idx_name))
        stats = diff.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
            geometry=region,
            scale=20,
            maxPixels=1e7,
        ).getInfo()

        mean_key = f'{idx_name}_mean'
        min_key = f'{idx_name}_min'
        max_key = f'{idx_name}_max'

        results[idx_name] = {
            'mean_change': stats.get(mean_key, 0),
            'min_change': stats.get(min_key, 0),
            'max_change': stats.get(max_key, 0),
        }

    # RGB mean absolute difference
    rgb_diff = post.select(['B4', 'B3', 'B2']).subtract(pre.select(['B4', 'B3', 'B2'])).abs()
    rgb_stats = rgb_diff.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=20,
        maxPixels=1e7,
    ).getInfo()
    results['RGB_diff'] = {
        'mean_B4': rgb_stats.get('B4', 0),
        'mean_B3': rgb_stats.get('B3', 0),
        'mean_B2': rgb_stats.get('B2', 0),
    }

    # Confidence score: max absolute change across all indices
    abs_changes = []
    for idx_name in ['NDVI', 'NBR', 'NDWI', 'NDMI']:
        m = results[idx_name]['mean_change']
        if m is not None:
            abs_changes.append(abs(m))
    max_change = max(abs_changes) if abs_changes else 0

    # Interpret
    signals = []
    ndvi_m = results['NDVI']['mean_change'] or 0
    nbr_m = results['NBR']['mean_change'] or 0
    ndwi_m = results['NDWI']['mean_change'] or 0
    ndmi_m = results['NDMI']['mean_change'] or 0

    if nbr_m < -0.1:
        signals.append(f"burn scar detected (dNBR={nbr_m:.3f})")
    if ndvi_m < -0.1:
        signals.append(f"vegetation loss (dNDVI={ndvi_m:.3f})")
    if ndwi_m > 0.1:
        signals.append(f"water increase / flooding (dNDWI={ndwi_m:.3f})")
    if ndmi_m < -0.1:
        signals.append(f"moisture decrease (dNDMI={ndmi_m:.3f})")

    if not signals:
        if max_change > 0.05:
            signals.append(f"minor change detected (max_delta={max_change:.3f})")
        else:
            signals.append("no significant change detected")

    results['confidence'] = max_change
    results['signals'] = signals

    log(f"Change detection results:")
    for idx_name in ['NDVI', 'NBR', 'NDWI', 'NDMI']:
        m = results[idx_name]['mean_change']
        print(f"    d{idx_name}: {m:.4f}" if m else f"    d{idx_name}: N/A")
    print(f"    Confidence: {max_change:.4f}")
    for s in signals:
        print(f"    >> {s}")

    return results


def download_images(center, start_date, end_date, event_idx, halfwidth=0.05,
                     pre_event_days=14, post_event_days=14, max_cloud_pct=30):
    region = ee.Geometry.Rectangle([
        [center[1] - halfwidth, center[0] - halfwidth],
        [center[1] + halfwidth, center[0] + halfwidth],
    ])

    event_start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_clean = end_date[:10] if len(end_date) > 10 else end_date
    event_end = datetime.datetime.strptime(end_clean, '%Y-%m-%d')

    pre_start = (event_start - relativedelta(days=pre_event_days)).strftime('%Y-%m-%d')
    post_end = (event_end + relativedelta(days=post_event_days)).strftime('%Y-%m-%d')

    log(f"EE query: pre={pre_start}, event={start_date} to {end_clean}, post to {post_end}")
    log(f"  center=({center[0]:.4f},{center[1]:.4f}), hw={halfwidth}")
    base_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region)

    # Pre-event
    log("EE: counting pre-event scenes...")
    pre_col_all = base_col.filterDate(pre_start, start_date)
    pre_col = pre_col_all.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
    num_pre_all = pre_col_all.size().getInfo()
    num_pre = pre_col.size().getInfo()
    log(f"Pre-event: {num_pre}/{num_pre_all} scenes under {max_cloud_pct}% cloud")

    # During event (FEMA start to FEMA end)
    log("EE: counting during-event scenes...")
    during_col_all = base_col.filterDate(start_date, end_clean)
    during_col = during_col_all.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
    num_during_all = during_col_all.size().getInfo()
    num_during = during_col.size().getInfo()
    log(f"During: {num_during}/{num_during_all} scenes under {max_cloud_pct}% cloud")

    # Post-event (FEMA end + buffer)
    log("EE: counting post-event scenes...")
    post_col_all = base_col.filterDate(end_clean, post_end)
    post_col = post_col_all.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
    num_post_all = post_col_all.size().getInfo()
    num_post = post_col.size().getInfo()
    log(f"Post-event: {num_post}/{num_post_all} scenes under {max_cloud_pct}% cloud")

    if num_pre + num_during + num_post == 0:
        print("  No images found")
        return {'pre': [], 'during': [], 'post': []}

    if not isdir(ODIR):
        mkdir(ODIR)
    outdir = join(ODIR, str(event_idx))
    if not isdir(outdir):
        mkdir(outdir)

    def _get_unique_dates(col, num, max_dates):
        if num == 0:
            return []
        log(f"EE: listing dates from {num} scenes (max {max_dates} unique)...")
        img_list = col.sort('system:time_start').toList(num)
        all_dates = []
        seen = set()
        for i in range(num):
            d = ee.Image(img_list.get(i)).date().format('YYYY-MM-dd').getInfo()
            if d not in seen:
                seen.add(d)
                all_dates.append(d)
                log(f"  date {len(all_dates)}/{max_dates}: {d}")
            if len(all_dates) >= max_dates:
                break
        return all_dates

    def _download_batch(col, num, label, max_dates=10):
        if num == 0:
            return []
        unique_dates = _get_unique_dates(col, num, max_dates)
        dates_downloaded = []
        for img_date in tqdm(unique_dates, desc=f"  {label}"):
            output_file = join(outdir, f'{event_idx}_{label}_{img_date}.png')
            if isfile(output_file):
                dates_downloaded.append(img_date)
                continue
            day_start = img_date
            day_end = (datetime.datetime.strptime(img_date, '%Y-%m-%d') + relativedelta(days=1)).strftime('%Y-%m-%d')
            mosaic = col.filterDate(day_start, day_end).mosaic()
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
                    reason = []
                    if mean_val < 25: reason.append("too dark")
                    if mean_val > 240: reason.append("too bright")
                    if black_pct > 0.05: reason.append(f"{black_pct:.0%} black/nodata")
                    if white_pct > 0.5: reason.append(f"{white_pct:.0%} white/cloud")
                    print(f"  [reject] {img_date}: {', '.join(reason)}")
                    os.remove(output_file)
                    continue
                dates_downloaded.append(img_date)
            except Exception as e:
                print(f"  [!] Error downloading {label} mosaic for {img_date}: {e}")
                continue
        return dates_downloaded

    pre_dates = _download_batch(pre_col, num_pre, 'pre', max_dates=999)
    during_dates = _download_batch(during_col, num_during, 'during', max_dates=999)
    post_dates = _download_batch(post_col, num_post, 'post', max_dates=999)

    print(f"  Downloaded {len(pre_dates)} pre + {len(during_dates)} during + {len(post_dates)} post to {outdir}/")
    return {'pre': pre_dates, 'during': during_dates, 'post': post_dates}


def _process_event(event_idx, links, df, results, results_file):
    row = df[df['index'] == event_idx].iloc[0]
    fema_lat, fema_lon = row['lat'], row['lon']
    fema_center = (fema_lat, fema_lon)
    state = row['state']
    county = row['designatedArea']

    start_date = row['incidentBeginDate']
    end_date = row['incidentEndDate']
    if len(str(end_date)) > 10:
        end_date = str(end_date)[:10]

    print(f"\n{'='*70}")
    print(f"EVENT {event_idx}: {row['declarationTitle']}")
    print(f"  Type: {row['incidentType']} | State: {state} | Area: {county}")
    print(f"  Dates: {start_date} to {end_date}")
    print(f"  FEMA Center: ({fema_lat:.4f}, {fema_lon:.4f})")
    print(f"  Articles: {len(links)}")

    # 1. Scrape articles
    log("Step 1: Scraping articles")
    content = ''
    scraped = 0
    for link in links:
        if any(b in link for b in BLACK_LIST):
            print(f"  [skip] {link} (blacklisted)")
            continue
        title, article_content = get_article_content(link)
        if article_content:
            content += article_content + '\n'
            scraped += 1
            print(f"  [ok]   {link[:80]}...")
        else:
            print(f"  [fail] {link[:80]}...")

    print(f"  Scraped {scraped}/{len(links)} articles, {len(content)} chars total")

    if not content:
        log("No content scraped, saving empty result")
        results[str(event_idx)] = {'event': row['declarationTitle'], 'error': 'no_content'}
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return

    # 2. Extract locations + events together (structured)
    log("Step 2: Extracting location-event pairs (Gemini)")
    loc_events = extract_locations_and_events(content, start_date, end_date, state, county)
    visual_events = [le for le in loc_events if le.get('type') == 'visual']
    contextual_events = [le for le in loc_events if le.get('type') != 'visual']
    print(f"  Extracted {len(loc_events)} events: {len(visual_events)} visual + {len(contextual_events)} contextual")
    for le in loc_events:
        tag = le.get('type', '?')[:3]
        print(f"    [{tag}] {le.get('date', '?')}: {le.get('location', '?')} — {le.get('event', '?')}")
    # Only use visual event locations for geolocation
    location_names = list(set(le.get('location', '') for le in visual_events if le.get('location')))

    # 3. Geocode with state/county hint
    log(f"Step 3: Geocoding (with {state}, {county} hint)")
    geocoded = {}
    for loc in location_names:
        lat, lon = geocode_location(loc, state_hint=state, county_hint=county)
        if lat is not None:
            dist_km = ((lat - fema_lat)**2 + (lon - fema_lon)**2)**0.5 * 111
            if dist_km < 500:
                geocoded[loc] = (lat, lon)
                print(f"  {loc}: ({lat:.4f}, {lon:.4f}) — {dist_km:.0f}km from FEMA center")
            else:
                print(f"  {loc}: ({lat:.4f}, {lon:.4f}) — {dist_km:.0f}km REJECTED (too far)")
            sleep(1.1)
        else:
            print(f"  {loc}: FAILED to geocode")

    # 4. LLM coordinate estimation (do this BEFORE bbox so we can filter locations)
    log("Step 4: LLM coordinate estimation")
    llm_estimate = estimate_damage_center(content, row['incidentType'], state, county, fema_center)
    if llm_estimate:
        llm_center = (llm_estimate['lat'], llm_estimate['lon'])
        llm_radius = llm_estimate.get('radius_km', 10)
        print(f"  LLM estimate: ({llm_center[0]:.4f}, {llm_center[1]:.4f}), radius ~{llm_radius}km")
        print(f"  Reasoning: {llm_estimate.get('reasoning', 'N/A')}")
        halfwidth_llm = max(0.05, llm_radius / 111 / 2)
    else:
        llm_center = fema_center
        halfwidth_llm = 0.05
        print("  LLM estimation failed, falling back to FEMA center")

    # 5. Compute bbox using geocoded locations filtered to LLM damage area
    log("Step 5: Computing bbox from LLM-filtered locations")
    llm_nearby = {}
    for loc, (lat, lon) in geocoded.items():
        dist_to_llm = ((lat - llm_center[0])**2 + (lon - llm_center[1])**2)**0.5 * 111
        if dist_to_llm < llm_radius * 2:
            llm_nearby[loc] = (lat, lon)
            print(f"  {loc}: {dist_to_llm:.0f}km from LLM center — KEPT")
        else:
            print(f"  {loc}: {dist_to_llm:.0f}km from LLM center — filtered out")

    if llm_nearby:
        center = find_optimal_center(llm_nearby, llm_center)
    else:
        center = llm_center
    print(f"\n  Bbox center (LLM-filtered): ({center[0]:.4f}, {center[1]:.4f})")
    print(f"  Image bbox: [{center[0]-0.05:.4f} to {center[0]+0.05:.4f}] lat, "
          f"[{center[1]-0.05:.4f} to {center[1]+0.05:.4f}] lon")

    inside = []
    outside = []
    for loc, (lat, lon) in geocoded.items():
        if abs(lat - center[0]) <= 0.05 and abs(lon - center[1]) <= 0.05:
            inside.append(loc)
        else:
            outside.append(loc)
    print(f"  Locations INSIDE bbox: {inside}")
    if outside:
        print(f"  Locations OUTSIDE bbox: {outside}")

    # 6. Filter location-events to inside-bbox only
    inside_events = [le for le in loc_events if le.get('location') in inside]
    print(f"\n  Location-events inside bbox: {len(inside_events)}/{len(loc_events)}")

    # 7. Generate observational captions
    log("Step 7: Generating captions (Gemini)")
    sample_dates = pd.date_range(start_date, end_date, periods=min(5, 10)).strftime('%Y-%m-%d').tolist()
    captions = generate_captions(content, sample_dates, row['incidentType'], county, state, inside)
    print(f"  Captions:\n{captions}")

    # 8. FIRMS for fire events — use as PRIMARY center if available
    firms_eval = None
    hotspots = None
    firms_center = None
    if row['incidentType'] == 'Fire':
        log("Step 8: FIRMS ground truth (fire only)")
        hotspots = get_firms_hotspots(fema_center, start_date, end_date)
        if hotspots:
            h_lats = [h['lat'] for h in hotspots]
            h_lons = [h['lon'] for h in hotspots]
            firms_center = (np.mean(h_lats), np.mean(h_lons))
            firms_eval = evaluate_centers_vs_firms(hotspots, fema_center, center, llm_center)
            if not isdir(ODIR):
                mkdir(ODIR)
            firms_dir = join(ODIR, str(event_idx) + '_firms')
            if not isdir(firms_dir):
                mkdir(firms_dir)
            firms_csv = join(firms_dir, 'hotspots.csv')
            with open(firms_csv, 'w') as fh:
                fh.write('latitude,longitude,date,confidence,source\n')
                for h in hotspots:
                    fh.write(f"{h['lat']},{h['lon']},{h['date']},{h['confidence']},{h['source']}\n")
            print(f"  Saved {len(hotspots)} hotspots to {firms_csv}")

    # 9. Change detection on each candidate center
    log("Step 9: Change detection verification")
    candidates = {'fema': (fema_center, 0.05)}
    if llm_center:
        candidates['llm'] = (llm_center, min(halfwidth_llm, 0.15))
    if center and center != fema_center:
        candidates['bbox'] = (center, 0.05)
    if firms_center:
        candidates['firms'] = (firms_center, 0.05)

    change_results = {}
    best_center = llm_center or fema_center
    best_strategy = 'llm' if llm_center else 'fema'
    best_halfwidth = min(halfwidth_llm, 0.15) if llm_center else 0.05
    best_confidence = -1

    for name, (ctr, hw) in candidates.items():
        if ctr is None:
            continue
        log(f"Change detection for {name} center ({ctr[0]:.4f}, {ctr[1]:.4f})")
        try:
            cr = compute_change_indices(ctr, start_date, end_date, halfwidth=hw)
            if cr:
                change_results[name] = cr
                conf = cr['confidence']
                print(f"    {name}: confidence={conf:.4f} — {', '.join(cr['signals'])}")
                if conf > best_confidence:
                    best_confidence = conf
                    best_center = ctr
                    best_strategy = name
                    best_halfwidth = hw
        except Exception as e:
            log(f"  Change detection failed for {name}: {e}")

    log(f"Best center: {best_strategy} ({best_center[0]:.4f}, {best_center[1]:.4f}), confidence={best_confidence:.4f}")

    # 10. Download images for ALL strategies
    log("Step 10a: Downloading FEMA center images")
    fema_dates = download_images(fema_center, start_date, end_date, f"{event_idx}_fema")

    llm_dates = None
    if llm_center:
        log("Step 10b: Downloading LLM center images")
        llm_dates = download_images(llm_center, start_date, end_date, f"{event_idx}_llm",
                                    halfwidth=min(halfwidth_llm, 0.15))

    bbox_dates = None
    if center and center != fema_center:
        log("Step 10c: Downloading geocoded bbox images")
        bbox_dates = download_images(center, start_date, end_date, f"{event_idx}_bbox")

    firms_dates = None
    if firms_center:
        log("Step 10d: Downloading FIRMS center images")
        firms_dates = download_images(firms_center, start_date, end_date, f"{event_idx}_firms")

    results[str(event_idx)] = {
        'event': row['declarationTitle'],
        'type': row['incidentType'],
        'state': state,
        'county': county,
        'fema_center': fema_center,
        'best_center': best_center,
        'best_strategy': best_strategy,
        'llm_center': llm_center,
        'firms_center': firms_center,
        'llm_estimate': llm_estimate,
        'location_events': loc_events,
        'locations_geocoded': {k: list(v) for k, v in geocoded.items()},
        'locations_inside_bbox': inside,
        'locations_outside_bbox': outside,
        'inside_events': inside_events,
        'captions': captions,
        'num_articles_scraped': scraped,
        'image_dates_fema': fema_dates,
        'image_dates_llm': llm_dates,
        'image_dates_bbox': bbox_dates,
        'image_dates_firms': firms_dates,
        'firms_eval': firms_eval,
        'firms_hotspots_count': len(hotspots) if hotspots else 0,
        'change_detection': change_results,
        'best_confidence': best_confidence,
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Event {event_idx} done — saved to {results_file}")


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

    test_events = {k: v for k, v in events.items() if k in TEST_EVENT_IDS}

    results_file = 'Data/test_pipeline_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {results_file}")
    else:
        results = {}

    for event_idx, links in sorted(test_events.items()):
        if str(event_idx) in results:
            print(f"\n{'='*70}")
            print(f"EVENT {event_idx}: SKIPPING (already completed)")
            continue

        try:
            _process_event(event_idx, links, df, results, results_file)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress saved.")
            sys.exit(0)
        except Exception as e:
            log(f"EVENT {event_idx} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[str(event_idx)] = {'error': str(e)}
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            continue

    print(f"\n\nAll results saved to {results_file}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for idx, r in results.items():
        if 'error' in r:
            print(f"  Event {idx}: ERROR — {r['error']}")
            continue
        n_locs = len(r.get('locations_geocoded', {}))
        n_in = len(r.get('locations_inside_bbox', []))
        n_out = len(r.get('locations_outside_bbox', []))
        best = r.get('best_strategy', '?')
        conf = r.get('best_confidence', 0)
        print(f"  Event {idx} ({r['event']}): {n_locs} locs, {n_in} inside | best={best} conf={conf:.4f}")


if __name__ == '__main__':
    main()
