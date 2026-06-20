"""
MONITRS v2 — Gap filler for failed events
Retries failed events using:
1. Re-scrape the original article URLs (some may have been temporarily down)
2. DuckDuckGo search for new articles if scraping still fails

Usage:
    pip install duckduckgo-search
    export GCP_PROJECT_ID=your-project-id
    export FIRMS_MAP_KEY=your-firms-key
    python MONITRS/fill_gaps.py
"""

import os
import sys
import json
import datetime
import pandas as pd
import numpy as np
from time import sleep
from dateutil.relativedelta import relativedelta
from google import genai
from duckduckgo_search import DDGS

sys.path.insert(0, os.path.dirname(__file__))
from run_language_pipeline import (
    get_article_content, extract_location_events, estimate_damage_center,
    generate_aligned_captions, align_events_to_images,
    get_firms_hotspots, llm_call, log, BLACK_LIST,
)

PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-project-id')
LOCATION = os.environ.get('GCP_LOCATION', 'us-central1')
FIRMS_MAP_KEY = os.environ.get('FIRMS_MAP_KEY', '')

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
MODEL = "gemini-2.5-flash-lite"

RESULTS_FILE = 'Data/events_processed.json'


def search_ddg(event_name, event_type, county, state, start_date, max_results=5):
    queries = [
        f"{event_name} {county} {state} {start_date}",
        f"{event_name} {state} {event_type}",
        f"{event_name} disaster",
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
        except Exception as e:
            log(f"DDG search failed for '{query}': {e}")
            sleep(2)
    return all_links


def scrape_articles(links):
    content = ''
    scraped = 0
    for link in links:
        if any(b in link for b in BLACK_LIST):
            continue
        title, article_content = get_article_content(link)
        if article_content and len(article_content) > 100:
            content += article_content + '\n'
            scraped += 1
    return content, scraped


def process_gap_event(event_idx, original_links, df):
    row = df[df['index'] == event_idx].iloc[0]
    fema_lat, fema_lon = row['lat'], row['lon']
    fema_center = (fema_lat, fema_lon)
    state = row['state']
    county = row['designatedArea']
    event_type = row['incidentType']
    event_name = row['declarationTitle']
    start_date = row['incidentBeginDate']
    end_date = str(row['incidentEndDate'])
    if len(end_date) > 10:
        end_date = end_date[:10]

    print(f"\n{'='*60}")
    print(f"GAP FILL {event_idx}: {event_name}")
    print(f"  {event_type} | {state} | {county} | {start_date} to {end_date}")

    # Step 1: Try re-scraping original URLs
    log("Re-scraping original articles")
    content, scraped = scrape_articles(original_links)
    log(f"Re-scraped {scraped}/{len(original_links)} articles, {len(content)} chars")

    # Step 2: If still no content, search DuckDuckGo
    if len(content) < 200:
        log("Searching DuckDuckGo for articles")
        new_links = search_ddg(event_name, event_type, county, state, start_date)
        log(f"DDG found {len(new_links)} links")
        sleep(1)

        for link in new_links:
            if any(b in link for b in BLACK_LIST):
                continue
            title, article_content = get_article_content(link)
            if article_content and len(article_content) > 100:
                content += article_content + '\n'
                scraped += 1
                log(f"  [ok] {link[:70]}...")

        log(f"Total after DDG: {scraped} articles, {len(content)} chars")

    if len(content) < 100:
        log("Still no content — skipping")
        return {'error': 'no_content_after_retry', 'event': event_name}

    # Step 3: Extract location-events
    log("Extracting location-event pairs")
    loc_events = extract_location_events(content, start_date, end_date, state, county)
    visual = [le for le in loc_events if le.get('type') == 'visual']
    contextual = [le for le in loc_events if le.get('type') != 'visual']
    log(f"{len(visual)} visual + {len(contextual)} contextual")

    # Step 4: Determine center
    center = list(fema_center)
    strategy = 'fema'
    halfwidth = 0.05
    llm_estimate = None
    llm_center = None
    firms_data = None

    log("LLM coordinate estimation")
    llm_estimate = estimate_damage_center(content, event_type, state, county, fema_center)
    if llm_estimate and llm_estimate.get('lat') and llm_estimate.get('lon'):
        llm_center = [llm_estimate['lat'], llm_estimate['lon']]
        radius = llm_estimate.get('radius_km', 10)
        halfwidth = min(max(0.05, radius / 111 / 2), 0.15)
        center = llm_center
        strategy = 'llm'
        log(f"LLM center: ({center[0]:.4f}, {center[1]:.4f})")

    if event_type == 'Fire' and FIRMS_MAP_KEY:
        log("Querying FIRMS")
        hotspots = get_firms_hotspots(fema_center, start_date, end_date)
        if hotspots:
            h_lats = [h['lat'] for h in hotspots]
            h_lons = [h['lon'] for h in hotspots]
            firms_c = [float(np.mean(h_lats)), float(np.mean(h_lons))]
            firms_data = {'count': len(hotspots), 'center': firms_c}
            center = firms_c
            strategy = 'firms'
            halfwidth = 0.05
            log(f"FIRMS center: ({center[0]:.4f}, {center[1]:.4f}), {len(hotspots)} hotspots")

    # Step 5: Generate captions
    log("Generating captions")
    caption_dates = pd.date_range(start_date, end_date, freq='5D').strftime('%Y-%m-%d').tolist()
    if end_date not in caption_dates:
        caption_dates.append(end_date)
    aligned = align_events_to_images(loc_events, caption_dates, start_date)
    captions = generate_aligned_captions(
        content, aligned, event_type, county, state, center, strategy)

    log(f"Done — {strategy} center")

    return {
        'event': event_name,
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
        'num_articles_scraped': scraped,
        'gap_filled': True,
    }


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"No results file: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

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

    # Find failed events
    failed = {k: v for k, v in results.items() if 'error' in v}
    print(f"Found {len(failed)} failed events to retry")

    fixed = 0
    still_failed = 0

    for event_idx_str, error_data in sorted(failed.items(), key=lambda x: int(x[0])):
        event_idx = int(event_idx_str)
        links = events.get(event_idx, [])

        try:
            result = process_gap_event(event_idx, links, df)
            results[event_idx_str] = result
            if 'error' not in result:
                fixed += 1
            else:
                still_failed += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            break
        except Exception as e:
            log(f"FAILED again: {e}")
            still_failed += 1

        # Save periodically
        if (fixed + still_failed) % 10 == 0:
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\n{'='*60}")
    print(f"Gap fill complete: {fixed} fixed, {still_failed} still failed out of {len(failed)}")


if __name__ == '__main__':
    main()
