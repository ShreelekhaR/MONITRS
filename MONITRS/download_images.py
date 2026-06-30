"""
MONITRS v2 — Image download
Downloads all Sentinel-2 imagery for each event across the full date span.
One query per event, downloads all clear images.

Usage:
    export EE_PROJECT_ID=your-ee-project-id
    python MONITRS/download_images.py

    # Process specific batch:
    python MONITRS/download_images.py --start 0 --end 1000
"""

import os
import sys
import json
import re
import datetime
import argparse
import numpy as np
import urllib.request
from os.path import join, isfile
from os import makedirs
from dateutil.relativedelta import relativedelta
import ee
from PIL import Image

EE_PROJECT_ID = os.environ.get('EE_PROJECT_ID', 'your-project-id')
ee.Initialize(project=EE_PROJECT_ID)

INPUT_FILE = 'Data/events_processed.json'
ODIR = 'Data/images'
PRE_DAYS = 14
POST_DAYS = 14
MAX_CLOUD_PCT = 50


def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"  [{ts}] {msg}")
    sys.stdout.flush()


def parse_caption_dates(caption_text):
    dates = []
    if not caption_text:
        return dates
    for line in caption_text.strip().split('\n'):
        match = re.match(r'(\d{4}-\d{2}-\d{2}):', line.strip())
        if match:
            dates.append(match.group(1))
    return sorted(set(dates))


def download_event_images(event_idx, event_data):
    center = event_data['center']
    halfwidth = event_data.get('halfwidth', 0.05)
    fema_start = event_data['start_date']
    fema_end = event_data['end_date'][:10]
    caption_dates = parse_caption_dates(event_data.get('captions', ''))

    # Union of FEMA + caption date range + buffers
    all_dates = [fema_start, fema_end] + caption_dates
    all_dts = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in all_dates]
    query_start = (min(all_dts) - relativedelta(days=PRE_DAYS)).strftime('%Y-%m-%d')
    query_end = (max(all_dts) + relativedelta(days=POST_DAYS)).strftime('%Y-%m-%d')

    region = ee.Geometry.Rectangle([
        [center[1] - halfwidth, center[0] - halfwidth],
        [center[1] + halfwidth, center[0] + halfwidth],
    ])

    # Single query for the full range
    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filterDate(query_start, query_end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PCT)) \
        .sort('system:time_start')

    num = col.size().getInfo()
    log(f"Found {num} scenes ({query_start} to {query_end})")

    if num == 0:
        return []

    outdir = join(ODIR, str(event_idx))
    makedirs(outdir, exist_ok=True)

    # Get all dates at once
    img_list = col.toList(num)
    all_img_dates = []
    for i in range(num):
        try:
            d = ee.Image(img_list.get(i)).date().format('YYYY-MM-dd').getInfo()
            all_img_dates.append(d)
        except Exception:
            continue

    # Deduplicate dates
    unique_dates = list(dict.fromkeys(all_img_dates))

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
                'bands': ['B4', 'B3', 'B2'],
                'min': 0, 'max': 3000, 'gamma': 1,
                'dimensions': '512x512',
                'region': region,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(INPUT_FILE):
        print(f"No input file: {INPUT_FILE}. Run run_language_pipeline.py first.")
        sys.exit(1)

    with open(INPUT_FILE) as f:
        events = json.load(f)

    progress_file = 'Data/download_progress.json'
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {}

    event_ids = sorted(events.keys(), key=int)
    if args.end:
        event_ids = [e for e in event_ids if args.start <= int(e) < args.end]
    else:
        event_ids = [e for e in event_ids if int(e) >= args.start]

    total = len(event_ids)
    for i, eid in enumerate(event_ids):
        if eid in progress and 'error' not in progress[eid]:
            continue

        event_data = events[eid]
        if 'error' in event_data:
            progress[eid] = {'skipped': event_data['error']}
            continue

        print(f"\n[{i+1}/{total}] Event {eid}: {event_data['event']} ({event_data['strategy']})")

        try:
            downloaded = download_event_images(eid, event_data)
            progress[eid] = {'downloaded': downloaded, 'total': len(downloaded)}
            log(f"Total: {len(downloaded)} images")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved.")
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            sys.exit(0)
        except Exception as e:
            log(f"FAILED: {e}")
            progress[eid] = {'error': str(e)}

        if (i + 1) % 10 == 0:
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

    n_ok = sum(1 for v in progress.values() if 'error' not in v and 'skipped' not in v)
    print(f"\n\nDone. {n_ok}/{len(progress)} events downloaded.")


if __name__ == '__main__':
    main()
