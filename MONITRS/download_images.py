"""
MONITRS v2 — Image download (runs on Workbench after language pipeline)
Reads Data/events_processed.json and downloads the nearest Sentinel-2 image
for each caption date.

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


def find_nearest_image(col, target_date, region, search_days=7):
    target_dt = datetime.datetime.strptime(target_date, '%Y-%m-%d')

    for offset in range(search_days + 1):
        for direction in [0, 1, -1]:
            if offset == 0 and direction != 0:
                continue
            check_dt = target_dt + relativedelta(days=offset * (direction if direction != 0 else 1))
            check_str = check_dt.strftime('%Y-%m-%d')
            next_str = (check_dt + relativedelta(days=1)).strftime('%Y-%m-%d')

            day_col = col.filterDate(check_str, next_str)
            n = day_col.size().getInfo()
            if n > 0:
                return day_col.mosaic(), check_str

    return None, None


def download_event_images(event_idx, event_data, max_cloud_pct=50):
    center = event_data['center']
    halfwidth = event_data.get('halfwidth', 0.05)
    start_date = event_data['start_date']
    end_date = event_data['end_date']

    caption_dates = parse_caption_dates(event_data.get('captions', ''))
    if not caption_dates:
        log("No caption dates found")
        return []

    region = ee.Geometry.Rectangle([
        [center[1] - halfwidth, center[0] - halfwidth],
        [center[1] + halfwidth, center[0] + halfwidth],
    ])

    # Wide date range to search for nearest images
    earliest = datetime.datetime.strptime(caption_dates[0], '%Y-%m-%d') - relativedelta(days=14)
    latest = datetime.datetime.strptime(caption_dates[-1], '%Y-%m-%d') + relativedelta(days=14)

    base_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filterDate(earliest.strftime('%Y-%m-%d'), latest.strftime('%Y-%m-%d')) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))

    outdir = join(ODIR, str(event_idx))
    makedirs(outdir, exist_ok=True)

    downloaded = []

    for target_date in caption_dates:
        output_file = join(outdir, f'{event_idx}_{target_date}.png')
        if isfile(output_file) and os.path.getsize(output_file) > 1000:
            downloaded.append(target_date)
            continue

        mosaic, actual_date = find_nearest_image(base_col, target_date, region)
        if mosaic is None:
            log(f"  {target_date}: no image within 7 days")
            continue

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
                log(f"  {target_date}: rejected (dark/nodata)")
                continue

            downloaded.append(target_date)
            suffix = f" (actual: {actual_date})" if actual_date != target_date else ""
            log(f"  {target_date}: downloaded{suffix}")

        except Exception as e:
            if isfile(output_file):
                os.remove(output_file)
            log(f"  {target_date}: error — {e}")
            continue

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
            progress[eid] = {'downloaded_dates': downloaded, 'total': len(downloaded)}
            log(f"Downloaded {len(downloaded)} images")
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
