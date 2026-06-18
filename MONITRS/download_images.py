"""
MONITRS v2 — Image download (runs on Workbench after language pipeline)
Reads Data/events_processed.json and downloads Sentinel-2 images for each event.

Usage:
    export EE_PROJECT_ID=your-ee-project-id
    python MONITRS/download_images.py

    # Process specific batch:
    python MONITRS/download_images.py --start 0 --end 1000
"""

import os
import sys
import json
import datetime
import argparse
import numpy as np
import urllib.request
from os.path import join, isfile
from os import makedirs
from dateutil.relativedelta import relativedelta
import ee
from PIL import Image
from tqdm import tqdm

EE_PROJECT_ID = os.environ.get('EE_PROJECT_ID', 'your-project-id')
ee.Initialize(project=EE_PROJECT_ID)

INPUT_FILE = 'Data/events_processed.json'
ODIR = 'Data/images'


def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"  [{ts}] {msg}")
    sys.stdout.flush()


def download_event_images(event_idx, event_data, pre_days=14, post_days=14, max_cloud_pct=30):
    center = event_data['center']
    halfwidth = event_data.get('halfwidth', 0.05)
    start_date = event_data['start_date']
    end_date = event_data['end_date']

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

    outdir = join(ODIR, str(event_idx))
    makedirs(outdir, exist_ok=True)
    all_dates = {}

    for phase, (d_start, d_end) in phases.items():
        if d_start >= d_end:
            all_dates[phase] = []
            continue

        col = base_col.filterDate(d_start, d_end).filter(
            ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))
        try:
            num = col.size().getInfo()
        except Exception:
            all_dates[phase] = []
            continue

        if num == 0:
            all_dates[phase] = []
            continue

        img_list = col.sort('system:time_start').toList(num)
        seen_dates = set()
        phase_dates = []

        for i in range(num):
            try:
                img_date = ee.Image(img_list.get(i)).date().format('YYYY-MM-dd').getInfo()
            except Exception:
                continue
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
                if isfile(output_file):
                    os.remove(output_file)
                continue

        all_dates[phase] = phase_dates
        log(f"{phase}: {len(phase_dates)} images")

    return all_dates


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

    # Track which events already have images
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
        if eid in progress:
            continue

        event_data = events[eid]
        if 'error' in event_data:
            progress[eid] = {'skipped': event_data['error']}
            continue

        print(f"\n[{i+1}/{total}] Event {eid}: {event_data['event']} ({event_data['strategy']})")

        try:
            image_dates = download_event_images(eid, event_data)
            n_total = sum(len(v) for v in image_dates.values())
            progress[eid] = {'image_dates': image_dates, 'total_images': n_total}
            log(f"Downloaded {n_total} images")
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
    print(f"\n\nDone. {n_ok}/{len(progress)} events downloaded. Progress: {progress_file}")


if __name__ == '__main__':
    main()
