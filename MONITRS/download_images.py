"""
MONITRS v2 — Image download by caption dates
Downloads the nearest Sentinel-2 image for each caption date,
plus 2 pre-event images.

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


def download_mosaic(col, target_date, region, output_file, search_days=7):
    target_dt = datetime.datetime.strptime(target_date, '%Y-%m-%d')

    for offset in range(search_days + 1):
        for direction in [0, 1, -1]:
            if offset == 0 and direction != 0:
                continue
            check_dt = target_dt + relativedelta(days=offset * (direction if direction != 0 else 1))
            check_str = check_dt.strftime('%Y-%m-%d')
            next_str = (check_dt + relativedelta(days=1)).strftime('%Y-%m-%d')

            day_col = col.filterDate(check_str, next_str)
            if day_col.size().getInfo() == 0:
                continue

            mosaic = day_col.mosaic()
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

            actual = check_str if check_str != target_date else None
            return True, actual

    return False, None


def download_event_images(event_idx, event_data, max_cloud_pct=50, pre_images=2):
    center = event_data['center']
    halfwidth = event_data.get('halfwidth', 0.05)
    start_date = event_data['start_date']

    caption_dates = parse_caption_dates(event_data.get('captions', ''))
    if not caption_dates:
        log("No caption dates")
        return {'pre': [], 'caption': []}

    region = ee.Geometry.Rectangle([
        [center[1] - halfwidth, center[0] - halfwidth],
        [center[1] + halfwidth, center[0] + halfwidth],
    ])

    earliest_caption = datetime.datetime.strptime(caption_dates[0], '%Y-%m-%d')
    latest_caption = datetime.datetime.strptime(caption_dates[-1], '%Y-%m-%d')
    search_start = (earliest_caption - relativedelta(days=30)).strftime('%Y-%m-%d')
    search_end = (latest_caption + relativedelta(days=14)).strftime('%Y-%m-%d')

    base_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filterDate(search_start, search_end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_pct))

    outdir = join(ODIR, str(event_idx))
    makedirs(outdir, exist_ok=True)

    # Download pre-event images (last 2 before first caption date)
    pre_dates = []
    pre_col = base_col.filterDate(search_start, caption_dates[0])
    num_pre = pre_col.size().getInfo()
    if num_pre > 0:
        img_list = pre_col.sort('system:time_start').toList(num_pre)
        seen = set()
        all_pre = []
        for i in range(num_pre):
            try:
                d = ee.Image(img_list.get(i)).date().format('YYYY-MM-dd').getInfo()
                if d not in seen:
                    seen.add(d)
                    all_pre.append(d)
            except Exception:
                continue
        for pre_date in all_pre[-pre_images:]:
            output_file = join(outdir, f'{event_idx}_pre_{pre_date}.png')
            if isfile(output_file) and os.path.getsize(output_file) > 1000:
                pre_dates.append(pre_date)
                continue
            try:
                ok, _ = download_mosaic(base_col, pre_date, region, output_file, search_days=0)
                if ok:
                    pre_dates.append(pre_date)
            except Exception:
                if isfile(output_file):
                    os.remove(output_file)
    log(f"pre: {len(pre_dates)} images")

    # Download one image per caption date
    caption_downloaded = []
    for target_date in caption_dates:
        output_file = join(outdir, f'{event_idx}_{target_date}.png')
        if isfile(output_file) and os.path.getsize(output_file) > 1000:
            caption_downloaded.append(target_date)
            continue
        try:
            ok, actual = download_mosaic(base_col, target_date, region, output_file)
            if ok:
                caption_downloaded.append(target_date)
                suffix = f" (nearest: {actual})" if actual else ""
                log(f"  {target_date}: ok{suffix}")
            else:
                log(f"  {target_date}: no image within 7 days")
        except Exception as e:
            if isfile(output_file):
                os.remove(output_file)
            log(f"  {target_date}: error — {e}")

    log(f"caption: {len(caption_downloaded)}/{len(caption_dates)} images")

    # Download post-event images (first 2 after last caption date)
    post_dates = []
    last_caption = caption_dates[-1]
    post_start = (datetime.datetime.strptime(last_caption, '%Y-%m-%d') + relativedelta(days=1)).strftime('%Y-%m-%d')
    post_col = base_col.filterDate(post_start, search_end)
    num_post = post_col.size().getInfo()
    if num_post > 0:
        img_list = post_col.sort('system:time_start').toList(num_post)
        seen = set()
        all_post = []
        for i in range(num_post):
            try:
                d = ee.Image(img_list.get(i)).date().format('YYYY-MM-dd').getInfo()
                if d not in seen:
                    seen.add(d)
                    all_post.append(d)
            except Exception:
                continue
        for post_date in all_post[:2]:
            output_file = join(outdir, f'{event_idx}_post_{post_date}.png')
            if isfile(output_file) and os.path.getsize(output_file) > 1000:
                post_dates.append(post_date)
                continue
            try:
                ok, _ = download_mosaic(base_col, post_date, region, output_file, search_days=0)
                if ok:
                    post_dates.append(post_date)
            except Exception:
                if isfile(output_file):
                    os.remove(output_file)
    log(f"post: {len(post_dates)} images")

    return {'pre': pre_dates, 'caption': caption_downloaded, 'post': post_dates}


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
            image_dates = download_event_images(eid, event_data)
            n_total = len(image_dates['pre']) + len(image_dates['caption'])
            progress[eid] = {'image_dates': image_dates, 'total_images': n_total}
            log(f"Total: {n_total} images")
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
