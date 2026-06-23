"""
Check available Sentinel-2 imagery for an event at different cloud thresholds.

Usage:
    python temp.py --event 1
    python temp.py --lat 40.224 --lon -100.191 --start 2022-04-08 --end 2022-06-05
"""

import ee
import os
import argparse
import json

EE_PROJECT_ID = os.environ.get('EE_PROJECT_ID', os.environ.get('GCP_PROJECT_ID'))
ee.Initialize(project=EE_PROJECT_ID)


def check_imagery(center, start, end, halfwidth=0.05):
    region = ee.Geometry.Rectangle([
        [center[1] - halfwidth, center[0] - halfwidth],
        [center[1] + halfwidth, center[0] + halfwidth],
    ])

    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region).filterDate(start, end)
    num = col.size().getInfo()

    print(f"Center: ({center[0]:.4f}, {center[1]:.4f})")
    print(f"Date range: {start} to {end}")
    print(f"Total scenes (no filter): {num}")
    print()

    for threshold in [30, 50, 70, 100]:
        n = col.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', threshold)).size().getInfo()
        print(f"  Under {threshold}% cloud: {n}")

    print()
    print(f"{'Date':<14} {'Cloud %':>8}")
    print('-' * 24)

    img_list = col.sort('system:time_start').toList(num)
    for i in range(num):
        img = ee.Image(img_list.get(i))
        d = img.date().format('YYYY-MM-dd').getInfo()
        cloud = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        marker = ' <-- filtered at 30%' if cloud >= 30 else ''
        print(f"  {d:<14} {cloud:>7.1f}%{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event', type=int, default=None)
    parser.add_argument('--lat', type=float, default=None)
    parser.add_argument('--lon', type=float, default=None)
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    args = parser.parse_args()

    if args.event is not None:
        r = json.load(open('Data/events_processed.json'))
        e = r[str(args.event)]
        center = e['center']
        start = e['start_date']
        end = e['end_date']
        print(f"Event {args.event}: {e['event']} ({e['type']})")
        print(f"Strategy: {e['strategy']}")
        print()
    else:
        center = [args.lat, args.lon]
        start = args.start
        end = args.end

    check_imagery(center, start, end)


if __name__ == '__main__':
    main()
