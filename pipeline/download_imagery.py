"""
Download Sentinel-2 imagery for harvested events, aligned to fact dates.

Unlike the earlier download (which used a fixed cadence over the FEMA window),
this brackets the dates where we actually have article-verified facts, so every
image has a defensible statement attached to it:

  - one baseline frame BEFORE the earliest fact date
  - frames near each fact date (the nearest clear acquisition)
  - one recovery frame AFTER the last fact date

Usage:
    export EE_PROJECT_ID=planet-earthengine-staging
    earthengine authenticate          # first time only
    python pipeline/download_imagery.py --event 5357
    python pipeline/download_imagery.py --all --workers 4
"""

import argparse
import json
import os
from datetime import datetime, timedelta

HARVEST_DIR = 'Data/harvest'
IMAGES_DIR = 'Data/images'

CLOUD_THRESHOLD = 40      # max scene cloud cover %
BLACK_PCT_MAX = 5.0       # reject frames with too many no-data pixels
IMG_SIZE = 512


def fact_dates(rec):
    """Dates we have verified facts for, sorted."""
    out = set()
    for f in rec['facts']:
        if not f.get('is_about_target_event'):
            continue
        d = f.get('extent_as_of_date') or f.get('pub_date')
        if d:
            out.add(d)
        for k, v in (f.get('notable_dates') or {}).items():
            if v:
                out.add(v)
    return sorted(out)


def target_dates(rec, baseline_days=21, recovery_days=30):
    """Bracket the fact dates with a baseline and a recovery frame."""
    fd = fact_dates(rec)
    if not fd:
        s, e = rec.get('fema_start'), rec.get('fema_end')
        return [d for d in (s, e) if d]
    try:
        first = datetime.strptime(fd[0], '%Y-%m-%d')
        last = datetime.strptime(fd[-1], '%Y-%m-%d')
    except Exception:
        return fd
    baseline = (first - timedelta(days=baseline_days)).strftime('%Y-%m-%d')
    recovery = (last + timedelta(days=recovery_days)).strftime('%Y-%m-%d')
    return [baseline] + fd + [recovery]


def download_event(rec, out_root=IMAGES_DIR, window_days=7, max_images=12):
    import ee
    eid = rec['event_id']
    center = rec.get('center')
    hw = rec.get('halfwidth', 0.05)
    if not center:
        return {'event_id': eid, 'error': 'no center'}

    out_dir = os.path.join(out_root, str(eid))
    os.makedirs(out_dir, exist_ok=True)

    lat, lon = center[0], center[1]
    region = ee.Geometry.Rectangle([lon - hw, lat - hw, lon + hw, lat + hw])

    tds = target_dates(rec)
    if not tds:
        return {'event_id': eid, 'error': 'no target dates'}

    lo = (datetime.strptime(tds[0], '%Y-%m-%d') - timedelta(days=window_days)
          ).strftime('%Y-%m-%d')
    hi = (datetime.strptime(tds[-1], '%Y-%m-%d') + timedelta(days=window_days)
          ).strftime('%Y-%m-%d')

    coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(lo, hi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD))
            .sort('system:time_start'))

    try:
        info = coll.getInfo()
    except Exception as e:
        return {'event_id': eid, 'error': f'EE query failed: {e}'}

    feats = info.get('features', [])
    if not feats:
        return {'event_id': eid, 'error': 'no clear scenes', 'window': [lo, hi]}

    # available acquisition dates
    avail = []
    for ft in feats:
        ts = ft['properties'].get('system:time_start')
        if ts:
            avail.append((datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d'), ft['id']))
    avail = sorted(set(avail))

    # pick nearest available scene to each target date
    picked, seen = [], set()
    for td in tds:
        t = datetime.strptime(td, '%Y-%m-%d')
        best = min(avail, key=lambda a: abs(
            (datetime.strptime(a[0], '%Y-%m-%d') - t).days), default=None)
        if best and best[0] not in seen:
            seen.add(best[0])
            picked.append(best)
    picked = sorted(picked)[:max_images]

    saved, errors = [], []
    for date_str, img_id in picked:
        path = os.path.join(out_dir, f'{eid}_{date_str}.png')
        if os.path.exists(path):
            saved.append(date_str)
            continue
        try:
            img = (ee.Image(img_id).select(['B4', 'B3', 'B2'])
                   .divide(10000).clamp(0, 0.3).divide(0.3).multiply(255).toByte())
            url = img.getThumbURL({'region': region, 'dimensions': IMG_SIZE,
                                   'format': 'png'})
            import requests
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) > 5000:
                with open(path, 'wb') as f:
                    f.write(r.content)
                saved.append(date_str)
            else:
                errors.append(f'{date_str}: HTTP {r.status_code}')
        except Exception as e:
            errors.append(f'{date_str}: {str(e)[:80]}')

    return {'event_id': eid, 'target_dates': tds, 'saved': saved,
            'errors': errors, 'n_saved': len(saved)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--harvest-dir', default=HARVEST_DIR)
    ap.add_argument('--out', default=IMAGES_DIR)
    ap.add_argument('--event', nargs='+', type=int, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()

    import ee
    proj = os.environ.get('EE_PROJECT_ID', 'planet-earthengine-staging')
    try:
        ee.Initialize(project=proj)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=proj)
    print(f'Earth Engine ready (project={proj})')

    recs = []
    for fn in sorted(os.listdir(args.harvest_dir)):
        if not fn.endswith('.json'):
            continue
        eid = int(fn.split('.')[0])
        if args.event and eid not in args.event:
            continue
        recs.append(json.load(open(os.path.join(args.harvest_dir, fn))))

    if not recs:
        print('No harvest records'); return
    print(f'Downloading imagery for {len(recs)} events\n')

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_event, r, args.out): r['event_id'] for r in recs}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            if res.get('error'):
                print(f'  ev{res["event_id"]}: ERROR {res["error"]}')
            else:
                print(f'  ev{res["event_id"]}: {res["n_saved"]} images '
                      f'({len(res.get("errors", []))} failed)')

    total = sum(r.get('n_saved', 0) for r in results)
    ok = sum(1 for r in results if r.get('n_saved'))
    print(f'\n{total} images across {ok}/{len(results)} events -> {args.out}')


if __name__ == '__main__':
    main()
