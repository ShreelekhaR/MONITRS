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

CLOUD_THRESHOLD = 70      # max scene-level cloud cover %
NODATA_PCT_MAX = 60.0     # reject only frames that are mostly off-swath
WHITE_PCT_MAX = 80.0      # reject only near-total cloud/snow whiteout
IMG_SIZE = 512


def frame_quality(png_bytes):
    """Reject unusable frames. Returns (ok, stats).

    Distinguishes true no-data from dark-but-valid surface. Water renders
    near-black in RGB, so a coastal chip can be 60-80% dark while being
    perfectly good imagery — which matters because most hurricane and coastal
    storm events are on the coast. True off-swath no-data is exactly zero
    across all three channels.

    nodata_pct — pixels at literal zero in R, G and B
    white_pct  — near-saturated pixels (cloud / snow)
    valid_std  — variation among non-no-data pixels; a featureless frame
                 carries no information even if technically valid
    """
    try:
        from PIL import Image
        import io
        import numpy as np
    except ImportError:
        return True, {}

    try:
        im = Image.open(io.BytesIO(png_bytes)).convert('RGB')
        a = np.asarray(im, dtype=np.float32)
    except Exception:
        return False, {'error': 'undecodable'}

    if a.size == 0:
        return False, {'error': 'empty'}

    lum = a.mean(axis=2)
    nodata = a.max(axis=2) < 1.0
    nodata_pct = float(nodata.mean() * 100.0)
    white_pct = float((lum > 240).mean() * 100.0)
    std = float(lum.std())
    valid = ~nodata
    valid_std = float(lum[valid].std()) if valid.any() else 0.0

    ok = (nodata_pct <= NODATA_PCT_MAX
          and white_pct <= WHITE_PCT_MAX
          and valid_std >= 4.0)
    return ok, {'nodata_pct': round(nodata_pct, 1),
                'white_pct': round(white_pct, 1),
                'std': round(std, 1),
                'valid_std': round(valid_std, 1)}


def _parse(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except Exception:
        return None


def fact_dates(rec, window_slack_days=45):
    """Dates we have verified facts for, clamped to the FEMA event window.

    Retrospective articles ("3 years later...") and mis-parsed notable_dates
    otherwise pull in acquisition targets years away from the event.
    """
    fs, fe = rec.get('fema_start'), rec.get('fema_end')
    lo = hi = None
    try:
        if fs:
            lo = datetime.strptime(fs, '%Y-%m-%d') - timedelta(days=window_slack_days)
        if fe:
            hi = datetime.strptime(fe, '%Y-%m-%d') + timedelta(days=window_slack_days)
    except Exception:
        pass

    def ok(d):
        try:
            t = datetime.strptime(d, '%Y-%m-%d')
        except Exception:
            return False
        if lo and t < lo:
            return False
        if hi and t > hi:
            return False
        return True

    out = set()
    for f in rec['facts']:
        if not f.get('is_about_target_event'):
            continue
        d = f.get('extent_as_of_date') or f.get('pub_date')
        if d and ok(d):
            out.add(d)
        for _, v in (f.get('notable_dates') or {}).items():
            if v and ok(v):
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


def download_event(rec, out_root=IMAGES_DIR, pad_before=30, pad_after=45,
                   max_images=24):
    """Fetch every clear acquisition across the event window.

    Sentinel-2 revisits every ~5 days, so a padded event window yields 10-20
    frames. Frames between article dates still carry useful bounds from the
    alignment layer, and a dense sequence is what lets progression emerge.
    """
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

    # Window: pad around the union of FEMA window and verified fact dates
    fd = fact_dates(rec)
    anchors = [d for d in [rec.get('fema_start'), rec.get('fema_end')] if d] + fd
    anchors = sorted(a for a in anchors if _parse(a))
    if not anchors:
        return {'event_id': eid, 'error': 'no anchor dates'}

    lo = (_parse(anchors[0]) - timedelta(days=pad_before)).strftime('%Y-%m-%d')
    hi = (_parse(anchors[-1]) + timedelta(days=pad_after)).strftime('%Y-%m-%d')

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

    # One scene per acquisition date (lowest cloud wins)
    best_per_date = {}
    for ft in feats:
        ts = ft['properties'].get('system:time_start')
        if not ts:
            continue
        d = datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d')
        cc = ft['properties'].get('CLOUDY_PIXEL_PERCENTAGE', 100)
        if d not in best_per_date or cc < best_per_date[d][1]:
            best_per_date[d] = (ft['id'], cc)

    dates = sorted(best_per_date)
    # If more acquisitions than the cap, thin uniformly but always keep the
    # frames nearest each verified fact date.
    if len(dates) > max_images:
        must = set()
        for f in fd:
            t = _parse(f)
            if t:
                must.add(min(dates, key=lambda d: abs((_parse(d) - t).days)))
        remaining = max_images - len(must)
        others = [d for d in dates if d not in must]
        if remaining > 0 and others:
            step = len(others) / remaining
            keep = {others[min(int(i * step), len(others) - 1)]
                    for i in range(remaining)}
        else:
            keep = set()
        dates = sorted(must | keep)

    import requests
    saved, rejected, errors = [], [], []
    for date_str in dates:
        img_id = best_per_date[date_str][0]
        path = os.path.join(out_dir, f'{eid}_{date_str}.png')
        if os.path.exists(path):
            saved.append(date_str)
            continue
        try:
            img = (ee.Image(img_id).select(['B4', 'B3', 'B2'])
                   .divide(10000).clamp(0, 0.3).divide(0.3).multiply(255).toByte())
            url = img.getThumbURL({'region': region, 'dimensions': IMG_SIZE,
                                   'format': 'png'})
            r = requests.get(url, timeout=120)
            if r.status_code != 200 or len(r.content) < 5000:
                errors.append(f'{date_str}: HTTP {r.status_code}')
                continue
            ok, stats = frame_quality(r.content)
            if not ok:
                rejected.append({'date': date_str, **stats})
                continue
            with open(path, 'wb') as f:
                f.write(r.content)
            saved.append(date_str)
        except Exception as e:
            errors.append(f'{date_str}: {str(e)[:80]}')

    return {'event_id': eid, 'window': [lo, hi], 'n_available': len(best_per_date),
            'saved': saved, 'rejected': rejected, 'errors': errors,
            'n_saved': len(saved), 'n_rejected': len(rejected)}


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
                rej = res.get('n_rejected', 0)
                rej_s = f', {rej} rejected' if rej else ''
                print(f'  ev{res["event_id"]}: {res["n_saved"]} frames '
                      f'of {res.get("n_available", "?")} available{rej_s}')

    total = sum(r.get('n_saved', 0) for r in results)
    total_rej = sum(r.get('n_rejected', 0) for r in results)
    ok = sum(1 for r in results if r.get('n_saved'))
    print(f'\n{total} frames across {ok}/{len(results)} events '
          f'({total_rej} rejected for cloud/no-data) -> {args.out}')


if __name__ == '__main__':
    main()
