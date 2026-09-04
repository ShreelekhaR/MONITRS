"""
Place fire events on the fire, and size the chip to it.

Fire centers came from NASA FIRMS thermal detections, which is the right
source and was applied the wrong way: the stored center was the plain MEAN of
every detection in the declared county. September 2020 in Lane County, Oregon
had two fire complexes burning at once. Averaging them put the center 36 km
from either — on untouched forest, which is exactly what the chips show.

    10,593 detections -> 4 spatially connected clusters
      8,769 dets  FRP 235,364  at 44.1449, -122.4647   26 x 44 km  (Holiday Farm)
      1,822 dets  FRP  32,232  at 43.4330, -122.9629    9 x 25 km
      plain mean of everything: 44.0271, -122.5593  <- between them, on nothing

The second defect is the chip. Every event is fetched at halfwidth 0.05, an
11 km square. The cluster above is 44 km across, so even a perfectly centered
chip would show 7% of the fire — and the bounded facts we ask questions
against ("at least 30,000 acres had burned") are claims about the whole fire.
A chip that cannot contain it cannot support the question.

So: cluster the detections, take the strongest cluster that falls inside the
declared county, and size the chip to cover it. Resolution is traded for
coverage and the trade is recorded per event as gsd_m, because a chip that
misses the fire has no resolution worth keeping.

Needs FIRMS_MAP_KEY (free: https://firms.modaps.eosdis.nasa.gov/api/map_key/).

Reports without changing anything unless --write is passed.

Usage:
    python pipeline/place_fires.py --harvested            # report only
    python pipeline/place_fires.py --event 5381 5357
    python pipeline/place_fires.py --harvested --write
"""

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_centers import (CACHE_PATH, EVENTS_PATH, bbox_usable, county_geo,
                         haversine_km, in_county)

FIRMS_CACHE = 'Data/firms_hotspots'
IMG_SIZE = 512            # must match download_imagery.IMG_SIZE

# FIRMS archive sources. VIIRS at 375 m resolves a fire front; MODIS at 1 km
# is the fallback for events before VIIRS coverage.
SOURCES = ['VIIRS_SNPP_SP', 'MODIS_SP']
MAX_DAY_RANGE = 5         # hard limit of the area/csv endpoint
THROTTLE_S = 1.0

# Padding around the FEMA window. Fires are usually burning before the
# declaration and keep burning after the end date.
PAD_BEFORE, PAD_AFTER = 3, 10

CLUSTER_RES_DEG = 0.02    # ~2 km cells; detections in touching cells are one fire
MIN_CLUSTER_DETS = 5

# Chip sizing. The floor keeps small fires from being fetched at a resolution
# no coarser than they need; the ceiling stops one 200 km megafire from
# dragging the whole dataset to a 400 m ground sample.
MIN_HALFWIDTH = 0.05
MAX_HALFWIDTH = 0.25
CHIP_MARGIN = 1.25        # show some unburned ground for contrast


def _parse(d):
    try:
        return datetime.strptime(d, '%Y-%m-%d')
    except Exception:
        return None


def fetch_hotspots(bbox, start, end, key, cache_path):
    """FIRMS detections in bbox over [start, end]. Cached; returns a list.

    The endpoint takes at most a 5-day range per call, so a month-long fire is
    several requests. Both are cached together under the event.
    """
    if os.path.exists(cache_path):
        try:
            return json.load(open(cache_path))
        except Exception:
            pass

    s, e = _parse(start), _parse(end)
    if not s or not e:
        return []
    south, north, west, east = bbox
    area = f'{west},{south},{east},{north}'

    rows, seen = [], set()
    for source in SOURCES:
        cur = s
        while cur <= e:
            span = min(MAX_DAY_RANGE, (e - cur).days + 1)
            url = (f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/'
                   f'{key}/{source}/{area}/{span}/{cur.strftime("%Y-%m-%d")}')
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200 and r.text.lstrip().startswith('lat'):
                    for d in csv.DictReader(io.StringIO(r.text)):
                        try:
                            rec = (float(d['latitude']), float(d['longitude']),
                                   float(d.get('frp') or 0), d.get('acq_date'))
                        except (TypeError, ValueError):
                            continue
                        # The two sources overlap in space and time; a MODIS
                        # and a VIIRS detection of the same flame front are
                        # one fire, not two votes for it.
                        k = (round(rec[0], 3), round(rec[1], 3), rec[3])
                        if k in seen:
                            continue
                        seen.add(k)
                        rows.append(rec)
            except Exception:
                pass
            cur += timedelta(days=span)
            time.sleep(THROTTLE_S)
        if rows and source == SOURCES[0]:
            break        # VIIRS covered it; no need to pay for MODIS as well

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(rows, f)
    return rows


def cluster(rows, res=CLUSTER_RES_DEG):
    """Group detections into spatially connected fires, strongest first.

    Detections land on a grid and touching cells (including diagonally) are
    flooded into one cluster. Ranked by total radiative power rather than
    count, so a dense scatter of small agricultural burns does not outrank the
    fire the declaration is about.
    """
    if not rows:
        return []
    lat = np.array([r[0] for r in rows])
    lon = np.array([r[1] for r in rows])
    frp = np.array([r[2] for r in rows])

    gi = ((lat - lat.min()) / res).astype(int)
    gj = ((lon - lon.min()) / res).astype(int)
    occ = {}
    for k, cell in enumerate(zip(gi, gj)):
        occ.setdefault(cell, []).append(k)

    label, n = {}, 0
    for cell in occ:
        if cell in label:
            continue
        n += 1
        label[cell] = n
        q = deque([cell])
        while q:
            a, b = q.popleft()
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    c = (a + da, b + db)
                    if c in occ and c not in label:
                        label[c] = n
                        q.append(c)

    groups = {}
    for cell, idxs in occ.items():
        groups.setdefault(label[cell], []).extend(idxs)

    cosl = float(np.cos(np.radians(lat.mean())))
    out = []
    for idxs in groups.values():
        if len(idxs) < MIN_CLUSTER_DETS:
            continue
        i = np.array(idxs)
        # Weight the centroid by radiative power so it sits on the fire's
        # core rather than being pulled out by a thin scatter of edge pixels.
        w = frp[i] + 1.0
        out.append({
            'n': int(len(i)),
            'frp': float(frp[i].sum()),
            'center': [float(np.average(lat[i], weights=w)),
                       float(np.average(lon[i], weights=w))],
            'span_km': [float((lat[i].max() - lat[i].min()) * 111.0),
                        float((lon[i].max() - lon[i].min()) * 111.0 * cosl)],
            'span_deg': [float(lat[i].max() - lat[i].min()),
                         float(lon[i].max() - lon[i].min())],
            'dates': sorted({rows[k][3] for k in idxs if rows[k][3]}),
        })
    out.sort(key=lambda c: -c['frp'])
    return out


def halfwidth_for(cl):
    """Chip halfwidth that contains the cluster, within limits."""
    need = max(cl['span_deg'][0], cl['span_deg'][1]) / 2.0 * CHIP_MARGIN
    return round(min(max(need, MIN_HALFWIDTH), MAX_HALFWIDTH), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--cache', default=CACHE_PATH)
    ap.add_argument('--firms-cache', default=FIRMS_CACHE)
    ap.add_argument('--event', nargs='+', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--harvested', action='store_true',
                    help='Only events with a harvest record. Those are the '
                         'ones with imagery, and each unharvested event is '
                         'several FIRMS calls spent on a chip nobody fetches.')
    ap.add_argument('--harvest-dir', default='Data/harvest')
    ap.add_argument('--write', action='store_true',
                    help='Write centers back; otherwise report only')
    ap.add_argument('--dry-run', action='store_true',
                    help='Explicit no-op: reporting is already the default')
    args = ap.parse_args()

    key = os.environ.get('FIRMS_MAP_KEY', '').strip()
    if not key:
        print('FIRMS_MAP_KEY is not set. Get a free key at\n'
              '  https://firms.modaps.eosdis.nasa.gov/api/map_key/\n'
              'then: export FIRMS_MAP_KEY=...')
        return 1

    ev = json.load(open(args.events))
    geo = json.load(open(args.cache)) if os.path.exists(args.cache) else {}

    fires = [k for k, v in ev.items()
             if isinstance(v, dict) and v.get('type') == 'Fire' and 'error' not in v]
    if args.harvested:
        try:
            have = {os.path.splitext(f)[0]
                    for f in os.listdir(args.harvest_dir) if f.endswith('.json')}
            before = len(fires)
            fires = [k for k in fires if k in have]
            print(f'{before - len(fires)} fire events skipped: not harvested yet')
        except OSError:
            print(f'no harvest dir at {args.harvest_dir}; placing all fires')
    if args.event:
        want = {str(e) for e in args.event}
        fires = [k for k in fires if k in want]
    fires.sort(key=int)
    if args.limit:
        fires = fires[:args.limit]
    print(f'{len(fires)} fire events\n')

    moved, resized, unplaced, out_of_county = [], [], [], []
    for k in fires:
        v = ev[k]
        centroid, bbox = county_geo(v.get('county'), v.get('state'), geo)
        if not bbox_usable(bbox):
            unplaced.append((k, 'no county bbox'))
            continue
        start = v.get('start_date')
        end = v.get('end_date') or start
        s, e = _parse(start), _parse(end)
        if not s:
            unplaced.append((k, 'no start date'))
            continue
        lo = (s - timedelta(days=PAD_BEFORE)).strftime('%Y-%m-%d')
        hi = ((e or s) + timedelta(days=PAD_AFTER)).strftime('%Y-%m-%d')

        rows = fetch_hotspots(bbox, lo, hi, key,
                              os.path.join(args.firms_cache, f'{k}.json'))
        clusters = cluster(rows)
        if not clusters:
            unplaced.append((k, f'no hotspot cluster in {len(rows)} detections'))
            continue

        # The declaration is for one county; the chip has to show that
        # county's fire. A complex straddling a line has a cluster centroid
        # that may sit on the neighbour's side, and then the next cluster down
        # is the right one for this event.
        pick = next((c for c in clusters
                     if in_county(c['center'], centroid, bbox)), None)
        if pick is None:
            out_of_county.append((k, len(clusters)))
            continue

        hw = halfwidth_for(pick)
        old_c = v.get('center')
        old_hw = v.get('halfwidth', 0.05)
        d = haversine_km(tuple(old_c[:2]), tuple(pick['center'])) if old_c else None

        if d is not None and d > 1.0:
            moved.append((k, d, old_c, pick, len(clusters)))
        if abs(hw - float(old_hw)) > 1e-6:
            resized.append((k, old_hw, hw, pick['span_km']))

        if args.write:
            if v.get('center_original') is None:
                v['center_original'] = old_c
            v['center'] = list(pick['center'])
            v['halfwidth'] = hw
            v['strategy'] = 'firms_cluster'
            v['firms_cluster'] = {
                'n_detections': pick['n'], 'total_frp': round(pick['frp'], 1),
                'span_km': [round(x, 1) for x in pick['span_km']],
                'n_clusters_in_county': len(clusters),
                'gsd_m': round(hw * 2 * 111000.0 / IMG_SIZE, 1),
            }

    print(f'moved more than 1 km          {len(moved):>5}')
    print(f'chip resized                  {len(resized):>5}')
    print(f'no cluster inside the county  {len(out_of_county):>5}')
    print(f'could not place               {len(unplaced):>5}')

    if moved:
        moved.sort(key=lambda r: -r[1])
        print(f'\nlargest moves:')
        for k, d, old, c, nc in moved[:15]:
            print(f'  ev{k:<6} {d:>6.1f} km   {old[0]:.3f},{old[1]:.3f} -> '
                  f'{c["center"][0]:.3f},{c["center"][1]:.3f}   '
                  f'{c["n"]} dets, {nc} cluster(s), '
                  f'{c["span_km"][0]:.0f}x{c["span_km"][1]:.0f} km')
        ds = np.array([m[1] for m in moved])
        print(f'\n  median move {np.median(ds):.1f} km, '
              f'{(ds > 11).sum()} moved further than one chip width')

    if resized:
        big = [r for r in resized if r[2] > r[1]]
        print(f'\n{len(big)} chips enlarged; the old 11 km chip could not '
              f'contain the fire:')
        for k, o, n, span in sorted(big, key=lambda r: -r[2])[:10]:
            print(f'  ev{k:<6} halfwidth {o} -> {n}  '
                  f'(fire spans {span[0]:.0f}x{span[1]:.0f} km, '
                  f'{n * 2 * 111000 / IMG_SIZE:.0f} m/px)')

    if unplaced:
        print(f'\nunplaced (sample): '
              f'{[(k, r) for k, r in unplaced[:5]]}')

    if args.write:
        with open(args.events, 'w') as f:
            json.dump(ev, f, indent=2)
        with open(args.cache, 'w') as f:
            json.dump(geo, f, indent=2)
        print(f'\nwrote {args.events}')
        print('Chips at a moved center or a changed halfwidth expire on the '
              'next download_imagery.py run.')
    else:
        print('\n(dry run — nothing written; pass --write to apply)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
