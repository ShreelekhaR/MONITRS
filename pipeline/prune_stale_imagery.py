"""
Delete chips that were downloaded at a center the event no longer has.

fix_centers.py ends by printing that imagery for repaired events points at the
old location and should be deleted. Nothing enforced that, so 141 events kept
chips of somewhere else — one of them 12,546 km away. A chip of the wrong
continent is worse than a missing chip: the missing one is skipped, the wrong
one teaches the model that a hurricane looks like whatever happened to be at
those coordinates.

An event needs re-downloading when center_original exists and differs from
center by more than the chip can absorb. The chip spans 2*halfwidth degrees,
so a shift much smaller than the halfwidth still shows the same ground; a
shift larger than it shows different ground entirely.

Usage:
    python pipeline/prune_stale_imagery.py                # report only
    python pipeline/prune_stale_imagery.py --delete
    python pipeline/prune_stale_imagery.py --delete --min-km 5
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_centers import haversine_km

EVENTS_PATH = 'Data/events_processed.json'
IMAGES_DIR = 'Data/images'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--images', default=IMAGES_DIR)
    ap.add_argument('--min-km', type=float, default=None,
                    help='Override the per-event halfwidth test with a fixed '
                         'distance in km')
    ap.add_argument('--delete', action='store_true',
                    help='Actually remove the directories')
    args = ap.parse_args()

    ev = json.load(open(args.events))
    dirs = sorted(d for d in os.listdir(args.images)
                  if os.path.isdir(os.path.join(args.images, d)))

    stale, fresh, no_record, tiny_shift = [], 0, [], 0
    for d in dirs:
        eid = d.replace('event_', '').replace('ev', '')
        v = ev.get(eid)
        if not v:
            no_record.append(d)
            continue
        o, c = v.get('center_original'), v.get('center')
        if not o or not c:
            fresh += 1
            continue
        km = haversine_km(tuple(o[:2]), tuple(c[:2]))
        # A chip covers roughly halfwidth degrees from its center; 1 deg of
        # latitude is ~111 km. A shift under a tenth of that still images
        # essentially the same scene.
        if args.min_km is not None:
            limit = args.min_km
        else:
            limit = max(1.0, float(v.get('halfwidth', 0.05)) * 111.0 * 0.1)
        if km <= limit:
            tiny_shift += 1
            continue
        n = len([f for f in os.listdir(os.path.join(args.images, d))
                 if f.endswith(('.png', '.jpg'))])
        stale.append((d, km, n, v))

    n_frames = sum(x[2] for x in stale)
    print(f'{len(dirs)} event dirs')
    print(f'  center never moved            {fresh:>5}')
    print(f'  moved less than the chip span {tiny_shift:>5}  (same scene, kept)')
    print(f'  STALE                         {len(stale):>5}  '
          f'({n_frames} frames of the wrong place)')
    if no_record:
        print(f'  no event record               {len(no_record):>5}  '
              f'{no_record[:5]}')

    if stale:
        stale.sort(key=lambda x: -x[1])
        print(f'\n  furthest off:')
        for d, km, n, v in stale[:12]:
            print(f'    {d:<8} {km:>9.1f} km  {n:>3} frames  '
                  f'{v.get("state")} {v.get("county")}')

    if not stale:
        print('\nNothing to prune.')
        return
    if not args.delete:
        print(f'\nDry run. Re-run with --delete to remove these '
              f'{len(stale)} directories, then re-download:')
        print('    python pipeline/prune_stale_imagery.py --delete')
        print('    python pipeline/download_imagery.py')
        return

    removed = 0
    for d, _, _, _ in stale:
        try:
            shutil.rmtree(os.path.join(args.images, d))
            removed += 1
        except Exception as e:
            print(f'  could not remove {d}: {e}')
    print(f'\nremoved {removed} directories ({n_frames} frames)')
    print('Re-download with: python pipeline/download_imagery.py')


if __name__ == '__main__':
    main()
