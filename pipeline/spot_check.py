"""
Contact sheets for spot-checking what the signal test actually saw.

Both real defects found so far — chips centered 36 km off the fire, and a
render so dark that 92% of a scene sat below DN 40 — were found by looking at
the pixels, not by reading scores. A number can only tell you a chip
disagrees with its label; it cannot tell you the chip is of the wrong place.

Sheets label every frame with its phase and date, and the header carries what
the signal test concluded, so a sheet answers "does this look like what the
metric claims" in one glance.

Usage:
    python pipeline/spot_check.py --verdict CONTRADICTED --n 12
    python pipeline/spot_check.py --type "Severe Ice Storm" --top --n 8
    python pipeline/spot_check.py --event 5381 5357
"""

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw

ALIGNED = 'Data/aligned_frames.json'
SIGNAL = 'Data/visual_signal.json'
EVENTS_PATH = 'Data/events_processed.json'
OUT_DIR = 'Data/spot_checks'

TILE = 200
HEADER = 26
LABEL = 16
COLS = 8


def sheet(eid, ev, sig, meta, path):
    frames = ev.get('frames', [])
    if not frames:
        return False
    # Chronological, so the eye reads a burn scar appearing rather than
    # hunting for it among frames grouped by phase.
    frames = sorted(frames, key=lambda f: f['date'])

    cols = min(len(frames), COLS)
    rows = (len(frames) + cols - 1) // cols
    cell = TILE + LABEL
    im = Image.new('RGB', (cols * TILE, HEADER + rows * cell), (17, 17, 17))
    d = ImageDraw.Draw(im)

    strength = sig.get('signal_strength', 0)
    cells = sig.get('scale_cells')
    area = sig.get('area_frac')
    head = (f'ev{eid}  {sig.get("type","?")}  {meta.get("state","")} '
            f'{meta.get("county","")}  |  {sig.get("verdict","?")} '
            f'{strength:.3f}'
            + (f'  @{cells}x{cells}' if cells else '')
            + (f'  {area:.0%} of chip' if area is not None else '')
            + (f'  |  WRONG WAY {sig["wrong_way_strength"]:.3f} on '
               f'{sig.get("wrong_way_channel","?")}'
               if sig.get('wrong_way_strength') else
               f'  |  chip-wide {sig.get("chip_wide_strength",0):.3f}')
            + f'  hw={meta.get("halfwidth","?")} {meta.get("strategy","?")}')
    d.text((6, 7), head[:170], fill=(230, 230, 230))

    for i, f in enumerate(frames):
        x, y = (i % cols) * TILE, HEADER + (i // cols) * cell
        try:
            t = Image.open(f['path']).convert('RGB').resize((TILE, TILE))
        except Exception:
            continue
        im.paste(t, (x, y))
        ph = f.get('phase', '')
        # Pre-event frames are the baseline everything is measured against;
        # colour them apart so a sheet with one usable pre frame is obvious.
        col = (120, 200, 255) if ph == 'pre-event' else (255, 220, 140)
        d.text((x + 3, y + TILE + 2), f'{f["date"][5:]} {ph[:9]}', fill=col)

    im.save(path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aligned', default=ALIGNED)
    ap.add_argument('--signal', default=SIGNAL)
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--verdict', default=None)
    ap.add_argument('--type', default=None)
    ap.add_argument('--event', nargs='+', default=None)
    ap.add_argument('--top', action='store_true',
                    help='Strongest first rather than a spread across the range')
    ap.add_argument('--n', type=int, default=10)
    args = ap.parse_args()

    aligned = json.load(open(args.aligned))
    sig = json.load(open(args.signal))
    meta = json.load(open(args.events)) if os.path.exists(args.events) else {}

    keys = [k for k in sig if k in aligned]
    if args.event:
        want = {str(e) for e in args.event}
        keys = [k for k in keys if k in want]
    if args.verdict:
        keys = [k for k in keys if sig[k].get('verdict') == args.verdict]
    if args.type:
        keys = [k for k in keys if sig[k].get('type') == args.type]
    if not keys:
        print('nothing matched those filters')
        return 1

    keys.sort(key=lambda k: -sig[k].get('signal_strength', 0))
    if args.top or len(keys) <= args.n:
        pick = keys[:args.n]
    else:
        # An even spread beats the top N: the interesting failures are usually
        # in the middle of the range, not at either end.
        step = len(keys) / args.n
        pick = [keys[int(i * step)] for i in range(args.n)]

    os.makedirs(args.out, exist_ok=True)
    made = 0
    for k in pick:
        p = os.path.join(args.out, f'{k}.png')
        if sheet(k, aligned[k], sig[k], meta.get(k, {}), p):
            made += 1
            s = sig[k]
            print(f'  {p}   {s.get("type","?"):<18} {s.get("verdict","?"):<12} '
                  f'{s.get("signal_strength",0):.3f}')

    print(f'\n{made} sheets in {args.out}/ (of {len(keys)} matching events)')
    print('Blue labels are pre-event frames, amber are during/post.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
