"""
Test 1 — visual signal detection.

Does the disaster actually show up in the pixels? For each event, compare
pre-event frames against during/post frames and measure whether the expected
spectral change appears.

We only have RGB thumbnails (no NIR), so we use RGB proxies:

  greenness   G / (R+G+B)      vegetation proxy; drops after fire
  wetness     B / (R+G+B)      water proxy; rises with flooding
  brightness  mean luminance    rises with snow/ash, drops with burn scar
  texture     local std         drops when terrain is smoothed by snow/water

Each proxy is measured AT EVERY SPATIAL SCALE, not once over the whole chip.
Disasters differ in footprint by three orders of magnitude: an ice storm
whitens all 121 km2 of the chip, a 500-acre fire burns 2 km2 of it. A
chip-wide mean sees the first and dilutes the second below seasonal noise --
which is a fact about averaging, not about whether the fire is visible. So we
pool the chip into grids from 1x1 down to 16x16 (~0.7 km cells) and keep the
scale where the change is strongest, reporting which scale that was.

The change also has to beat the site's own pre-event variability: the same
statistic is computed for how far the pre-event frames sit from each other,
and that is subtracted. A cell that shifts as much between two ordinary
pre-event dates as it does at the disaster is not evidence.

Per event we emit:
  signal_strength   magnitude of change in the expected channel (0..1)
  direction_match   does the change go the way the disaster type predicts
  verdict           strong / weak / none / CONTRADICTED

CONTRADICTED usually means a mis-centered chip or wrong geolocation, not a
subtle signature — worth surfacing as a data error.

This produces a LABEL, not a filter. Weak-signal events stay in the dataset:
they are exactly the cases no existing model handles, and stratifying results
by signal strength is how we show representation learning.

Usage:
    python pipeline/test_visual_signal.py
    python pipeline/test_visual_signal.py --event 5357 453
"""

import argparse
import json
import os
import warnings
from collections import defaultdict

import numpy as np
from PIL import Image

ALIGNED = 'Data/aligned_frames.json'
OUT_PATH = 'Data/visual_signal.json'

# Chips arrive at whatever pixel size Earth Engine returned (445x512 is
# typical, and it varies with latitude). Resampling to a fixed square makes
# one grid cell the same fraction of the footprint in every event, so cell
# counts are comparable across the dataset.
# Chips are written for display, with the gamma from download_imagery's
# RENDER_VERSION 2. Display gamma is for the model's eyes; a ratio of display
# values is not a ratio of reflectance, and measuring on it shrinks every
# proxy by roughly the gamma exponent. Undo it here so the proxies are
# physical again and the thresholds below mean the same thing whatever
# rendering we ship.
DISPLAY_GAMMA = 2.2

RESIZE = 256
GRID = 16                    # finest grid: ~0.7 km cells on an 11 km chip
SCALES = (1, 2, 4, 8, 16)    # chip-wide -> ~0.7 km
TOP_FRAC = 0.02              # score the most-changed 2% of cells, not the max

# Floor on pre-event variability, used when there is only one pre frame to
# compare against and when two pre frames happen to agree suspiciously well.
# Finer cells are noisier -- fewer pixels, and misregistration of half a cell
# matters more -- so the floor rises with scale.
NOISE_FLOOR = {1: 0.02, 2: 0.03, 4: 0.04, 8: 0.05, 16: 0.07}

# Which channel should move, and in which direction, per disaster type.
EXPECTED = {
    'Fire':             [('greenness', 'down'), ('brightness', 'down')],
    'Flood':            [('wetness', 'up'), ('texture', 'down')],
    'Hurricane':        [('wetness', 'up'), ('greenness', 'down')],
    'Tropical Storm':   [('wetness', 'up')],
    'Coastal Storm':    [('wetness', 'up')],
    'Severe Storm':     [('greenness', 'down'), ('texture', 'down')],
    'Tornado':          [('greenness', 'down'), ('texture', 'down')],
    'Snowstorm':        [('brightness', 'up'), ('texture', 'down')],
    'Severe Ice Storm': [('brightness', 'up'), ('greenness', 'down')],
    'Winter Storm':     [('brightness', 'up'), ('texture', 'down')],
    'Landslide':        [('greenness', 'down'), ('texture', 'up')],
    'Earthquake':       [('texture', 'up')],
    'Volcanic Eruption':[('brightness', 'down'), ('greenness', 'down')],
}

# Types whose signature is TRANSIENT — visible for a day or two, then gone.
# Sentinel-2 revisits every ~5 days, so averaging post-event frames washes the
# spike out entirely. For these we look for a single extreme frame instead.
EPHEMERAL_TYPES = {
    'Snowstorm', 'Severe Ice Storm', 'Winter Storm',
    'Flood', 'Coastal Storm', 'Tropical Storm', 'Hurricane',
}

# Types whose signature PERSISTS (burn scar, debris path, landslide scarp)
PERSISTENT_TYPES = {
    'Fire', 'Landslide', 'Tornado', 'Severe Storm',
    'Earthquake', 'Volcanic Eruption',
}


def _cell_stats(arr, valid, n=GRID):
    """Per-cell mean and std of `arr` on an n x n grid, NaN where no-data.

    A cell needs half its pixels valid to count; a chip edge clipped by the
    scene footprint would otherwise report the mean of three pixels.
    """
    h, w = arr.shape
    ch, cw = h // n, w // n
    a = arr[:ch * n, :cw * n].reshape(n, ch, n, cw)
    v = valid[:ch * n, :cw * n].reshape(n, ch, n, cw).astype(np.float32)
    cnt = v.sum(axis=(1, 3))
    s = (a * v).sum(axis=(1, 3))
    sq = ((a * a) * v).sum(axis=(1, 3))
    mean = np.full((n, n), np.nan, np.float32)
    std = np.full((n, n), np.nan, np.float32)
    ok = cnt > (ch * cw * 0.5)
    mean[ok] = s[ok] / cnt[ok]
    std[ok] = np.sqrt(np.maximum(sq[ok] / cnt[ok] - mean[ok] ** 2, 0.0))
    return mean, std


def _denom(pre):
    """Denominator for relative change, floored so near-zero cells cannot
    manufacture signal. Texture over flat water is ~0.001, and a shift to 0.02
    there is not a 1900% change in any meaningful sense -- it is a change of
    one fiftieth of what this scene's texture typically is. Floor at a quarter
    of the scene's own typical level."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        typical = float(np.nanmedian(np.abs(pre)))
    if not np.isfinite(typical):
        typical = 0.0
    return np.maximum(np.abs(pre), max(0.25 * typical, 1e-3))


def _contrast(grid):
    """A cell's value relative to the rest of ITS OWN frame.

    Sun angle, haze and the thumbnail's own stretch move every pixel of a
    Sentinel-2 chip together, and between two pre-event dates that swing is
    larger than most disasters: brightness noise floors of 0.45 and 1.33 in
    the first run, against real signals of a few percent. Dividing by the
    frame's own mean cancels the global term and leaves how a cell differs
    from its surroundings -- which is what a burn scar is.

    It cancels genuinely chip-wide change too (snow that whitens everything
    leaves every cell equally bright relative to its neighbours), so this is
    searched ALONGSIDE the absolute view, never instead of it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = float(np.nanmean(grid))
    if not np.isfinite(m) or abs(m) < 1e-9:
        return None
    return grid / m


def _pool(grid, k):
    """Coarsen a GRID x GRID map to k x k by averaging, ignoring no-data."""
    n = grid.shape[0]
    if k >= n:
        return grid
    f = n // k
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # all-NaN cells are expected
        return np.nanmean(grid.reshape(k, f, k, f), axis=(1, 3))


def _top_mean(x, frac=TOP_FRAC):
    """Mean of the largest `frac` of finite values -- a max that one bad cell
    cannot carry. At coarse scales this is just the single largest cell."""
    v = x[np.isfinite(x)]
    if v.size == 0:
        return None
    k = max(1, int(round(v.size * frac)))
    return float(np.sort(v)[-k:].mean())


def frame_metrics(path):
    """RGB-proxy indices for one frame: chip-wide scalars plus per-cell maps."""
    try:
        im = Image.open(path).convert('RGB')
    except Exception:
        return None
    if im.size[0] < GRID * 2 or im.size[1] < GRID * 2:
        return None
    im = im.resize((RESIZE, RESIZE), Image.BILINEAR)
    dn = np.asarray(im, dtype=np.float32)

    # Ignore no-data and saturated pixels so they don't dominate the means.
    # Thresholded on display values, where they were chosen.
    dn_lum = dn.mean(axis=2)
    valid = (dn_lum > 12) & (dn_lum < 245)
    if valid.mean() < 0.35:
        return None

    a = np.power(dn / 255.0, DISPLAY_GAMMA) * 255.0
    lum = a.mean(axis=2)

    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    total = np.clip(r + g + b, 1e-6, None)
    chans = {
        'greenness':  g / total,
        'wetness':    b / total,
        'brightness': lum / 255.0,
    }

    grids, scalars = {}, {}
    for name, arr in chans.items():
        mean, _ = _cell_stats(arr, valid)
        grids[name] = mean
        scalars[name] = float(arr[valid].mean())

    # Texture is roughness WITHIN a cell, so the cell's own std is the map and
    # the chip-wide number stays the spread of cell means it always was.
    cell_mean, cell_std = _cell_stats(lum / 255.0, valid)
    grids['texture'] = cell_std
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scalars['texture'] = float(np.nanstd(cell_mean))

    scalars['valid_frac'] = float(valid.mean())
    return {**scalars, 'grids': grids}


def _noise_grid(pre_grids):
    """How far a TYPICAL pre-event frame sits from the pre-event mean, per cell.

    This is the bar a real change has to clear: the post frame has to deviate
    more than the same ground deviates between two ordinary pre-event dates.

    It has to be a typical excursion, not the worst one. Taking a max over pre
    frames makes the bar rise with the number of pre frames and with the span
    they cover, so a fire with seven pre frames across two seasons would be
    held to a stricter standard than one with two frames a week apart -- and
    the burn scar, which is the most visible thing in the dataset, would score
    zero. With one pre frame there is nothing to measure and the caller falls
    back to NOISE_FLOOR.
    """
    if len(pre_grids) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanstd(np.stack(pre_grids), axis=0)


def scale_search(pre, post, noise, want, scales=SCALES):
    """Strongest change in the wanted direction, over every spatial scale.

    Returns the scale that maximises change-in-excess-of-noise, so a chip-wide
    snowfall and a four-cell burn scar are both found by the same search
    without either being scored on the other's footprint.
    """
    best = None
    for k in scales:
        a, b = _pool(pre, k), _pool(post, k)
        denom = _denom(a)
        rel = (b - a) / denom
        if want == 'down':
            rel = -rel
        signal = _top_mean(rel)
        if signal is None:
            continue
        floor = NOISE_FLOOR[k]
        if noise is not None:
            nrel = _top_mean(_pool(noise, k) / denom)
            floor = max(floor, nrel if nrel is not None else 0.0)
        net = signal - floor
        # Fraction of the chip that moved in the wanted direction by more than
        # the noise -- how big the change was, as distinct from how strong.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fin = np.isfinite(rel)
            area = float((rel[fin] > floor).mean()) if fin.any() else 0.0
        if best is None or net > best['net']:
            best = {'net': net, 'raw': signal, 'floor': floor,
                    'cells': k, 'area_frac': area}
    return best


def analyze_event(ev):
    etype = ev.get('type', '')
    frames = ev.get('frames', [])

    pre, post = [], []
    for fr in frames:
        m = frame_metrics(fr['path'])
        if not m:
            continue
        entry = {'date': fr['date'], 'phase': fr['phase'], **m}
        if fr['phase'] == 'pre-event':
            pre.append(entry)
        elif fr['phase'] in ('onset', 'during', 'post'):
            post.append(entry)

    if not pre or not post:
        return {
            'event_id': ev['event_id'], 'type': etype,
            'verdict': 'insufficient_frames',
            'n_pre': len(pre), 'n_post': len(post),
        }

    def mean_of(rows, k):
        return float(np.mean([r[k] for r in rows]))

    def combine(grids):
        """Median, not mean: thin haze passes the frame-quality gate and one
        hazy frame in a post-event set drags the whole composite the wrong
        way. A median ignores it as long as most frames are clear."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.nanmedian(np.stack(grids), axis=0)

    ephemeral = etype in EPHEMERAL_TYPES
    expected = EXPECTED.get(etype, [])

    # Chip-wide numbers, kept for reporting and for comparison against the
    # localized result -- the gap between them is the dilution.
    deltas, rel = {}, {}
    for k in ['greenness', 'wetness', 'brightness', 'texture']:
        base = mean_of(pre, k)
        val = mean_of(post, k)
        deltas[k] = val - base
        rel[k] = (val - base) / max(abs(base), 1e-6)

    def views(grids):
        """(absolute, contrast) versions of a list of per-cell maps."""
        con = [_contrast(g) for g in grids]
        return grids, (None if any(c is None for c in con) else con)

    def search(pre_g, post_g, noise, want, mode):
        # The absolute view sees chip-wide change; the contrast view sees
        # localized change and is blind to illumination. Neither alone covers
        # both an ice storm and a burn scar.
        scales = SCALES if mode == 'absolute' else SCALES[1:]
        r = scale_search(pre_g, post_g, noise, want, scales)
        if r is not None:
            r['mode'] = mode
        return r

    def better(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return a if a['net'] >= b['net'] else b

    checks = []
    for chan, want in expected:
        opp = 'down' if want == 'up' else 'up'
        pre_abs, pre_con = views([r['grids'][chan] for r in pre])
        post_raw = [(r['date'], r['grids'][chan]) for r in post]

        modes = [('absolute', pre_abs, [g for _, g in post_raw])]
        if pre_con is not None:
            post_con = [_contrast(g) for _, g in post_raw]
            if not any(c is None for c in post_con):
                modes.append(('contrast', pre_con, post_con))

        hit = miss = peak = None
        for mode, pg, pv in modes:
            base = combine(pg)
            noise = _noise_grid(pg)
            if ephemeral:
                # Transient signature: the strongest single post frame, not
                # the mean. Snow and standing water are gone within a day or
                # two, and Sentinel-2's ~5-day revisit means most post frames
                # miss it entirely -- averaging washes the spike out. Score
                # each frame in both directions and keep the most extreme, so
                # a frame is never selected for moving the way we hoped.
                for (d, _), g in zip(post_raw, pv):
                    a = search(base, g, noise, want, mode)
                    b = search(base, g, noise, opp, mode)
                    if a is None or b is None:
                        continue
                    if hit is None or max(a['net'], b['net']) > max(hit['net'], miss['net']):
                        hit, miss, peak = a, b, d
            else:
                a = search(base, combine(pv), noise, want, mode)
                b = search(base, combine(pv), noise, opp, mode)
                if a is None or b is None:
                    continue
                hit, miss = better(hit, a), better(miss, b)

        if hit is None or miss is None:
            continue
        matched = hit['net'] >= miss['net']
        win = hit if matched else miss
        c = {'channel': chan, 'expected': want,
             'observed': want if matched else opp,
             'delta': round(deltas[chan], 5),
             'rel_change': round(rel[chan], 4),
             'match': matched,
             'magnitude': round(min(max(win['net'], 0.0), 1.0), 4),
             'chip_wide': round(min(abs(rel[chan]), 1.0), 4),
             'scale_cells': win['cells'],
             'mode': win['mode'],
             'area_frac': round(win['area_frac'], 4),
             'noise_floor': round(win['floor'], 4)}
        if peak:
            c['peak_frame'] = peak
        checks.append(c)

    if not checks:
        verdict, strength, win = 'no_expectation', 0.0, None
    else:
        matched = [c for c in checks if c['match']]
        win = max(matched, key=lambda c: c['magnitude'], default=None)
        strength = win['magnitude'] if win else 0.0
        strong_wrong = [c for c in checks
                        if not c['match'] and c['magnitude'] > 0.15]
        if not matched and strong_wrong:
            verdict = 'CONTRADICTED'
        elif strength >= 0.15:
            # A single pre-event frame gives nothing to measure ordinary
            # variation against, so a cloud edge and a burn scar look alike.
            # Cap the claim rather than dropping the event.
            verdict = 'strong' if len(pre) >= 2 else 'weak'
        elif strength >= 0.05:
            verdict = 'weak'
        else:
            verdict = 'none'

    # How close did we actually get to the event? An ephemeral signature is
    # only observable if a frame landed within a couple of days of onset.
    nearest_gap = None
    nd = ev.get('notable_dates') or {}
    onset = nd.get('start') or ev.get('fema_start')
    if onset:
        from datetime import datetime
        try:
            o = datetime.strptime(onset, '%Y-%m-%d')
            gaps = []
            for r in post:
                try:
                    gaps.append(abs((datetime.strptime(r['date'], '%Y-%m-%d') - o).days))
                except Exception:
                    pass
            nearest_gap = min(gaps) if gaps else None
        except Exception:
            pass

    if (ephemeral and verdict in ('none', 'weak')
            and nearest_gap is not None and nearest_gap > 3):
        verdict = 'unobserved_transient'

    return {
        'event_id': ev['event_id'], 'type': etype,
        'ephemeral': ephemeral,
        'n_pre': len(pre), 'n_post': len(post),
        'nearest_post_frame_gap_days': nearest_gap,
        'deltas': {k: round(v, 5) for k, v in deltas.items()},
        'rel_change': {k: round(v, 4) for k, v in rel.items()},
        'checks': checks,
        'signal_strength': round(strength, 4),
        # The scale the signal was found at, and how much of the chip moved.
        # A 16-cell answer means a localized scar; a 1-cell answer means the
        # whole footprint shifted. Both are real; they are different events.
        'scale_cells': win['scale_cells'] if win else None,
        'area_frac': win['area_frac'] if win else None,
        # What a chip-wide mean would have scored, for comparison.
        'chip_wide_strength': win['chip_wide'] if win else 0.0,
        'verdict': verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aligned', default=ALIGNED)
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--event', nargs='+', type=int, default=None)
    args = ap.parse_args()

    aligned = json.load(open(args.aligned))
    keys = sorted(aligned, key=lambda k: int(k))
    if args.event:
        keys = [k for k in keys if int(k) in args.event]

    results = {}
    for k in keys:
        results[k] = analyze_event(aligned[k])

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'{"event":>7} {"type":<18} {"pre/post":>9} {"strength":>9} '
          f'{"chipwide":>9} {"grid":>5} {"area":>6}  verdict')
    print('-' * 88)
    for k in keys:
        r = results[k]
        cells = r.get('scale_cells')
        area = r.get('area_frac')
        print(f'{r["event_id"]:>7} {str(r["type"])[:18]:<18} '
              f'{r.get("n_pre",0):>4}/{r.get("n_post",0):<4} '
              f'{r.get("signal_strength",0):>9.3f} '
              f'{r.get("chip_wide_strength",0):>9.3f} '
              f'{(str(cells)+"x"+str(cells)) if cells else "-":>5} '
              f'{(f"{area:.1%}" if area is not None else "-"):>6}  {r["verdict"]}')

    counts = defaultdict(int)
    for r in results.values():
        counts[r['verdict']] += 1
    print(f'\nverdicts: {dict(counts)}')

    # Events with one pre frame have no measure of ordinary variation, so
    # their scores are not comparable with the rest and would distort a mean.
    scored = [r for r in results.values()
              if 'signal_strength' in r and r.get('n_pre', 0) >= 2]
    n_thin = sum(1 for r in results.values()
                 if 'signal_strength' in r and r.get('n_pre', 0) < 2)
    by_type = defaultdict(list)
    for r in scored:
        by_type[r['type']].append(r['signal_strength'])
    if by_type:
        print('\nmean signal strength by type '
              '(localized vs what a chip-wide mean would have seen):')
        chip = defaultdict(list)
        for r in scored:
            chip[r['type']].append(r.get('chip_wide_strength', 0.0))
        for t, xs in sorted(by_type.items(), key=lambda x: -np.mean(x[1])):
            print(f'   {t:<20} {np.mean(xs):.3f}  '
                  f'(chip-wide {np.mean(chip[t]):.3f})  (n={len(xs)})')
    if n_thin:
        print(f'\n{n_thin} events excluded from the table: one pre-event '
              f'frame, so nothing to measure ordinary variation against.')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
