"""
Dataset-wide quality checks.

Every stage asserts on its inputs rather than trusting them. This exists
because a 25% coordinate error in the inherited events file went undetected
through several rounds of downstream validation — the checks were all on
derived artifacts (features, questions, signal) while the foundation was
assumed correct.

Principle: cheap assertions on upstream data beat sophisticated checks on
derived data.

Checks, by stage:

  EVENTS    center inside declared county; halfwidth sane; dates ordered;
            required fields present
  HARVEST   relevance rate plausible; articles dated within window; scope
            distribution; extent magnitudes reasonable for the type
  IMAGERY   frame count; dates within window; frames actually differ from
            each other; no duplicate images across events
  ALIGNMENT phase coverage; bounds monotonic where required
  QA        answer-position balance; blind-answerable heuristics; duplicate
            questions; split leakage

Usage:
    python pipeline/quality_checks.py                # all stages
    python pipeline/quality_checks.py --stage events
    python pipeline/quality_checks.py --fail-fast    # exit 1 on first FAIL
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_timeseries import _canonical_extent

EVENTS_PATH = 'Data/events_processed.json'
HARVEST_DIR = 'Data/harvest'
IMAGES_DIR = 'Data/images'
ALIGNED_PATH = 'Data/aligned_frames.json'
SIGNAL_PATH = 'Data/visual_signal.json'
QA_GLOB = 'Data/qa_visual_mcq*.json'
COUNTY_CACHE = 'Data/county_geo.json'

RESULTS = []


def report(stage, name, status, detail='', n=None):
    RESULTS.append((stage, name, status, detail, n))
    mark = {'PASS': ' ok ', 'WARN': 'warn', 'FAIL': 'FAIL', 'SKIP': 'skip'}[status]
    count = f'  [{n}]' if n is not None else ''
    print(f'  [{mark}] {name}{count}')
    if detail and status != 'PASS':
        for line in str(detail).split('\n')[:6]:
            print(f'         {line}')


def _d(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except Exception:
        return None


def haversine_km(a, b):
    R = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(h))


# ── EVENTS ──────────────────────────────────────────────────────────────────

def check_events(max_km=150.0):
    print('\nEVENTS')
    if not os.path.exists(EVENTS_PATH):
        report('events', 'file present', 'SKIP', EVENTS_PATH); return
    ev = json.load(open(EVENTS_PATH))
    valid = {k: v for k, v in ev.items() if 'error' not in v}

    # required fields
    missing = defaultdict(list)
    for k, v in valid.items():
        for f in ('type', 'state', 'county', 'start_date', 'center'):
            if not v.get(f):
                missing[f].append(k)
    if missing:
        report('events', 'required fields', 'FAIL',
               '\n'.join(f'{f}: {len(ids)} events missing'
                         for f, ids in missing.items()), len(valid))
    else:
        report('events', 'required fields', 'PASS', n=len(valid))

    # dates ordered
    bad_dates = [k for k, v in valid.items()
                 if _d(v.get('start_date')) and _d(v.get('end_date'))
                 and _d(v['end_date']) < _d(v['start_date'])]
    report('events', 'start_date <= end_date',
           'FAIL' if bad_dates else 'PASS',
           f'{len(bad_dates)} reversed: {bad_dates[:5]}' if bad_dates else '',
           len(valid))

    # halfwidth sane
    hw = [v.get('halfwidth', 0.05) for v in valid.values()]
    odd = [h for h in hw if not (0.01 <= h <= 0.5)]
    report('events', 'halfwidth in [0.01, 0.5]',
           'FAIL' if odd else 'PASS',
           f'{len(odd)} outside range, e.g. {odd[:5]}' if odd else '', len(hw))

    # THE check that was missing: center inside declared county
    cache = json.load(open(COUNTY_CACHE)) if os.path.exists(COUNTY_CACHE) else {}
    if not cache:
        report('events', 'center within county', 'SKIP',
               'no county cache — run: python pipeline/fix_centers.py --audit')
        return
    try:
        from fix_centers import county_geo_cached, in_county
    except Exception as e:
        report('events', 'center within county', 'SKIP', str(e)); return

    # Two different questions, deliberately held apart.
    #
    # Outside the county boundary is a placement we tolerate: an LLM estimate
    # that lands just over the line still images ground near the event, and
    # replacing it with a county centroid can put the chip further away than
    # the estimate it discards. We accept ~8% of these.
    #
    # Hundreds of km away is never tolerable — that was the Fajardo-in-Georgia
    # bug — so it stays a hard failure at any rate above zero.
    outside, gross, checked = [], [], 0
    for k, v in valid.items():
        centroid, bbox = county_geo_cached(v.get('county'), v.get('state'),
                                           cache)
        if centroid is None:
            continue
        checked += 1
        d = haversine_km(tuple(v['center'][:2]), centroid)
        if d > max_km:
            gross.append((k, round(d)))
        if not in_county(v['center'], centroid, bbox, max_km):
            outside.append((k, round(d)))
    if checked:
        pct = 100 * len(outside) / checked
        status = 'PASS' if pct < 5 else ('WARN' if pct < 10 else 'FAIL')
        worst = sorted(outside, key=lambda x: -x[1])[:5]
        report('events', 'center inside county boundary', status,
               f'{len(outside)}/{checked} ({pct:.1f}%) outside; '
               f'accepted below 10%. worst: {worst}' if outside else '',
               checked)
        report('events', f'no center more than {max_km:.0f}km from its county',
               'PASS' if not gross else 'FAIL',
               f'{len(gross)} grossly misplaced: '
               f'{sorted(gross, key=lambda x: -x[1])[:5]}' if gross else '',
               checked)


# ── HARVEST ─────────────────────────────────────────────────────────────────

def check_harvest():
    print('\nHARVEST')
    files = sorted(glob.glob(os.path.join(HARVEST_DIR, '*.json')))
    if not files:
        report('harvest', 'records present', 'SKIP', HARVEST_DIR); return

    recs = []
    for p in files:
        try:
            recs.append(json.load(open(p)))
        except Exception:
            pass
    report('harvest', 'records readable', 'PASS', n=len(recs))

    # relevance rate — both extremes are bugs
    rates = []
    for r in recs:
        facts = [f for f in r.get('facts', []) if not f.get('error')]
        if len(facts) >= 3:
            rel = sum(1 for f in facts if f.get('is_about_target_event'))
            rates.append(rel / len(facts))
    if rates:
        mean = sum(rates) / len(rates)
        zero = sum(1 for x in rates if x == 0.0)
        status = 'PASS' if 0.15 <= mean <= 0.9 else 'WARN'
        report('harvest', 'relevance rate plausible', status,
               f'mean {mean:.2f}; {zero}/{len(rates)} events rejected everything. '
               f'Near 0 = gate too strict; near 1 = gate not filtering.',
               len(rates))

    # articles dated inside the event window
    out_of_window = 0
    total_dated = 0
    for r in recs:
        s, e = _d(r.get('fema_start')), _d(r.get('fema_end'))
        if not s:
            continue
        for f in r.get('facts', []):
            if not f.get('is_about_target_event'):
                continue
            d = _d(f.get('pub_date'))
            if not d:
                continue
            total_dated += 1
            lo = s.timestamp() - 90 * 86400
            hi = (e or s).timestamp() + 365 * 86400
            if not (lo <= d.timestamp() <= hi):
                out_of_window += 1
    if total_dated:
        pct = 100 * out_of_window / total_dated
        report('harvest', 'accepted articles near event window',
               'PASS' if pct < 20 else 'WARN',
               f'{out_of_window}/{total_dated} ({pct:.0f}%) outside '
               f'[-90d, +365d]; retrospectives inflate timeseries tails',
               total_dated)

    # extent magnitudes sane per type
    LIMITS = {'acres': 3_000_000, 'sq_miles': 50_000,
              'structures': 200_000, 'homes': 200_000}
    absurd, salvaged, unusable = [], [], []
    for r in recs:
        for f in r.get('facts', []):
            v, u = f.get('extent_number'), f.get('extent_unit')
            if v is None:
                continue
            # The LLM sometimes returns a range ([500, 1000]) or a string
            # rather than a scalar. build_timeseries._canonical_extent already
            # normalizes those (a range collapses to its upper end), so the
            # question is not "is it a scalar" but "can it be read as one".
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                canon = _canonical_extent(v, u)
                if canon is None:
                    unusable.append((r['event_id'], repr(v)[:40], u))
                else:
                    salvaged.append((r['event_id'], repr(v)[:40], canon[0]))
                continue
            if u in LIMITS and v > LIMITS[u]:
                absurd.append((r['event_id'], v, u))
    if unusable:
        status, detail = 'FAIL', (f'{len(unusable)} values cannot be read as a '
                                  f'number and are dropped: {unusable[:4]}')
    elif salvaged:
        status, detail = 'WARN', (f'{len(salvaged)} non-scalar values '
                                  f'normalized by _canonical_extent: '
                                  f'{salvaged[:4]}')
    else:
        status, detail = 'PASS', ''
    report('harvest', 'extent_number readable as a number', status, detail)
    report('harvest', 'extent magnitudes plausible',
           'PASS' if not absurd else 'WARN',
           f'{len(absurd)} implausible: {absurd[:4]}' if absurd else '')

    # scope distribution
    scopes = Counter(f.get('extent_scope') for r in recs
                     for f in r.get('facts', []) if f.get('extent_number'))
    if scopes:
        non_local = sum(v for k, v in scopes.items() if k not in (None, 'local'))
        report('harvest', 'recorded extents are local-scope',
               'PASS' if non_local == 0 else 'FAIL',
               f'{non_local} non-local extents recorded: {dict(scopes)}'
               if non_local else '')


# ── IMAGERY ─────────────────────────────────────────────────────────────────

def check_imagery():
    print('\nIMAGERY')
    if not os.path.isdir(IMAGES_DIR):
        report('imagery', 'directory present', 'SKIP', IMAGES_DIR); return
    dirs = [d for d in os.listdir(IMAGES_DIR)
            if os.path.isdir(os.path.join(IMAGES_DIR, d))]
    if not dirs:
        report('imagery', 'events with imagery', 'SKIP', 'none'); return

    counts, all_hashes = {}, defaultdict(list)
    for d in dirs:
        p = os.path.join(IMAGES_DIR, d)
        frames = [f for f in os.listdir(p) if f.endswith(('.png', '.jpg'))]
        counts[d] = len(frames)
        for f in frames[:40]:
            try:
                with open(os.path.join(p, f), 'rb') as fh:
                    all_hashes[hashlib.md5(fh.read()).hexdigest()].append(f'{d}/{f}')
            except Exception:
                pass

    report('imagery', 'events with frames', 'PASS', n=len(counts))
    thin = [d for d, n in counts.items() if n < 3]
    report('imagery', 'at least 3 frames per event',
           'PASS' if not thin else 'WARN',
           f'{len(thin)} events with <3 frames: {thin[:8]}' if thin else '',
           len(counts))

    # Identical frames across events. Two cases, and only one is a defect.
    #
    # Different counties sharing a chip means a center is wrong — that is how
    # the pre-relocation coordinates surfaced, and it stays a hard failure.
    #
    # The same county sharing a chip is expected: FEMA declares one county for
    # several disasters, sometimes on the same day (Nogal Canyon and McBride
    # both hit Lincoln County NM on 2022-04-12), and one chip covers both.
    # County-level holdout keeps these on the same side of the split, so it is
    # duplicate supervision rather than leakage — a deduplication decision.
    ev_meta = {}
    if os.path.exists(EVENTS_PATH):
        try:
            ev_meta = json.load(open(EVENTS_PATH))
        except Exception:
            pass

    def county_of(d):
        v = ev_meta.get(d.replace('event_', '').replace('ev', ''), {})
        return (v.get('state'), v.get('county'))

    cross_county, same_county = {}, {}
    for h, v in all_hashes.items():
        eids = {x.split('/')[0] for x in v}
        if len(eids) < 2:
            continue
        counties = {county_of(e) for e in eids}
        if len(counties) > 1 or counties == {(None, None)}:
            cross_county[h] = v
        else:
            same_county[h] = v

    report('imagery', 'no identical frames across counties',
           'PASS' if not cross_county else 'FAIL',
           f'{len(cross_county)} images shared between events in DIFFERENT '
           f'counties — a center is wrong. '
           f'e.g. {list(cross_county.values())[0][:3]}' if cross_county else '')

    if same_county:
        pairs = set()
        for v in same_county.values():
            e = sorted({x.split('/')[0] for x in v})
            for i in range(len(e)):
                for j in range(i + 1, len(e)):
                    pairs.add((e[i], e[j]))
        ex = ', '.join(f'{a}/{b}' for a, b in sorted(pairs)[:5])
        report('imagery', 'no duplicate chips within a county', 'WARN',
               f'{len(same_county)} images shared by {len(pairs)} event pairs '
               f'in the same county ({ex}). Same ground, two declarations — '
               f'duplicate supervision, not leakage. Run '
               f'find_identical_frames.py to decide which to keep.')
    else:
        report('imagery', 'no duplicate chips within a county', 'PASS')

    # identical frames within an event => nothing changed, or a stuck download
    dupe_within = 0
    for h, v in all_hashes.items():
        if len(v) > 1 and len({x.split('/')[0] for x in v}) == 1:
            dupe_within += 1
    report('imagery', 'no duplicate frames within an event',
           'PASS' if dupe_within == 0 else 'WARN',
           f'{dupe_within} duplicated frames' if dupe_within else '')

    # frame dates inside event window
    if os.path.exists(EVENTS_PATH):
        ev = json.load(open(EVENTS_PATH))
        bad = []
        for d, _ in counts.items():
            e = ev.get(d)
            if not e or 'error' in e:
                continue
            s = _d(e.get('start_date'))
            if not s:
                continue
            for f in os.listdir(os.path.join(IMAGES_DIR, d)):
                m = re.search(r'(\d{4}-\d{2}-\d{2})', f)
                if not m:
                    continue
                fd = _d(m.group(1))
                if fd and abs((fd - s).days) > 365:
                    bad.append(f'{d}/{m.group(1)}')
        report('imagery', 'frame dates within a year of event',
               'PASS' if not bad else 'WARN',
               f'{len(bad)} far-off frames: {bad[:5]}' if bad else '')


# ── ALIGNMENT ───────────────────────────────────────────────────────────────

def check_alignment():
    print('\nALIGNMENT')
    if not os.path.exists(ALIGNED_PATH):
        report('alignment', 'file present', 'SKIP', ALIGNED_PATH); return
    aligned = json.load(open(ALIGNED_PATH))
    report('alignment', 'events aligned', 'PASS', n=len(aligned))

    phases = Counter(fr['phase'] for v in aligned.values() for fr in v['frames'])
    n_frames = sum(len(v['frames']) for v in aligned.values())
    unknown = phases.get('unknown', 0)
    report('alignment', 'frames have a resolved phase',
           'PASS' if unknown / max(1, n_frames) < 0.1 else 'WARN',
           f'{unknown}/{n_frames} unknown; distribution {dict(phases)}', n_frames)

    both = sum(1 for v in aligned.values() for fr in v['frames']
               if fr['phase'] in ('pre-event',) )
    post = sum(1 for v in aligned.values() for fr in v['frames']
               if fr['phase'] in ('onset', 'during', 'post'))
    report('alignment', 'have both pre- and post-event frames',
           'PASS' if both and post else 'WARN',
           f'pre={both} post={post}')

    # monotonic extent for fires
    bad = []
    for k, v in aligned.items():
        if (v.get('type') or '') != 'Fire':
            continue
        vals = [r['value'] for r in v.get('extent_timeseries', [])]
        if any(b < a for a, b in zip(vals, vals[1:])):
            bad.append(k)
    report('alignment', 'fire extent monotonic',
           'PASS' if not bad else 'FAIL',
           f'{len(bad)} decreasing: {bad[:5]}' if bad else '')

    # bounds ordered
    bad_bounds = []
    for k, v in aligned.items():
        for fr in v['frames']:
            lo, hi = fr['bounds'].get('lower'), fr['bounds'].get('upper')
            if lo and hi and lo.get('unit') == hi.get('unit') \
               and lo['value'] > hi['value']:
                bad_bounds.append(f"{k}@{fr['date']}")
    report('alignment', 'lower bound <= upper bound',
           'PASS' if not bad_bounds else 'FAIL',
           f'{len(bad_bounds)} inverted: {bad_bounds[:5]}' if bad_bounds else '')


# ── QA ──────────────────────────────────────────────────────────────────────

NEG = re.compile(r'no (significant|discernible|noticeable|visible)|unchanged|'
                 r'insufficient evidence|cannot be determined', re.I)


def check_qa():
    print('\nQA')
    files = [f for f in glob.glob(QA_GLOB)
             if not f.endswith(('_train.json', '_test.json'))]
    if not files:
        report('qa', 'file present', 'SKIP', QA_GLOB); return
    qa = json.load(open(files[0]))
    report('qa', 'questions present', 'PASS', n=len(qa))

    # answer-position balance
    dist = Counter(q['conversations'][1]['value'].strip().lower() for q in qa)
    tot = sum(dist.values())
    if tot:
        worst = max(dist.values()) / tot
        report('qa', 'answer positions balanced',
               'PASS' if worst < 0.32 else ('WARN' if worst < 0.4 else 'FAIL'),
               f'most common {worst:.1%}; {dict(sorted(dist.items()))}', tot)

    # duplicates
    stems = Counter(q['conversations'][0]['value'].split('\n')[0] for q in qa)
    dupes = {k: v for k, v in stems.items() if v > 1}
    report('qa', 'no duplicate question stems',
           'PASS' if not dupes else 'WARN',
           f'{len(dupes)} repeated, e.g. {list(dupes)[0][:70]}' if dupes else '')

    # null-finding keys
    nulls = 0
    for q in qa:
        opts = re.findall(r'^([a-d])\. (.+)$',
                          q['conversations'][0]['value'], re.M)
        gold = q['conversations'][1]['value'].strip().lower()
        for letter, text in opts:
            if letter == gold and NEG.search(text):
                nulls += 1
    report('qa', 'no null-finding correct answers',
           'PASS' if nulls == 0 else 'FAIL',
           f'{nulls} questions answer "nothing changed"' if nulls else '')

    # split leakage
    tr = files[0].replace('.json', '_train.json')
    te = files[0].replace('.json', '_test.json')
    if os.path.exists(tr) and os.path.exists(te):
        a = {q['event_id'] for q in json.load(open(tr))}
        b = {q['event_id'] for q in json.load(open(te))}
        report('qa', 'no event in both splits',
               'PASS' if not (a & b) else 'FAIL',
               f'{len(a & b)} shared: {sorted(a & b)[:5]}' if a & b else '',
               len(a) + len(b))
        if os.path.exists(EVENTS_PATH):
            ev = json.load(open(EVENTS_PATH))
            ck = lambda i: (f"{ev.get(str(i),{}).get('state')}::"
                            f"{ev.get(str(i),{}).get('county')}")
            ca, cb = {ck(i) for i in a}, {ck(i) for i in b}
            report('qa', 'no county in both splits',
                   'PASS' if not (ca & cb) else 'FAIL',
                   f'{len(ca & cb)} shared counties: {sorted(ca & cb)[:3]}'
                   if ca & cb else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['events', 'harvest', 'imagery',
                                        'alignment', 'qa', 'all'], default='all')
    ap.add_argument('--max-km', type=float, default=150.0)
    ap.add_argument('--fail-fast', action='store_true')
    args = ap.parse_args()

    print('=' * 70)
    print('MONITRS quality checks')
    print('=' * 70)

    stages = {'events': lambda: check_events(args.max_km),
              'harvest': check_harvest, 'imagery': check_imagery,
              'alignment': check_alignment, 'qa': check_qa}
    for name, fn in stages.items():
        if args.stage in (name, 'all'):
            try:
                fn()
            except Exception as e:
                import traceback
                tb = traceback.format_exc().strip().split('\n')
                # Last frame before the exception line tells us the real site
                where = [l.strip() for l in tb if l.strip().startswith('File')]
                report(name, 'check crashed', 'FAIL',
                       f'{type(e).__name__}: {e}\n' +
                       ('\n'.join(where[-2:]) if where else ''))
                if os.environ.get('QC_TRACEBACK'):
                    print('\n'.join(tb))
            if args.fail_fast and any(r[2] == 'FAIL' for r in RESULTS):
                break

    counts = Counter(r[2] for r in RESULTS)
    print('\n' + '=' * 70)
    print(f'{counts.get("PASS",0)} pass, {counts.get("WARN",0)} warn, '
          f'{counts.get("FAIL",0)} fail, {counts.get("SKIP",0)} skipped')
    fails = [r for r in RESULTS if r[2] == 'FAIL']
    if fails:
        print('\nFAILURES:')
        for stage, name, _, detail, _ in fails:
            print(f'  {stage}: {name}')
            if detail:
                print(f'      {str(detail).splitlines()[0][:100]}')
    print('=' * 70)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
