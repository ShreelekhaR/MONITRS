"""
Re-estimate event centers with the declared county as a hard constraint.

The v1 geolocation step produced centers outside their own county for 32% of
events (3,139 of 9,747 checkable) — Minnesota counties placed in Puerto Rico,
Delaware in New Mexico. Errors concentrate in hurricane states (LA 364, NC 322,
FL 276, GA 269), where one storm hits dozens of counties and the model appears
to have returned wherever the storm was most discussed rather than the county
FEMA declared.

The fix is not a better prompt alone — it is validation. FEMA gives the county
authoritatively, so any proposal outside its bounding box is rejected outright
and retried. What the LLM contributes is placement WITHIN the county, which a
centroid cannot provide.

Order of preference per event:
    1. FIRMS thermal detections (fires) — measured, not estimated
    2. LLM proposal that falls inside the county bbox
    3. Nearest point inside the county to a near-miss proposal (<50 km out)
    4. County centroid, flagged as low confidence

Usage:
    export GCP_PROJECT_ID=ai-sandbox-dev-f139
    python pipeline/relocate_events.py --audit
    python pipeline/relocate_events.py --repair --limit 50     # try a batch
    python pipeline/relocate_events.py --repair
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_centers import (county_geo, haversine_km, clean_county, STATE_NAMES)

EVENTS_PATH = 'Data/events_processed.json'
CACHE_PATH = 'Data/county_geo.json'
THRESHOLD_KM = 150.0


PROMPT = """Locate the centre of a natural disaster's impact WITHIN a specific county.

EVENT
  name:   {event_name}
  type:   {event_type}
  county: {county_full}, {state_full}
  dates:  {start_date} to {end_date}

The county is authoritative — FEMA declared this disaster for {county_full}
specifically. Your task is only to place the impact centre INSIDE it.

The county spans:
  latitude  {south} to {north}
  longitude {west} to {east}

Return STRICT JSON only:

{{"lat": 0.0, "lon": 0.0, "place": "", "confidence": "high|medium|low",
  "reasoning": ""}}

Rules:
- lat MUST be between {south} and {north}. lon MUST be between {west} and {east}.
  A coordinate outside this box is invalid and will be rejected.
- Aim for where the impact was concentrated: the town or watershed most affected,
  the burn area, the flooded river reach. Not simply the geographic middle.
- "place" names the location you chose, e.g. "near Ashland City" or
  "the Cumberland River floodplain".
- confidence "high" only if you know this specific event and where it hit;
  "medium" if reasoning from the disaster type and county geography;
  "low" if guessing.
- If you have no basis for preferring one part of the county, say so in
  reasoning and return the county's geographic centre.

Output only the JSON."""


def call_llm(client, prompt, model_id):
    from google.genai import types
    try:
        cfg = types.GenerateContentConfig(
            max_output_tokens=350, temperature=0.0,
            response_mime_type='application/json',
            thinking_config=types.ThinkingConfig(thinking_budget=0))
    except Exception:
        cfg = types.GenerateContentConfig(
            max_output_tokens=350, temperature=0.0,
            response_mime_type='application/json')
    try:
        r = client.models.generate_content(model=model_id, contents=prompt,
                                           config=cfg)
        return r.text or ''
    except Exception as e:
        return f'[ERR: {e}]'


def clamp_to_bbox(lat, lon, bbox):
    """Nearest point inside the county box."""
    s, n, w, e = bbox
    return min(max(lat, s), n), min(max(lon, w), e)


def relocate(eid, ev, client, model_id, cache, max_retries=2):
    """Return (center, strategy, note)."""
    centroid, bbox = county_geo(ev.get('county'), ev.get('state'), cache)
    if centroid is None:
        return None, None, 'county not geocodable'

    # 1. FIRMS is measured, not estimated — always prefer it when in-county
    firms = ev.get('firms')
    if isinstance(firms, dict):
        fc = firms.get('center') or firms.get('centroid')
        if fc and len(fc) == 2 and bbox:
            s, n, w, e = bbox
            if s <= fc[0] <= n and w <= fc[1] <= e:
                return [fc[0], fc[1]], 'firms', 'thermal detections in county'

    if not bbox:
        return list(centroid), 'county_centroid', 'no county bbox available'

    s, n, w, e = bbox
    bare, desig = clean_county(ev.get('county'))
    prompt = (PROMPT
              .replace('{event_name}', str(ev.get('event')))
              .replace('{event_type}', str(ev.get('type')))
              .replace('{county_full}', f'{bare} {desig}')
              .replace('{state_full}', STATE_NAMES.get(
                  (ev.get('state') or '').upper(), ev.get('state') or ''))
              .replace('{start_date}', str(ev.get('start_date')))
              .replace('{end_date}', str(ev.get('end_date')))
              .replace('{south}', f'{s:.4f}').replace('{north}', f'{n:.4f}')
              .replace('{west}', f'{w:.4f}').replace('{east}', f'{e:.4f}'))

    best_near = None
    for _ in range(max_retries):
        raw = call_llm(client, prompt, model_id)
        if raw.startswith('[ERR'):
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                continue
            try:
                d = json.loads(m.group(0))
            except Exception:
                continue
        try:
            lat, lon = float(d['lat']), float(d['lon'])
        except Exception:
            continue

        if s <= lat <= n and w <= lon <= e:
            return [lat, lon], 'llm_in_county', d.get('place') or ''

        # Near miss: keep the closest for a possible clamp
        km = haversine_km((lat, lon), centroid)
        if km <= 50 and (best_near is None or km < best_near[2]):
            best_near = (lat, lon, km, d.get('place') or '')

    if best_near:
        lat, lon = clamp_to_bbox(best_near[0], best_near[1], bbox)
        return [lat, lon], 'llm_clamped', \
               f'{best_near[3]} (clamped from {best_near[2]:.0f}km out)'

    return list(centroid), 'county_centroid', 'no valid in-county proposal'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default=EVENTS_PATH)
    ap.add_argument('--cache', default=CACHE_PATH)
    ap.add_argument('--model', default='gemini-2.5-flash')
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--repair', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-km', type=float, default=THRESHOLD_KM)
    args = ap.parse_args()
    if not (args.audit or args.repair):
        args.audit = True

    events = json.load(open(args.events))
    if not os.path.exists(args.cache):
        print(f'missing {args.cache} — run: python pipeline/fix_centers.py --audit')
        return
    cache = json.load(open(args.cache))

    broken = []
    for k, v in events.items():
        if 'error' in v or not v.get('center'):
            continue
        c, _ = county_geo(v.get('county'), v.get('state'), cache)
        if c is None:
            continue
        if haversine_km(tuple(v['center'][:2]), c) > args.max_km:
            broken.append(k)

    print(f'{len(broken)} events with centers outside their county')
    if args.audit and not args.repair:
        print(f'by state: {dict(Counter(events[k].get("state") for k in broken).most_common(10))}')
        print('\nRe-run with --repair to re-estimate with county as a hard constraint.')
        return

    targets = broken[:args.limit] if args.limit else broken
    print(f'Relocating {targets and len(targets)} events '
          f'(model={args.model}, workers={args.workers})\n')

    from google import genai
    from google.genai.types import HttpOptions
    client = genai.Client(vertexai=True,
                          project=os.environ.get('GCP_PROJECT_ID',
                                                 'ai-sandbox-dev-f139'),
                          location='us-central1',
                          http_options=HttpOptions(api_version='v1'))

    methods = Counter()
    done = 0

    def work(eid):
        return eid, relocate(eid, events[eid], client, args.model, cache)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, k) for k in targets]
        for fut in as_completed(futs):
            eid, (center, strategy, note) = fut.result()
            done += 1
            if center is None:
                methods['failed'] += 1
            else:
                v = events[eid]
                v.setdefault('center_original', v['center'])
                v['center'] = center
                v['strategy'] = strategy
                v['center_note'] = note
                methods[strategy] += 1
            if done % 25 == 0:
                json.dump(cache, open(args.cache, 'w'))
                json.dump(events, open(args.events, 'w'), indent=2)
                print(f'  {done}/{len(targets)}  {dict(methods)}', flush=True)

    json.dump(cache, open(args.cache, 'w'))
    json.dump(events, open(args.events, 'w'), indent=2)

    print(f'\n{dict(methods)}')
    print(f'wrote {args.events}; previous values kept in center_original')

    n_centroid = methods.get('county_centroid', 0)
    if n_centroid:
        print(f'\n{n_centroid} fell back to the county centroid — right county, '
              f'but no evidence of where inside it, so the chip may miss the '
              f'event. Marked strategy=county_centroid; re-estimate from '
              f'article evidence once those events are harvested.')
    print('\nNext: python pipeline/quality_checks.py --stage events')
    print('Then delete Data/images for relocated events and re-download.')


if __name__ == '__main__':
    main()
