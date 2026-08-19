"""
Step 1b: extract facts from scraped article content (Data/article_content.json).

Same as extract_facts.py but reads the scraped-content cache instead of a CSV
that has no content.

Usage:
    export GCP_PROJECT_ID=ai-sandbox-dev-f139
    python pipeline/extract_facts_from_scrape.py --limit 25
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

CONTENT_PATH = 'Data/article_content.json'
FACTS_PATH = 'Data/article_facts.json'


PROMPT = """You are given a target disaster event and a news article.
First decide whether the article is actually about THAT event. Then extract facts.

TARGET EVENT:
  name:   {event_name}
  type:   {event_type}
  place:  {county} County, {state}
  window: {fema_start} to {fema_end}

Return STRICT JSON only (no prose, no markdown fence):

{
  "is_about_target_event": true,
  "relevance_reason": "",
  "extent_number": null,
  "extent_unit": null,
  "extent_as_of_date": null,
  "contained_pct": null,
  "affected_features": [],
  "notable_dates": {"start": null, "peak": null, "contained": null}
}

Field rules:
- is_about_target_event: true ONLY if the article describes the SAME incident —
  same disaster, same general place, same time window. Set false if it's a
  different fire/storm, a different state, a different year, a generic
  explainer, a site index page, or a page that has since been updated to cover
  a newer incident. When false, leave all other fields null/empty.
- relevance_reason: one short sentence justifying your true/false call.
- extent_number / extent_unit: size of the impact. unit ∈
  {"acres","sq_miles","structures","homes"}. Use the LARGEST clearly-stated figure.
- extent_as_of_date: the date that figure describes (YYYY-MM-DD). Resolve
  relative phrases ("as of Wednesday") against the publication date. If unclear,
  use the publication date.
- contained_pct: fire containment percent (0-100), null if not a fire.
- affected_features: NAMED physical things visible from ~10m satellite imagery
  that the article says were affected — roads/highways, rivers, lakes, forests,
  airports, ports, named neighborhoods, named ridges or mountains.
  They must be plausibly within or near {county} County, {state}.
  EXCLUDE: people, dollar amounts, agencies, generic phrases.
- notable_dates: when the event started / peaked / was contained (YYYY-MM-DD).

Only extract what the text states. Use null generously. No guessing.

PUBLICATION DATE: {pub_date}
TITLE: {title}

ARTICLE:
{content}
"""


def call(client, prompt, model_id):
    from google.genai import types
    try:
        cfg = types.GenerateContentConfig(
            max_output_tokens=700,
            temperature=0.0,
            response_mime_type='application/json',
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    except Exception:
        cfg = types.GenerateContentConfig(
            max_output_tokens=700, temperature=0.0,
            response_mime_type='application/json')
    try:
        r = client.models.generate_content(model=model_id, contents=prompt, config=cfg)
        return r.text or ''
    except Exception as e:
        return f'[ERR: {e}]'


def process(url, entry, client, model_id, event_meta):
    content = entry.get('content', '')
    if len(content) < 250:
        return url, {'error': 'too_short', 'event_id': entry.get('event_id')}
    meta = event_meta or {}
    p = (PROMPT
         .replace('{event_name}', str(meta.get('event', 'unknown')))
         .replace('{event_type}', str(meta.get('type', 'unknown')))
         .replace('{county}', str(meta.get('county', 'unknown')))
         .replace('{state}', str(meta.get('state', 'unknown')))
         .replace('{fema_start}', str(meta.get('start_date', 'unknown')))
         .replace('{fema_end}', str(meta.get('end_date', 'unknown')))
         .replace('{pub_date}', entry.get('pub_date') or 'unknown')
         .replace('{title}', entry.get('title', ''))
         .replace('{content}', content[:18000]))
    resp = call(client, p, model_id)
    if resp.startswith('[ERR'):
        return url, {'error': resp[:160], 'event_id': entry.get('event_id')}
    try:
        facts = json.loads(resp)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if not m:
            return url, {'error': f'no_json: {resp[:120]}', 'event_id': entry.get('event_id')}
        try:
            facts = json.loads(m.group(0))
        except Exception:
            return url, {'error': f'bad_json: {resp[:120]}', 'event_id': entry.get('event_id')}
    facts['event_id'] = entry.get('event_id')
    facts['pub_date'] = entry.get('pub_date')
    facts['domain'] = entry.get('domain')
    facts['url'] = url
    return url, facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--content', default=CONTENT_PATH)
    ap.add_argument('--out', default=FACTS_PATH)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--event', nargs='+', type=int, default=None)
    ap.add_argument('--model', default='gemini-2.5-flash-lite')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--events', default='Data/events_processed.json')
    args = ap.parse_args()

    content = json.load(open(args.content))
    events = json.load(open(args.events)) if os.path.exists(args.events) else {}
    facts = json.load(open(args.out)) if os.path.exists(args.out) else {}
    if facts:
        print(f"Resumed {len(facts)} cached facts")

    targets = []
    for url, e in content.items():
        if url in facts or not e.get('content'):
            continue
        if args.event and int(e.get('event_id', -1)) not in args.event:
            continue
        targets.append((url, e))
        if args.limit and len(targets) >= args.limit:
            break

    print(f"To extract: {len(targets)}")
    if not targets:
        return

    from google import genai
    from google.genai.types import HttpOptions
    project = os.environ.get('GCP_PROJECT_ID', 'ai-sandbox-dev-f139')
    client = genai.Client(vertexai=True, project=project, location='us-central1',
                          http_options=HttpOptions(api_version='v1'))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process, u, e, client, args.model,
                            events.get(str(e.get('event_id'))))
                for u, e in targets]
        for i, fut in enumerate(as_completed(futs), 1):
            url, entry = fut.result()
            facts[url] = entry
            if i % 10 == 0:
                json.dump(facts, open(args.out, 'w'))
                print(f"  {i}/{len(targets)}", flush=True)

    json.dump(facts, open(args.out, 'w'))
    good = [f for f in facts.values() if not f.get('error')]
    relevant = [f for f in good if f.get('is_about_target_event')]
    with_extent = [f for f in relevant if f.get('extent_number')]
    with_feats = [f for f in relevant if f.get('affected_features')]
    print(f"\nExtracted {len(facts)} → {args.out}")
    print(f"  parsed ok:          {len(good)}")
    print(f"  about target event: {len(relevant)} ({100*len(relevant)/max(1,len(good)):.0f}%)")
    print(f"  with extent:        {len(with_extent)}")
    print(f"  with features:      {len(with_feats)}")


if __name__ == '__main__':
    main()
