"""
Step 1: extract visually-verifiable facts from each article using cheap LLM.

Reads Data/articles.csv (columns: event_id, url, title, pub_date, content).
Writes Data/article_facts.json keyed by URL:

    {
      "https://...": {
        "event_id": 88,
        "pub_date": "2022-07-22",
        "extent_acres": 4000,
        "contained_pct": 15,
        "affected_features": ["FM 205", "Glen Rose ISD", "Chalk Mountain"],
        "notable_dates": {"start": "2022-07-18", "contained": null},
        "raw_fact_snippets": ["by Friday morning had scorched 4,000 acres", ...]
      }
    }

Cached per URL — re-runs skip already-processed articles.

Usage:
    export GCP_PROJECT_ID=ai-sandbox-dev-f139
    python pipeline/extract_facts.py                    # all articles
    python pipeline/extract_facts.py --event 0 1 88     # specific events
    python pipeline/extract_facts.py --limit 100        # first N
"""

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


ARTICLES_CSV = 'Data/articles.csv'
FACTS_PATH = 'Data/article_facts.json'


# Schema-restricted prompt: only extract facts that could be visible in
# high-resolution satellite imagery. No casualties, dollar amounts, wind
# speeds (invisible), rescue efforts, etc.
EXTRACTION_PROMPT = """You are extracting facts from a news article about a natural disaster.
Return STRICT JSON with these fields (use null if not found):

{
  "extent_number": <number or null>,          // e.g. acres burned, sq miles flooded, structures destroyed
  "extent_unit": "<acres|sq_miles|structures|homes|null>",
  "extent_as_of_date": "<YYYY-MM-DD or null>",  // when this extent was measured; usually article's implied "as of" date
  "contained_pct": <0-100 or null>,           // fire containment percentage
  "affected_features": [                       // NAMED, potentially-visible physical features affected
    "highway 101", "the airport", "downtown Glen Rose", "Chalk Mountain"
  ],
  "notable_dates": {
    "start": "<YYYY-MM-DD or null>",          // when event began
    "peak": "<YYYY-MM-DD or null>",           // when extent peaked
    "contained": "<YYYY-MM-DD or null>",      // when contained/ended
    "recovery": "<YYYY-MM-DD or null>"        // when normalcy returned
  }
}

Rules:
- Extract ONLY facts stated in the text. No guessing.
- affected_features must be things visible from ~10m resolution satellite:
  roads, water bodies, forests, airports, ports, stadiums, neighborhoods,
  named ridges/valleys. NOT: people, insurance, evacuations, closures without
  a named location, generic terms like "the area".
- Skip facts about: casualties, dollar amounts, wind speeds, rescue efforts.
- If article is not about a natural disaster, return {} for all fields.
- Output ONLY the JSON, no prose.

Article title: {title}
Article publication date: {pub_date}
Article text:
{content}
"""


def call_gemini(client, text, model_id='gemini-2.5-flash-lite'):
    from google.genai import types
    try:
        thinking = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        thinking = None

    try:
        resp = client.models.generate_content(
            model=model_id,
            contents=text,
            config=types.GenerateContentConfig(
                max_output_tokens=800,
                temperature=0.0,
                response_mime_type='application/json',
                thinking_config=thinking,
            ),
        )
        return resp.text or ''
    except Exception as e:
        return f'[ERR: {e}]'


def load_articles(csv_path):
    """Yield article rows as dicts."""
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            yield row


def process_article(row, client, model_id):
    url = row.get('url', '').strip()
    if not url:
        return None
    content = (row.get('content') or row.get('text') or '').strip()
    if len(content) < 200:
        return url, {'error': 'article too short', 'event_id': row.get('event_id')}

    # Truncate very long articles
    content = content[:20000]
    prompt = EXTRACTION_PROMPT.replace('{title}', row.get('title', ''))\
                              .replace('{pub_date}', row.get('pub_date', ''))\
                              .replace('{content}', content)

    resp = call_gemini(client, prompt, model_id=model_id)
    if resp.startswith('[ERR'):
        return url, {'error': resp[:200], 'event_id': row.get('event_id')}

    # Parse JSON
    try:
        facts = json.loads(resp)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            try:
                facts = json.loads(m.group(0))
            except Exception:
                return url, {'error': f'bad JSON: {resp[:200]}', 'event_id': row.get('event_id')}
        else:
            return url, {'error': f'no JSON: {resp[:200]}', 'event_id': row.get('event_id')}

    facts['event_id'] = row.get('event_id')
    facts['pub_date'] = row.get('pub_date', '')
    facts['url'] = url
    return url, facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=ARTICLES_CSV)
    parser.add_argument('--out', default=FACTS_PATH)
    parser.add_argument('--event', nargs='+', type=int, default=None,
                        help='Only process articles for these event IDs')
    parser.add_argument('--limit', type=int, default=None,
                        help='Cap total number of articles processed')
    parser.add_argument('--model', default='gemini-2.5-flash-lite')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Missing {args.csv}")
        sys.exit(1)

    # Load existing facts (resume)
    facts = {}
    if os.path.exists(args.out):
        facts = json.load(open(args.out))
        print(f"Resumed {len(facts)} cached facts")

    # Filter articles
    targets = []
    for row in load_articles(args.csv):
        url = row.get('url', '').strip()
        if not url or url in facts:
            continue
        eid = row.get('event_id')
        if args.event and (not eid or int(eid) not in args.event):
            continue
        targets.append(row)
        if args.limit and len(targets) >= args.limit:
            break

    print(f"To process: {len(targets)} articles")
    if not targets:
        return

    # Init Gemini client
    from google import genai
    from google.genai.types import HttpOptions
    project = os.environ.get('GCP_PROJECT_ID', 'ai-sandbox-dev-f139')
    client = genai.Client(vertexai=True, project=project, location='us-central1',
                           http_options=HttpOptions(api_version='v1'))

    def _work(row):
        return process_article(row, client, args.model)

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_work, r): r for r in targets}
        for fut in as_completed(futs):
            res = fut.result()
            if not res:
                continue
            url, entry = res
            facts[url] = entry
            n_done += 1
            if n_done % 25 == 0:
                json.dump(facts, open(args.out, 'w'))
                print(f"  {n_done}/{len(targets)} extracted", flush=True)

    json.dump(facts, open(args.out, 'w'))
    print(f"\nSaved {len(facts)} article facts to {args.out}")


if __name__ == '__main__':
    main()
