"""
Verify caption-image alignment using Gemini vision.
Loads downloaded images + generated captions, asks Gemini to grade alignment.

Run after test_pipeline.py completes:
    python verify_captions.py
"""

import os
import json
import re
from os.path import join, isdir
from google import genai
from PIL import Image
import io
import sys
import datetime

PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-project-id')
LOCATION = os.environ.get('GCP_LOCATION', 'us-central1')

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)
MODEL = "gemini-2.5-pro"

ODIR = 'Data/images'
RESULTS_FILE = 'Data/events_processed.json'


def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def load_image_as_part(path):
    with open(path, 'rb') as f:
        data = f.read()
    return {
        "inline_data": {
            "mime_type": "image/png" if path.endswith('.png') else "image/jpeg",
            "data": data,
        }
    }


def parse_caption_dates(caption_text):
    lines = caption_text.strip().split('\n')
    captions = {}
    for line in lines:
        match = re.match(r'(\d{4}-\d{2}-\d{2}):\s*(.*)', line.strip())
        if match:
            captions[match.group(1)] = match.group(2)
    return captions


def get_images_for_event(event_idx, strategy='llm'):
    img_dir = join(ODIR, f"{event_idx}_{strategy}")
    if not isdir(img_dir):
        return {}
    images = {}
    for fname in sorted(os.listdir(img_dir)):
        if not (fname.endswith('.jpg') or fname.endswith('.png')):
            continue
        match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
        if match:
            date = match.group(1)
            if '_pre_' in fname:
                phase = 'pre'
            elif '_post_' in fname:
                phase = 'post'
            else:
                phase = 'during'
            images[f"{phase}_{date}"] = join(img_dir, fname)
    return images


def verify_event(event_idx, event_data, strategy='llm'):
    images = get_images_for_event(event_idx, strategy)
    if not images:
        print(f"  No images found for {event_idx}_{strategy}")
        return None

    captions_text = event_data.get('captions', '')
    if not captions_text:
        print(f"  No captions for event {event_idx}")
        return None

    date_captions = parse_caption_dates(captions_text)

    # Build image sequence with dates for Gemini
    parts = []
    parts.append(f"""You are verifying whether satellite images actually capture a natural disaster event.
Event: {event_data['type']} — {event_data['event']} in {event_data.get('county', '')}, {event_data.get('state', '')}.

These are Sentinel-2 optical (RGB) images at 512x512 pixels (~11km coverage).
Each image has a caption describing what should be visible based on news reports.

Your job: Verify that these are GOOD training samples for a vision-language model.

Check:
1. Is the IMAGE at the right LOCATION? (terrain type, geographic features, urban/rural)
2. Are the CAPTIONS factually accurate? (do the numbers, dates, and details match?)
3. Does the TIMELINE progress logically? (pre-event → onset → peak → aftermath)
4. Any VISUAL EVIDENCE of the disaster? (burn scars, flooding, smoke — bonus but not required)

Score each image-caption pair:
1. LOCATION_CORRECT (1-5): Does terrain match the expected area?
   1 = clearly wrong (ocean for inland fire), 5 = terrain matches perfectly
2. CAPTION_ACCURACY (1-5): Are facts plausible and internally consistent?
3. TIMELINE (1-5): Does progression across dates make sense?
4. VISUAL_EVIDENCE (1-5): Any visible disaster evidence?
   1 = none visible, 3 = subtle/ambiguous, 5 = obvious. Score of 1-2 is FINE
   if location and caption are correct — the model will learn to see what you can't.

Also:
- OVERALL_SCORE (1-5): Is this a good training sample? (location correct + caption accurate = good)
- BEST_IMAGE: which date
- WORST_IMAGE: which date
- EVIDENCE_FOUND: any visual features you notice (even subtle ones)
- ISSUES: ONLY flag real problems (wrong location, factual errors, broken timeline)

Return as JSON:
{{
  "per_date": {{
    "YYYY-MM-DD": {{"visual_evidence": N, "caption_accuracy": N, "timeline": N, "location_correct": N, "notes": "..."}},
    ...
  }},
  "overall_score": N,
  "best_image": "YYYY-MM-DD",
  "worst_image": "YYYY-MM-DD",
  "evidence_found": ["...", "..."],
  "issues": ["...", "..."],
  "summary": "1-2 sentence assessment: does this image set capture the disaster?"
}}

Here are the images and captions:
""")

    img_count = 0
    event_keys = sorted([k for k in images.keys() if k.startswith('during') or k.startswith('event')])
    if not event_keys:
        return None

    for key in event_keys:
        path = images[key]
        date = re.search(r'(\d{4}-\d{2}-\d{2})', key).group(1)
        caption = date_captions.get(date, '')
        # Also try without exact match — find closest caption date
        if not caption and date_captions:
            closest = min(date_captions.keys(), key=lambda d: abs(
                datetime.datetime.strptime(d, '%Y-%m-%d') -
                datetime.datetime.strptime(date, '%Y-%m-%d')).days)
            days_off = abs((datetime.datetime.strptime(closest, '%Y-%m-%d') -
                           datetime.datetime.strptime(date, '%Y-%m-%d')).days)
            if days_off <= 5:
                caption = f"[nearest caption, {days_off}d off] {date_captions[closest]}"
            else:
                caption = '(no caption for this date)'

        parts.append(f"\n--- Image: {date} ---")
        parts.append(f"Caption: {caption}")
        parts.append(load_image_as_part(path))
        img_count += 1

    if img_count == 0:
        print(f"  No valid images to verify")
        return None

    log(f"Sending {img_count} images to Gemini for verification...")
    try:
        response = client.models.generate_content(model=MODEL, contents=parts)
        raw = response.text.strip()
        if raw.startswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
        result = json.loads(raw)
        return result
    except Exception as e:
        print(f"  [!] Verification failed: {e}")
        return {'error': str(e), 'raw': response.text if 'response' in dir() else ''}


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"No results file found at {RESULTS_FILE}. Run test_pipeline.py first.")
        return

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    verify_results = {}
    strategies = ['firms']

    for event_idx, event_data in sorted(results.items(), key=lambda x: int(x[0])):
        if 'error' in event_data:
            print(f"\nEvent {event_idx}: skipping (had error)")
            continue

        print(f"\n{'='*70}")
        print(f"VERIFYING EVENT {event_idx}: {event_data['event']}")
        print(f"  Type: {event_data['type']} | Best strategy: {event_data.get('best_strategy', '?')}")

        # Print change detection summary if available
        cd = event_data.get('change_detection', {})
        if cd:
            print(f"  Change detection confidence:")
            for strat, scores in cd.items():
                conf = scores.get('confidence', 0)
                sigs = ', '.join(scores.get('signals', []))
                print(f"    {strat}: {conf:.4f} — {sigs}")

        event_scores = {}
        for strategy in strategies:
            img_dir = join(ODIR, f"{event_idx}_{strategy}")
            if not isdir(img_dir) or not os.listdir(img_dir):
                continue

            log(f"Verifying {strategy} strategy...")
            result = verify_event(event_idx, event_data, strategy)
            if result:
                event_scores[strategy] = result
                score = result.get('overall_score', '?')
                summary = result.get('summary', '')
                print(f"  {strategy}: overall={score}/5 — {summary}")
                if result.get('issues'):
                    for issue in result['issues']:
                        print(f"    [!] {issue}")

        verify_results[event_idx] = event_scores

    # Save verification results
    out_file = 'Data/verification_results.json'
    with open(out_file, 'w') as f:
        json.dump(verify_results, f, indent=2, default=str)

    # Summary table
    print(f"\n\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Event':<8} {'Event Name':<30} {'FEMA':>6} {'LLM':>6} {'FIRMS':>6} {'LLM+F':>6} {'Best Evidence'}")
    print(f"{'-'*95}")
    for event_idx, scores in sorted(verify_results.items(), key=lambda x: int(x[0])):
        event_data = results[event_idx]
        fema_s = scores.get('fema', {}).get('overall_score', '-')
        llm_s = scores.get('llm', {}).get('overall_score', '-')
        firms_s = scores.get('firms', {}).get('overall_score', '-')
        llm_firms_s = scores.get('llm_firms', {}).get('overall_score', '-')

        # Find which strategy had the best visual evidence
        best_evidence = ''
        best_score = 0
        for strat in ['fema', 'llm', 'firms', 'llm_firms']:
            s = scores.get(strat, {})
            sc = s.get('overall_score', 0)
            if isinstance(sc, (int, float)) and sc > best_score:
                best_score = sc
                evidence = s.get('evidence_found', [])
                best_evidence = evidence[0][:40] if evidence else ''

        print(f"{event_idx:<8} {event_data.get('event', '?')[:30]:<30} "
              f"{str(fema_s):>6} {str(llm_s):>6} {str(firms_s):>6} {str(llm_firms_s):>6} {best_evidence}")

    # Averages per strategy
    print(f"\n{'='*70}")
    print("AVERAGES BY STRATEGY")
    for strat in strategies:
        scores_list = [v.get(strat, {}).get('overall_score', None) for v in verify_results.values()]
        valid = [s for s in scores_list if isinstance(s, (int, float))]
        if valid:
            print(f"  {strat}: {sum(valid)/len(valid):.2f}/5 ({len(valid)} events)")

    # Averages per event type
    print(f"\n{'='*70}")
    print("BEST STRATEGY WINS BY EVENT TYPE")
    type_wins = {}
    for event_idx, event_data in results.items():
        if 'error' in event_data or event_idx not in verify_results:
            continue
        etype = event_data['type']
        best = event_data.get('best_strategy', '?')
        if etype not in type_wins:
            type_wins[etype] = {}
        type_wins[etype][best] = type_wins[etype].get(best, 0) + 1
    for etype, wins in sorted(type_wins.items()):
        print(f"  {etype}: {wins}")

    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
