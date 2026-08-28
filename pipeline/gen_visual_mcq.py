"""
Generate MCQ that require the imagery, from verified article facts.

The previous generator wrote questions from Gemini's own speculative captions,
so they inherited text-answerable structure — blind accuracy was 0.507 against
0.250 chance. Diagnosed leaks:

  1. non-visual attributes   "wind gusting up to 100 / 120 / 70 mph"
  2. implausible distractors "trees completely submerged" for a wind storm
  3. telegraphing phrasing   "what was explicitly stated as NOT visible"
  4. answer-position skew    (fixed separately by shuffling)

This generator writes from the harvested per-frame records instead, and the
prompt bans each failure mode explicitly. Every option must be a state of the
land surface that a satellite could distinguish, and all four must be
physically plausible for the given disaster type so the answer can't be reached
by elimination.

Usage:
    export GCP_PROJECT_ID=ai-sandbox-dev-f139
    python pipeline/gen_visual_mcq.py --limit 20
    python pipeline/gen_visual_mcq.py --event 5357 6186
"""

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

ALIGNED = 'Data/aligned_frames.json'
SIGNAL = 'Data/visual_signal.json'
OUT_PATH = 'Data/qa_visual_mcq.json'


PROMPT = """You are writing multiple-choice questions to test whether a model can
READ A SEQUENCE OF SATELLITE IMAGES. The questions will be shown alongside the
images. A model that cannot see the images must do no better than random guessing.

EVENT
  type:     {etype}
  place:    {county} County, {state}
  onset:    {onset}
  frames:   {frame_summary}

VERIFIED FACTS (from dated news reports, already checked as describing this event)
{facts}

FEATURES CONFIRMED INSIDE THE IMAGE FRAME
{features}

MEASURED VISUAL CHANGE (computed from the actual pixels)
{signal}

Write {n} multiple-choice questions as STRICT JSON:

{{"questions": [
  {{"question": "...", "options": ["...","...","...","..."],
    "answer_index": 0, "requires": "...", "why_blind_fails": "..."}}
]}}

HARD CONSTRAINTS — a question violating any of these is unusable:

1. VISUAL ONLY. The answer must be readable off the land surface: extent and
   shape of a burn scar, presence of standing water, snow cover, a debris path,
   vegetation colour, sediment plumes, changes between frames.
   NEVER ask about wind speed, casualties, dollar cost, evacuation counts,
   agency actions, or anything else invisible from orbit.

2. DISTRACTORS MUST BE SAME-SIGNATURE VARIANTS. This is the most important rule.
   All four options must describe the SAME KIND of phenomenon, differing only in
   extent, location within the frame, shape, timing, or degree.
   A model told "this is a {etype}" must NOT be able to eliminate options using
   that fact alone.

   WRONG (cross-type contrast — answerable without looking):
     a. Widespread flooding across the region
     b. Significant snow accumulation
     c. A dark burn scar covering the ridge      <-- obviously the fire one
     d. New urban development

   RIGHT (same-signature variants — must inspect the pixels):
     a. A burn scar confined to the northern third of the frame
     b. A burn scar spanning most of the frame, split by the river
     c. Two separate burn scars on opposite sides of the highway
     d. A narrow burn scar following the ridgeline

   Never build an option set where one option matches the disaster type and the
   others name different disaster types.

3. NO TELEGRAPHING. Do not phrase a question so the wording implies the answer.
   Banned: "explicitly stated", "as mentioned", "according to reports",
   "which was NOT visible", "described as".
   The question must read as though asked by someone looking at the images
   with no report in hand.

4. OPTIONS MUTUALLY EXCLUSIVE and comparable in length and specificity. Do not
   make the correct option noticeably longer or more detailed.

5. NO MAGNITUDE LADDER. Do not build options that differ only in how much
   happened. A disaster is known to have occurred, so the strongest option is
   guessable without looking.
   WRONG: a) remained the same  b) completely receded  c) slightly decreased
          d) significantly increased        <-- always the answer
   RIGHT: options differing in WHERE, WHAT SHAPE, or WHICH FRAME, all at
          comparable magnitude:
          a) water covering the fields north of the highway
          b) water covering the fields south of the highway
          c) water on both sides, deepest near the bridge
          d) water confined to the river channel itself

6. NO TEMPORALLY OBVIOUS KEYS. Do not ask which frame shows peak impact when
   the answer is "the one right after onset" — that follows from the dates
   alone. If you ask about timing, the options must be frames where the visual
   evidence genuinely differs and the ordering is not inferable from the
   event date.

7. GROUNDED IN THE FACTS ABOVE. The correct answer must follow from the verified
   facts or measured change. Do not invent numbers.

8. NEVER make "no change" / "nothing visible" / "insufficient evidence" the
   CORRECT answer. This event has a measured visual signature — ask about what
   IS there. A null answer is a shortcut: a model that always picks the
   negative option would score well without looking.
   Such phrasings may appear as distractors, never as the key.

9. Vary answer_index across questions — do not always use the same position.

THE TEST TO APPLY TO EVERY QUESTION YOU WRITE:
   Could someone who knows only "a {etype} happened in {county} County" — and
   has never seen the images — pick the right answer more often than 1 in 4?
   If yes, rewrite it.

For each question also give:
  "requires"          what the model must observe in the imagery to answer
  "why_blind_fails"   why text alone is insufficient

Output only the JSON.
"""


def summarize_frames(ev):
    out = []
    for fr in ev['frames']:
        b = fr['bounds']
        lo, hi = b.get('lower'), b.get('upper')
        s = f"{fr['date']} ({fr['phase']}"
        if lo:
            s += f", >={lo['value']:,.0f} {lo['unit']}"
        if hi:
            s += f", <={hi['value']:,.0f} {hi['unit']}"
        out.append(s + ')')
    return '; '.join(out)


def summarize_facts(ev):
    lines = []
    for r in ev.get('extent_timeseries', []):
        lines.append(f"  {r['date']}: {r['value']:,.0f} {r['unit']} "
                     f"(reported by {r.get('source','?')})")
    for r in ev.get('containment_timeseries', []):
        lines.append(f"  {r['date']}: {r['value']:.0f}% contained")
    nd = ev.get('notable_dates') or {}
    for k, v in nd.items():
        if v:
            lines.append(f"  {k}: {v}")
    return '\n'.join(lines) if lines else '  (no numeric facts recovered)'


def summarize_signal(sig):
    """Deliberately coarse.

    An earlier version passed the per-channel measurements into the prompt.
    Gemini then wrote questions about the metrics themselves — "what was the
    trend in greenness after the fire started?" — whose answers are both
    copied from the prompt and inferable from prior knowledge (fires reduce
    greenness). It also propagated proxy noise: one fire event measured
    brightness UP, so the key became "the area became noticeably brighter",
    which is physically wrong for a burn scar.

    The signal measurement decides WHICH events get questions. It must not
    become visible content the generator can restate.
    """
    if not sig:
        return '  (not measured)'
    v = sig.get('verdict')
    if v == 'strong':
        return ('  A clear change in the land surface is present between the '
                'pre-event and post-event frames.')
    if v == 'weak':
        return ('  A subtle change in the land surface is present between the '
                'pre-event and post-event frames.')
    return '  (change not characterised)'


def call_json(client, prompt, model_id, max_tokens=2200):
    from google.genai import types
    try:
        cfg = types.GenerateContentConfig(
            max_output_tokens=max_tokens, temperature=0.4,
            response_mime_type='application/json',
            thinking_config=types.ThinkingConfig(thinking_budget=0))
    except Exception:
        cfg = types.GenerateContentConfig(
            max_output_tokens=max_tokens, temperature=0.4,
            response_mime_type='application/json')
    try:
        r = client.models.generate_content(model=model_id, contents=prompt, config=cfg)
        return r.text or ''
    except Exception as e:
        return f'[ERR: {e}]'


BANNED_PATTERNS = [
    r'\bmph\b', r'\bwind speed', r'\bcasualt', r'\bdeath', r'\bkilled',
    r'\$', r'\bdollar', r'\bevacuat', r'\binsur', r'\bexplicitly stated',
    r'\bas mentioned', r'\baccording to (the )?report', r'\bdescribed as',
    r'\bnot visible', r'\bwere stated',
    # Questions about our own proxy metrics are answerable from prior
    # knowledge (fires reduce greenness) and often just restate the prompt.
    r'\bgreenness\b', r'\bwetness\b', r'\btexture (metric|value|index)\b',
    r'\btrend in brightness\b', r'\bbrightness (metric|value|index)\b',
]


# Vocabulary that identifies a disaster signature. If the four options draw on
# several of these groups, the answer is reachable by knowing the event type
# alone — no imagery needed.
SIGNATURE_VOCAB = {
    'fire':  [r'burn scar', r'\bburn(ed|t)\b', r'\bcharred\b', r'\bscorch',
              r'\bwildfire\b', r'\bash\b'],
    'water': [r'\bflood', r'\binundat', r'\bsubmerg', r'standing water',
              r'\boverflow', r'water level'],
    'snow':  [r'\bsnow', r'\bice\b', r'\bfrost\b', r'\bglaciat'],
    'wind':  [r'\bdebris\b', r'\bdamage path\b', r'\bdowned tree',
              r'\bdefoliat', r'\bblown down\b'],
    'urban': [r'\burban development\b', r'new (structure|building|construction)',
              r'\bnew road'],
    'geo':   [r'\blandslide\b', r'\bscarp\b', r'\bmudslide\b', r'\bfissure\b',
              r'\bliquefaction\b'],
}


def signature_groups(text):
    t = text.lower()
    return {g for g, pats in SIGNATURE_VOCAB.items()
            if any(re.search(p, t) for p in pats)}


def cross_type_contrast(opts):
    """True if options span multiple disaster signatures — a giveaway."""
    groups = set()
    for o in opts:
        groups |= signature_groups(o)
    return len(groups) >= 2


# Intensity words, ordered weak -> strong. If the four options differ mainly
# along this axis and the key is the strongest, "a disaster happened so pick
# the biggest" wins without looking at anything.
INTENSITY = [
    (r'\bno\b|\bnone\b|\bunchanged\b|\bnot\b', 0),
    (r'\bsubtle|\bslight|\bminor|\bimperceptible|\bnegligible|\blimited\b|'
     r'\bisolated\b|\blocalized\b|\bsmall\b', 1),
    (r'\bmoderate|\bpartial|\bsome\b|\bnoticeable\b', 2),
    (r'\bsignificant|\bsubstantial|\bmajor\b|\bwidespread|\bextensive|'
     r'\bdramatic|\bsevere\b|\bpronounced|\blarge-scale\b|\bcomplete', 3),
]


def intensity_rank(text):
    t = text.lower()
    best = None
    for pat, rank in INTENSITY:
        if re.search(pat, t):
            best = rank if best is None else max(best, rank)
    return best


def magnitude_ladder(opts, ai):
    """True if options form an intensity scale and the key is the extreme."""
    ranks = [intensity_rank(o) for o in opts]
    scored = [r for r in ranks if r is not None]
    if len(scored) < 3:
        return False
    if ranks[ai] is None:
        return False
    # Key is strictly the strongest, and the options span the scale
    return ranks[ai] == max(scored) and len(set(scored)) >= 3


# Asking which frame shows peak impact is answerable from the dates alone.
TEMPORALLY_OBVIOUS = re.compile(
    r'which (period|frame|image|date).{0,40}'
    r'(most pronounced|peak|greatest|maximum|strongest|most significant)|'
    r'(most pronounced|peak|greatest|maximum).{0,30}(period|frame|phase)',
    re.I)


NEGATIVE_ANSWER = re.compile(
    r'no (significant|discernible|noticeable|clear|visible|major|apparent)|'
    r'unchanged|not provide sufficient|no change|absence of|'
    r'do(es)? not (show|provide|indicate)|remains? (largely )?the same|'
    r'insufficient (visual )?evidence|cannot be determined|none of the above',
    re.I)


def validate(q):
    """Reject questions that violate the hard constraints."""
    text = (q.get('question', '') + ' ' + ' '.join(q.get('options', []))).lower()
    for pat in BANNED_PATTERNS:
        if re.search(pat, text):
            return False, f'banned pattern: {pat}'
    opts = q.get('options') or []
    if len(opts) != 4:
        return False, 'not 4 options'
    if len(set(o.strip().lower() for o in opts)) != 4:
        return False, 'duplicate options'
    ai = q.get('answer_index')
    if not isinstance(ai, int) or not 0 <= ai < 4:
        return False, 'bad answer_index'
    # A null finding as the key is a shortcut — always guessable without looking
    if NEGATIVE_ANSWER.search(opts[ai]):
        return False, f'null-finding correct answer: {opts[ai][:60]}'
    # Options spanning several disaster signatures are answerable from the
    # event type alone
    if cross_type_contrast(opts):
        groups = set()
        for o in opts:
            groups |= signature_groups(o)
        return False, f'cross-type distractors {sorted(groups)}: {opts[ai][:50]}'
    # "A disaster happened, so pick the biggest number" — no imagery needed
    if magnitude_ladder(opts, ai):
        return False, f'magnitude ladder, key is extreme: {opts[ai][:55]}'
    # "Peak impact is right after onset" follows from the dates
    if TEMPORALLY_OBVIOUS.search(q.get('question', '')):
        return False, f'temporally obvious: {q.get("question","")[:60]}'
    # Correct option markedly longer than the others is a giveaway
    lens = [len(o) for o in opts]
    if lens[ai] > 1.8 * (sum(lens) - lens[ai]) / 3:
        return False, 'correct option much longer'
    return True, ''


def gen_for_event(ev, sig, client, model_id, n_questions):
    verdict = (sig or {}).get('verdict', 'none')
    n_post = sum(1 for fr in ev['frames']
                 if fr['phase'] in ('onset', 'during', 'post'))

    # Events with no observable signature cannot support change-detection
    # questions. Asking anyway produces "nothing changed" as the answer, which
    # is its own shortcut — a blind model that always picks the negative option
    # scores well. Skip them; they still belong in the dataset for training,
    # just not as MCQ about visible change.
    if verdict in ('insufficient_frames', 'unobserved_transient') or n_post == 0:
        return [], [f'skipped: verdict={verdict}, n_post={n_post}']
    if verdict in ('none', 'CONTRADICTED'):
        return [], [f'skipped: no measurable signal (verdict={verdict})']

    prompt = (PROMPT
              .replace('{etype}', str(ev.get('type')))
              .replace('{county}', str(ev.get('county')))
              .replace('{state}', str(ev.get('state')))
              .replace('{onset}', str((ev.get('notable_dates') or {}).get('start')
                                      or ev.get('fema_start')))
              .replace('{frame_summary}', summarize_frames(ev))
              .replace('{facts}', summarize_facts(ev))
              .replace('{features}', ', '.join(
                  f['name'] for f in ev.get('features_in_chip', [])) or '  (none)')
              .replace('{signal}', summarize_signal(sig))
              .replace('{n}', str(n_questions)))

    raw = call_json(client, prompt, model_id)
    if raw.startswith('[ERR'):
        return [], [raw[:120]]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return [], ['unparseable']
        try:
            data = json.loads(m.group(0))
        except Exception:
            return [], ['unparseable']

    kept, rejected = [], []
    for q in data.get('questions', []):
        ok, why = validate(q)
        if not ok:
            rejected.append(f"{why}: {q.get('question','')[:70]}")
            continue
        kept.append(q)
    return kept, rejected


def to_training_format(ev, q, seed_material):
    """Shuffle option order, then emit in the shared conversations format.

    Takes seed material rather than a shared Random: generation runs under a
    ThreadPoolExecutor and random.Random is not thread-safe. A single shared
    instance produced a biased permutation — position 'd' was over-represented
    and the blind baseline learned to exploit it, which looked like a flaw in
    the questions rather than a bug here.
    """
    rng = random.Random(seed_material)
    opts = list(q['options'])
    correct = opts[q['answer_index']]
    rng.shuffle(opts)
    idx = opts.index(correct)
    letters = ['a', 'b', 'c', 'd']
    body = '\n'.join(f'{letters[i]}. {o}' for i, o in enumerate(opts))
    question = (f"{q['question']}\n{body}\n"
                f"Answer with a single letter (a, b, c, or d).")
    return {
        'event_id': ev['event_id'],
        'task': 'visual_mcq',
        'video': [fr['path'] for fr in ev['frames']],
        'timestamp': [fr['date'] for fr in ev['frames']],
        'requires': q.get('requires'),
        'why_blind_fails': q.get('why_blind_fails'),
        'conversations': [
            {'from': 'human', 'value': question},
            {'from': 'gpt', 'value': letters[idx]},
        ],
    }


def rebalance_answers(items, seed=42):
    """Force a near-uniform answer-position distribution.

    Even correct per-question shuffling leaves the aggregate distribution
    noticeably uneven at these sample sizes, and a blind model exploits it.
    Rotating each question's options by a deterministic offset guarantees
    balance without changing any question's content.
    """
    letters = ['a', 'b', 'c', 'd']
    order = sorted(range(len(items)),
                   key=lambda i: (items[i]['event_id'], i))
    out = list(items)

    for slot, i in enumerate(order):
        it = items[i]
        q_text = it['conversations'][0]['value']
        gold = it['conversations'][1]['value'].strip()
        m = re.findall(r'^([a-d])\.\s(.+)$', q_text, re.M)
        if len(m) != 4 or gold not in letters:
            continue
        opts = [t for _, t in m]
        cur = letters.index(gold)
        target = slot % 4                      # round-robin across a/b/c/d
        shift = (target - cur) % 4
        rotated = opts[-shift:] + opts[:-shift] if shift else opts

        stem = q_text.split('\n' + m[0][0] + '. ')[0]
        tail = ('\nAnswer with a single letter (a, b, c, or d).'
                if 'Answer with a single letter' in q_text else '')
        body = '\n'.join(f'{letters[j]}. {o}' for j, o in enumerate(rotated))
        new = dict(it)
        new['conversations'] = [
            {'from': 'human', 'value': f'{stem}\n{body}{tail}'},
            {'from': 'gpt', 'value': letters[target]},
        ]
        out[i] = new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aligned', default=ALIGNED)
    ap.add_argument('--signal', default=SIGNAL)
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--event', nargs='+', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--n-per-event', type=int, default=4)
    ap.add_argument('--model', default='gemini-2.5-flash')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    aligned = json.load(open(args.aligned))
    signal = json.load(open(args.signal)) if os.path.exists(args.signal) else {}
    rng = random.Random(args.seed)

    keys = sorted(aligned, key=lambda k: int(k))
    if args.event:
        keys = [k for k in keys if int(k) in args.event]
    if args.limit:
        keys = keys[:args.limit]

    from google import genai
    from google.genai.types import HttpOptions
    project = os.environ.get('GCP_PROJECT_ID', 'ai-sandbox-dev-f139')
    client = genai.Client(vertexai=True, project=project, location='us-central1',
                          http_options=HttpOptions(api_version='v1'))

    print(f'Generating visual MCQ for {len(keys)} events\n')
    all_q, all_rej = [], []

    def work(k):
        return k, gen_for_event(aligned[k], signal.get(k), client,
                                args.model, args.n_per_event)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, k) for k in keys]
        for fut in as_completed(futs):
            k, (kept, rejected) = fut.result()
            for qi, q in enumerate(kept):
                # Deterministic per-question seed — no shared RNG across threads
                all_q.append(to_training_format(
                    aligned[k], q, f'{args.seed}|{k}|{qi}'))
            all_rej.extend(rejected)
            print(f'  ev{k}: {len(kept)} kept, {len(rejected)} rejected')

    # Balance answer positions: rotate options so keys are evenly spread
    all_q = rebalance_answers(all_q, args.seed)

    with open(args.out, 'w') as f:
        json.dump(all_q, f, indent=2)

    from collections import Counter
    dist = Counter(q['conversations'][1]['value'] for q in all_q)
    print(f'\n{len(all_q)} questions -> {args.out}')
    print(f'answer distribution: {dict(sorted(dist.items()))}')
    if all_rej:
        print(f'\n{len(all_rej)} rejected by validator:')
        for r in all_rej[:10]:
            print(f'   {r}')


if __name__ == '__main__':
    main()
