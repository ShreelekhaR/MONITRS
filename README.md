# MONITRS v2

Multi-temporal Observation of Natural disasters through Intelligent Text-image
Retrieval from Sentinel-2.

News articles are dense with facts about disasters — acreage burned, dates,
affected roads and rivers — and cheap LLMs read them reliably. MONITRS extracts
those facts as a dated timeseries per event, aligns them to Sentinel-2
acquisitions, and trains a VLM to recover them from imagery alone.

**No image analysis anywhere in dataset construction.** The LLM only does
reading comprehension over article text, so no model's opinion about the pixels
leaks into training labels. The VLM has to learn the image→fact mapping itself.

---

## 1. Setup

### 1.1 Clone

```bash
git clone https://github.com/ShreelekhaR/MONITRS.git
cd MONITRS
git checkout v2
```

### 1.2 Install uv

[uv](https://docs.astral.sh/uv/) manages the environment — one lockfile, fast
installs, and it avoids the conda permission problems on GCP Workbench.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env      # or restart your shell
uv --version
```

On Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 1.3 Create the environment

```bash
uv sync                    # base: harvest, fact extraction, imagery, viz
uv sync --extra eval       # + BLEU / METEOR / ROUGE
uv sync --extra train      # + torch, transformers, ms-swift (large)
uv sync --all-extras
```

This creates `.venv/` in the repo. Either activate it or prefix commands with
`uv run`:

```bash
source .venv/bin/activate
# or
uv run python pipeline/harvest_event.py --limit 20
```

If you installed the `eval` extra:

```bash
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"
```

<details>
<summary>Without uv (pip)</summary>

```bash
python -m pip install ddgs requests google-genai pandas pyarrow numpy \
                      earthengine-api pillow matplotlib
python -m pip install nltk rouge-score        # evaluation
```
</details>

### 1.4 Environment variables

```bash
export GCP_PROJECT_ID=ai-sandbox-dev-f139                     # Vertex AI (Gemini)
export EE_PROJECT_ID=planet-earthengine-staging               # Earth Engine
export NOMINATIM_URL=http://nominatim.geocoder.internal:8080  # local geocoder
export ENABLE_LOCATION_QA=1
```

`NOMINATIM_URL` matters: if the geocoder is unreachable, feature validation
passes everything through unvalidated rather than failing loudly. A public
fallback exists but is rate-limited to ~1 req/s.

### 1.5 Authenticate

```bash
gcloud auth application-default login
earthengine authenticate
```

---

## 2. Dataset construction

Run from the repo root — the `pipeline/` scripts import each other.

### 2.1 Harvest articles (search → scrape → verify → extract)

The expensive step. Per event it iterates: build event-anchored queries, search,
scrape page text and publication date, ask an LLM whether each article is about
*this specific incident*, extract facts as strict JSON, score coverage, and
generate targeted gap queries if anything is missing.

```bash
python pipeline/harvest_event.py --limit 200 --workers 4    # sample first
python pipeline/harvest_event.py --all --workers 4          # full run
```

Writes `Data/harvest/<event_id>.json` — resumable and inspectable per event.

The relevance gate is the reason this exists. Built from generic FEMA
declaration titles, only **35% of scraped articles were about the right event** —
an Alaska landslide sourced from Texas winter-storm coverage, Hurricane Delta
from Hurricane Sally articles.

Facts are also scope-checked (statewide totals rejected in favour of local
figures) and features spatially validated (geocoded, kept only if their OSM
bounding box intersects the image chip).

### 2.2 Download imagery

Fetches every clear Sentinel-2 acquisition across the event window, padded 30
days before and 45 after.

```bash
python pipeline/download_imagery.py --all --workers 4
```

Frame quality rejects true no-data (literal zero across R, G and B — *not*
darkness, since water renders near-black and most hurricane events are coastal),
near-total whiteout, and featureless frames.

### 2.3 Align facts to frames

```bash
python pipeline/align_frames.py
```

Sentinel-2 revisits every ~5 days, so frames rarely land on an article date.
Each acquisition gets a **bounded** slice of the event timeseries — lower bound
from the latest report at or before it, upper from the earliest after — plus a
phase label (pre-event / onset / during / post / recovery). Fire acreage is
monotonic, so the bounds are strict.

### 2.4 Measure visual signal

```bash
python pipeline/test_visual_signal.py
```

Compares pre-event against post-event frames using RGB proxies and checks the
direction against what the disaster type predicts. Emits `signal_strength` as a
**label, not a filter** — weak-signal events are exactly the cases no existing
model handles, and stratifying results by strength is how representation
learning gets demonstrated.

Transient signatures (snow, ice, standing water) are measured at the peak frame
rather than the post-event mean; averaging washes them out. Events with no frame
within 3 days of onset are labelled `unobserved_transient` — a sampling gap, not
evidence of undetectability.

### 2.5 Generate questions

```bash
python MONITRS_QA/split.py                          # county holdout, inspect first
python pipeline/gen_visual_mcq.py --n-per-event 8
```

Writes `Data/qa_visual_mcq_{train,test}.json`, partitioned by the county-level
split so no county appears on both sides.

Generation is gated on the signal verdict: events with no measurable change
don't get change-detection questions, since the answer would be "nothing
changed" — itself a shortcut.

---

## 3. Validation

Every question type must pass these before it ships.

### Blind baseline — can it be answered without the imagery?

```bash
python pipeline/test_blind_baseline.py --test-file Data/qa_visual_mcq.json
```

Feeds each question to an LLM with no images. Anything answerable blind is
testing priors or format artifacts. **Blind accuracy must sit at chance (0.25).**

This check found and drove out four distinct leaks:

| version | blind acc | leak removed |
|---|---|---|
| original | 0.507 | generated from speculative captions |
| v2 | 0.408 | non-visual attributes (wind mph), null-finding keys |
| v3 | 0.386 | cross-type distractors (flood/snow/burn as options) |
| v4 | 0.467 | *regression that exposed a thread-unsafe shuffle* |
| v5 | 0.323 | shuffle fixed — at chance |

The v4 regression is instructive: adding *more* question constraints made things
worse, which meant the leak wasn't in the questions. It was a shared
`random.Random` across a `ThreadPoolExecutor` biasing answer position toward `d`.

### Provenance visualization

```bash
python pipeline/visualize_aligned.py     # -> Data/provenance.html
```

Per event: verified sources with dates and acceptance reasons, the fact
timeseries, what got filtered and why, and each frame with its phase, numeric
bounds, and defensible claims.

---

## 4. Training

```bash
uv sync --extra train
bash Train/setup_qwen.sh
python Train/convert_qa_to_qwen.py --train-max 20000 --test-max 2000
bash Train/finetune_qwen.sh
```

Qwen2.5-VL-7B with LoRA (rank 16, frozen vision tower, ~40M trainable of 8.3B).
Fits on a single A100 40GB.

## 5. Evaluation

```bash
python Train/benchmark.py --models qwen-base,qwen-ft,gemini --out results.json
python Train/print_tables.py --ckpts
```

Report accuracy stratified by `signal_strength`, and exclude any type with zero
training examples — `MONITRS_QA/split.py` prints these explicitly.

---

## Known limitations

- **Extent coverage is type-dependent.** Fires and floods give clean acreage
  curves; tornadoes reported one figure ("400 yards" path width); snowstorms
  gave almost nothing per-county. Quantitative questions skew fire/flood.
- **Chip size vs feature coverage.** At ±0.05° (~11 km) many county features
  fall outside the frame. Widening recovers them at the cost of resolution.
- **Only RGB thumbnails.** NIR bands would give real NDVI/NDWI instead of
  proxies and would likely improve flood and hurricane detection.
- **Blind baseline is loosely bounded at small n.** At n=61 it drifts 0.32–0.39
  between runs. Re-measure on several hundred questions once harvest scales.

## Citation

If you use MONITRS or MONITRS-QA in your research, please cite:
