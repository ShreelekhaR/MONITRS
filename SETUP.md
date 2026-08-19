# MONITRS v2 — environment setup

Uses [uv](https://docs.astral.sh/uv/) for dependency management. Single lockfile,
fast installs, no conda permission headaches on GCP Workbench.

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env      # or restart your shell
uv --version
```

## Create the environment

```bash
cd ~/MONITRS

# Base: article harvest + fact extraction + data + viz
uv sync

# Add evaluation metrics (BLEU / METEOR / ROUGE)
uv sync --extra eval

# Add training stack (torch, transformers, ms-swift) — big download
uv sync --extra train

# Everything
uv sync --all-extras
```

`uv sync` creates `.venv/` in the repo. Activate it or prefix commands with `uv run`.

```bash
source .venv/bin/activate
# or
uv run python pipeline/harvest_event.py --event 1346
```

## NLTK data (only if you installed the `eval` extra)

```bash
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"
```

## Environment variables

```bash
export GCP_PROJECT_ID=ai-sandbox-dev-f139       # Vertex AI for Gemini calls
export EE_PROJECT_ID=planet-earthengine-staging  # Earth Engine image download
export NOMINATIM_URL=http://nominatim.geocoder.internal:8080  # local geocoder
export ENABLE_LOCATION_QA=1                      # turn on location questions
```

## Pipeline quickstart

```bash
# 1. Harvest articles for a few events (search -> scrape -> verify -> extract)
uv run python pipeline/harvest_event.py --event 1346 5341 453

# 2. Inspect one event's harvest
cat Data/harvest/1346.json | python -m json.tool | head -60

# 3. Build fact timeseries across harvested events
uv run python pipeline/build_timeseries.py

# 4. Generate WHAT/WHERE/WHEN/HOW questions
uv run python pipeline/gen_qa_from_facts.py
```

## Notes

- `pipeline/` scripts import each other, so run them from the repo root
  (or set `PYTHONPATH=pipeline`).
- Torch is pinned to the cu121 index. On CPU-only machines, install with
  `uv sync` (base only) and skip the `train` extra.
